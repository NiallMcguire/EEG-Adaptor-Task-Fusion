"""Cache extraction and reusable fold artefacts without model training."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from eeg_task_fusion.data.cache import ArrayCache, Dataset, Trials
from eeg_task_fusion.data.openbmi import LOADERS, OpenBMI, Preprocessing, preprocess_rest, preprocess_trials
from eeg_task_fusion.data.rest import RestCache, RestReference
from eeg_task_fusion.data.splits import PARADIGMS, Split
from eeg_task_fusion.data.statistics import ChannelStandardiser


def cache_openbmi(
    root: Path, preprocessing: Preprocessing, subjects: tuple[int, ...], sessions: tuple[int, ...],
    paradigms: tuple[str, ...],
) -> list[dict[str, Any]]:
    source = OpenBMI(preprocessing)
    specification = preprocessing.specification()
    tasks = ArrayCache(root, specification)
    rest = RestCache(root, specification)
    rows = []
    for subject in subjects:
        for session in sessions:
            for paradigm in paradigms:
                task_missing = not tasks.contains(subject, paradigm, session)
                rest_missing = paradigm == preprocessing.rest_paradigm and not rest.contains(subject, session)
                if task_missing or rest_missing:
                    runs = source.load_runs(subject, paradigm, session)
                    if task_missing:
                        trials = preprocess_trials(runs["1train"], LOADERS[paradigm]().event_id, paradigm, preprocessing)
                        tasks.put(subject, paradigm, session, trials)
                    if rest_missing:
                        rest.put(subject, session, preprocess_rest(runs["0preTrainRest"], preprocessing))
                trials = tasks.get(subject, paradigm, session)
                rows.append({"subject_id": subject, "paradigm_id": paradigm, "session": session, "trials": len(trials.y), "state": "created" if task_missing else "reused"})
    return rows


def prepare_fold(
    dataset: Dataset, resting: RestCache, split: Split, destination: Path,
    subjects: tuple[int, ...], sessions: tuple[int, ...],
) -> list[dict[str, Any]]:
    split.validate()
    if set(subjects) != set(split.train_subjects) | set(split.validation_subjects) | set(split.test_subjects):
        raise ValueError("Subject list does not agree with the split")
    if set(sessions) != {1, 2}:
        raise ValueError("Fold preparation requires both sessions")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "split.json").write_text(json.dumps(asdict(split), indent=2) + "\n", encoding="utf-8")
    covariances = {(subject, session): resting.get(subject, session) for subject in subjects for session in sessions}
    reference = RestReference.fit(covariances, split)
    provenance: dict[str, Any] = {
        "rest": {"fit_subjects": reference.fit_subjects, "fit_session": reference.fit_session, "channels": reference.channels, "metric": "riemann", "tsupdate": False},
        "normalisation": {},
    }
    np.save(destination / "rest_reference.npy", reference.tangent.reference_, allow_pickle=False)
    for paradigm in PARADIGMS:
        standardiser = ChannelStandardiser.fit(dataset, split, paradigm)
        if standardiser.channels != reference.channels:
            raise ValueError("Task and resting channel orders differ")
        np.savez(destination / f"normalisation_{paradigm}.npz", mean=standardiser.mean, scale=standardiser.scale)
        provenance["normalisation"][paradigm] = {
            "fit_subjects": standardiser.fit_subjects, "fit_pairs": standardiser.fit_pairs,
            "fit_session": standardiser.fit_session, "channels": standardiser.channels,
        }
    (destination / "statistics.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    feature_root = destination / "rest_features"
    feature_root.mkdir()
    rows = []
    for (subject, session), record in covariances.items():
        features = reference.transform(record)
        np.save(feature_root / f"subject-{subject:03d}_session-{session}.npy", features, allow_pickle=False)
        rows.append({"setting": split.setting, "fold": split.fold, "subject_id": subject, "session": session, "rest_feature_dim": len(features)})
    return rows


class FoldDataset:
    def __init__(self, dataset: Dataset, fold_directory: Path) -> None:
        self.dataset = dataset
        self.provenance = json.loads((fold_directory / "statistics.json").read_text(encoding="utf-8"))
        self.standardisers = {}
        for paradigm in PARADIGMS:
            info = self.provenance["normalisation"][paradigm]
            with np.load(fold_directory / f"normalisation_{paradigm}.npz", allow_pickle=False) as arrays:
                self.standardisers[paradigm] = ChannelStandardiser(
                    arrays["mean"].copy(), arrays["scale"].copy(), tuple(info["channels"]),
                    tuple(info["fit_subjects"]), tuple(tuple(pair) for pair in info["fit_pairs"]), info["fit_session"],
                )

    def get(self, subject_id: int, paradigm_id: str, session: int) -> Trials:
        trials = self.dataset.get(subject_id, paradigm_id, session)
        standardiser = self.standardisers[paradigm_id]
        return Trials(standardiser.transform(trials.X, trials.channels), trials.y, trials.channels, trials.metadata)