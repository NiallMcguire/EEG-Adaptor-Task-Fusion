"""OpenBMI offline labelled trials and explicitly selected pre-task rest."""

from dataclasses import asdict, dataclass
import inspect
from importlib.metadata import version
from typing import Any

import mne
from mne.io import BaseRaw
from moabb.datasets import Lee2019_ERP, Lee2019_MI, Lee2019_SSVEP
import numpy as np

from eeg_task_fusion.data.cache import Trials
from eeg_task_fusion.data.rest import RestCovariance, rest_covariance
from eeg_task_fusion.data.splits import PARADIGMS


LOADERS = {"MI": Lee2019_MI, "ERP": Lee2019_ERP, "SSVEP": Lee2019_SSVEP}
CLASS_NAMES = {
    "MI": ("left_hand", "right_hand"),
    "ERP": ("NonTarget", "Target"),
    "SSVEP": ("5.45", "6.67", "8.57", "12.0"),
}


@dataclass(frozen=True)
class Preprocessing:
    sfreq: float = 250.0
    low_hz: float = 0.5
    high_hz: float = 40.0
    mi_seconds: float = 4.0
    erp_seconds: float = 1.0
    ssvep_seconds: float = 4.0
    rest_seconds: float = 30.0
    rest_paradigm: str = "MI"
    expected_channels: int = 62

    def validate(self) -> None:
        if not 0 < self.low_hz < self.high_hz < self.sfreq / 2:
            raise ValueError("Invalid frequency range")
        if self.rest_paradigm not in PARADIGMS or self.expected_channels < 1:
            raise ValueError("Invalid resting paradigm or channel count")
        durations = (self.mi_seconds, self.erp_seconds, self.ssvep_seconds, self.rest_seconds)
        if any(not np.isfinite(duration) or duration <= 0 for duration in durations):
            raise ValueError("Durations must be finite and positive")
        if any(not np.isclose(duration * self.sfreq, round(duration * self.sfreq)) for duration in durations):
            raise ValueError("Durations must correspond to a whole number of samples")

    def window(self, paradigm: str) -> float:
        return {"MI": self.mi_seconds, "ERP": self.erp_seconds, "SSVEP": self.ssvep_seconds}[paradigm]

    def specification(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1, "dataset": "OpenBMI", "preprocessing": asdict(self),
            "task_run": "1train", "rest_run": "0preTrainRest", "units": "V",
            "normalisation": "none_until_fold_fit",
            "versions": {name: version(name) for name in ("moabb", "mne", "numpy", "scipy", "scikit-learn", "pyriemann")},
        }


def verify_api() -> dict[str, Any]:
    datasets = {name: loader(train_run=True, test_run=False, resting_state=True, sessions=[1, 2]) for name, loader in LOADERS.items()}
    subjects = datasets["MI"].subject_list
    if subjects != list(range(1, 55)) or any(dataset.subject_list != subjects for dataset in datasets.values()):
        raise RuntimeError("OpenBMI subject IDs do not align across paradigms")
    for dataset in datasets.values():
        parameters = inspect.signature(type(dataset).__init__).parameters
        if "resting_state" not in parameters or not dataset.resting_state:
            raise RuntimeError("Installed MOABB lacks the required resting-state API")
        if tuple(dataset.sessions) != (1, 2):
            raise RuntimeError("Unexpected OpenBMI session IDs")
    return {
        "moabb_version": version("moabb"), "subject_ids": subjects, "sessions": [1, 2],
        "events": {name: dataset.event_id for name, dataset in datasets.items()},
        "resting_state_supported": True, "selected_task_run": "1train",
        "selected_rest_run": "0preTrainRest", "rest_run_keys_verified_on_recording": False,
    }


def _eeg(raw: BaseRaw, preprocessing: Preprocessing) -> BaseRaw:
    eeg = raw.copy().pick("eeg").load_data()
    if len(eeg.ch_names) != preprocessing.expected_channels:
        raise ValueError(f"Expected {preprocessing.expected_channels} EEG channels, got {len(eeg.ch_names)}")
    return eeg


def preprocess_trials(raw: BaseRaw, event_id: dict[str, int], paradigm: str, preprocessing: Preprocessing) -> Trials:
    preprocessing.validate()
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    if not len(events):
        raise ValueError("No labelled task events found")
    eeg = _eeg(raw, preprocessing)
    eeg.filter(preprocessing.low_hz, preprocessing.high_hz, verbose=False)
    eeg, events = eeg.resample(preprocessing.sfreq, events=events, verbose=False)
    names = CLASS_NAMES[paradigm]
    codes = {name: event_id[name] for name in names}
    samples = round(preprocessing.window(paradigm) * preprocessing.sfreq)
    epochs = mne.Epochs(
        eeg, events, event_id=codes, tmin=0.0, tmax=(samples - 1) / preprocessing.sfreq,
        baseline=None, preload=True, reject_by_annotation=True, verbose=False,
    )
    values = epochs.get_data(copy=True).astype(np.float32)
    labels_by_code = {codes[name]: index for index, name in enumerate(names)}
    labels = np.array([labels_by_code[int(code)] for code in epochs.events[:, 2]], dtype=np.int64)
    if values.shape[-1] != samples or set(labels.tolist()) != set(range(len(names))):
        raise ValueError("Unexpected epoch length or missing task classes")
    trials = Trials(values, labels, tuple(eeg.ch_names), {
        "class_names": names, "sfreq": preprocessing.sfreq, "units": "V", "run": "1train",
        "events_found": len(events), "trials_retained": len(labels),
    })
    trials.validate()
    return trials


def preprocess_rest(raw: BaseRaw, preprocessing: Preprocessing) -> RestCovariance:
    preprocessing.validate()
    eeg = _eeg(raw, preprocessing)
    original_samples = round(preprocessing.rest_seconds * eeg.info["sfreq"])
    if eeg.n_times < original_samples:
        raise ValueError("Pre-task resting recording is shorter than the requested duration")
    eeg.crop(tmin=0, tmax=(original_samples - 1) / eeg.info["sfreq"])
    eeg.filter(preprocessing.low_hz, preprocessing.high_hz, verbose=False)
    eeg.resample(preprocessing.sfreq, verbose=False)
    record = RestCovariance(rest_covariance(eeg.get_data()), tuple(eeg.ch_names))
    record.validate()
    return record


class OpenBMI:
    def __init__(self, preprocessing: Preprocessing) -> None:
        preprocessing.validate()
        self.preprocessing = preprocessing
        self.api = verify_api()

    def load_runs(self, subject_id: int, paradigm_id: str, session: int) -> dict[str, BaseRaw]:
        if subject_id not in self.api["subject_ids"] or paradigm_id not in LOADERS or session not in (1, 2):
            raise ValueError("Invalid OpenBMI key")
        dataset = LOADERS[paradigm_id](train_run=True, test_run=False, resting_state=True, sessions=[session])
        sessions = dataset.get_data(subjects=[subject_id])[subject_id]
        if str(session) not in sessions:
            raise RuntimeError(f"Expected MOABB session key {session!r}, found {list(sessions)}")
        runs = sessions[str(session)]
        if "1train" not in runs or "0preTrainRest" not in runs:
            raise RuntimeError(f"Expected labelled/pre-task runs, found {list(runs)}")
        return runs

    def get(self, subject_id: int, paradigm_id: str, session: int) -> Trials:
        runs = self.load_runs(subject_id, paradigm_id, session)
        return preprocess_trials(runs["1train"], LOADERS[paradigm_id]().event_id, paradigm_id, self.preprocessing)