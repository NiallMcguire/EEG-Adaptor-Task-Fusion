"""OAS covariance and training-subject-only Riemannian tangent features."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from pyriemann.tangentspace import TangentSpace
from sklearn.covariance import OAS

from eeg_task_fusion.data.cache import fingerprint
from eeg_task_fusion.data.splits import Split


def rest_covariance(recording: NDArray[np.floating]) -> NDArray[np.float64]:
    if recording.ndim != 2 or recording.shape[1] < 2 or not np.isfinite(recording).all():
        raise ValueError("Expected finite resting EEG shaped (channels, samples)")
    covariance = OAS().fit(np.asarray(recording.T, dtype=np.float64)).covariance_
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Resting covariance is not positive definite")
    return covariance


@dataclass(frozen=True)
class RestCovariance:
    covariance: NDArray[np.float64]
    channels: tuple[str, ...]

    def validate(self) -> None:
        if self.covariance.shape != (len(self.channels), len(self.channels)) or len(set(self.channels)) != len(self.channels):
            raise ValueError("Covariance and channel names do not agree")
        if not np.isfinite(self.covariance).all() or not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance must be finite and symmetric")
        if np.linalg.eigvalsh(self.covariance).min() <= 0:
            raise ValueError("Covariance must be positive definite")


@dataclass(frozen=True)
class RestReference:
    tangent: TangentSpace
    channels: tuple[str, ...]
    fit_subjects: tuple[int, ...]
    fit_session: int

    @classmethod
    def fit(cls, covariances: Mapping[tuple[int, int], RestCovariance], split: Split) -> "RestReference":
        split.validate()
        records = [covariances[(subject, split.train_session)] for subject in split.train_subjects]
        for record in records:
            record.validate()
            if record.channels != records[0].channels:
                raise ValueError("Resting channel order differs between subjects")
        tangent = TangentSpace(metric="riemann", tsupdate=False)
        tangent.fit(np.stack([record.covariance for record in records]))
        return cls(tangent, records[0].channels, split.train_subjects, split.train_session)

    def transform(self, record: RestCovariance) -> NDArray[np.float32]:
        record.validate()
        if record.channels != self.channels:
            raise ValueError("Resting channel order differs from the reference")
        return self.tangent.transform(record.covariance[None])[0].astype(np.float32)


class RestCache:
    def __init__(self, root: Path, specification: dict[str, Any]) -> None:
        self.specification = specification
        self.root = root / fingerprint(specification) / "rest"

    def _path(self, subject_id: int, session: int) -> Path:
        if subject_id < 1 or session < 1:
            raise ValueError("Invalid resting cache key")
        return self.root / f"subject-{subject_id:03d}" / f"session-{session}"

    def contains(self, subject_id: int, session: int) -> bool:
        return (self._path(subject_id, session) / "manifest.json").is_file()

    def put(self, subject_id: int, session: int, record: RestCovariance) -> None:
        record.validate()
        destination = self._path(subject_id, session)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "record"
            staging.mkdir()
            np.save(staging / "covariance.npy", record.covariance, allow_pickle=False)
            manifest = {
                "schema_version": 1, "subject_id": subject_id, "session": session,
                "channels": record.channels, "specification": self.specification,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.rename(staging, destination)

    def get(self, subject_id: int, session: int) -> RestCovariance:
        path = self._path(subject_id, session)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["specification"] != self.specification:
            raise ValueError("Resting cache specification mismatch")
        if (manifest["subject_id"], manifest["session"]) != (subject_id, session):
            raise ValueError("Resting cache key mismatch")
        record = RestCovariance(np.load(path / "covariance.npy", allow_pickle=False), tuple(manifest["channels"]))
        record.validate()
        return record