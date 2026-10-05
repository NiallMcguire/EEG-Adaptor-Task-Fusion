"""Versioned, non-pickled arrays indexed by subject, paradigm and session."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trials:
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    channels: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.X.ndim != 3 or self.y.shape != (len(self.X),) or not len(self.X):
            raise ValueError("Expected non-empty trials shaped (trials, channels, time) and one label per trial")
        if self.X.shape[1] != len(self.channels) or len(set(self.channels)) != len(self.channels):
            raise ValueError("Invalid channel names")
        if self.X.dtype != np.float32 or self.y.dtype != np.int64:
            raise ValueError("Cache requires float32 EEG and int64 labels")
        if not np.isfinite(self.X).all() or np.any(self.y < 0):
            raise ValueError("Non-finite EEG or invalid labels")


class Dataset(Protocol):
    def get(self, subject_id: int, paradigm_id: str, session: int) -> Trials: ...


def fingerprint(specification: dict[str, Any]) -> str:
    encoded = json.dumps(specification, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


class ArrayCache:
    def __init__(self, root: Path, specification: dict[str, Any]) -> None:
        self.specification = specification
        self.root = root / fingerprint(specification)

    def _path(self, subject_id: int, paradigm_id: str, session: int) -> Path:
        if subject_id < 1 or session < 1 or not paradigm_id.isalnum():
            raise ValueError("Invalid cache key")
        return self.root / f"subject-{subject_id:03d}" / paradigm_id / f"session-{session}"

    def contains(self, subject_id: int, paradigm_id: str, session: int) -> bool:
        return (self._path(subject_id, paradigm_id, session) / "manifest.json").is_file()

    def put(self, subject_id: int, paradigm_id: str, session: int, trials: Trials) -> None:
        trials.validate()
        destination = self._path(subject_id, paradigm_id, session)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "record"
            staging.mkdir()
            np.save(staging / "X.npy", trials.X, allow_pickle=False)
            np.save(staging / "y.npy", trials.y, allow_pickle=False)
            manifest = {
                "schema_version": 1, "subject_id": subject_id, "paradigm_id": paradigm_id,
                "session": session, "channels": trials.channels,
                "specification": self.specification, "metadata": trials.metadata,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.rename(staging, destination)

    def get(self, subject_id: int, paradigm_id: str, session: int) -> Trials:
        path = self._path(subject_id, paradigm_id, session)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["specification"] != self.specification:
            raise ValueError("Cache specification mismatch")
        if (manifest["subject_id"], manifest["paradigm_id"], manifest["session"]) != (subject_id, paradigm_id, session):
            raise ValueError("Cache key mismatch")
        trials = Trials(
            np.load(path / "X.npy", mmap_mode="r", allow_pickle=False),
            np.load(path / "y.npy", mmap_mode="r", allow_pickle=False),
            tuple(manifest["channels"]), manifest["metadata"],
        )
        trials.validate()
        return trials