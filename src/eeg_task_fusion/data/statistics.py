"""Fit normalisation statistics exclusively through a validated training split."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eeg_task_fusion.data.cache import Dataset
from eeg_task_fusion.data.splits import Split


@dataclass(frozen=True)
class ChannelStandardiser:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    channels: tuple[str, ...]
    fit_subjects: tuple[int, ...]
    fit_pairs: tuple[tuple[int, str], ...]
    fit_session: int

    @classmethod
    def fit(cls, dataset: Dataset, split: Split, paradigm: str) -> "ChannelStandardiser":
        split.validate()
        pairs = tuple(pair for pair in split.train_pairs if pair[1] == paradigm)
        if not pairs:
            raise ValueError("No training pairs for this paradigm")
        count = 0
        mean = None
        squared_deviation = None
        channels = None
        for subject, name in pairs:
            trials = dataset.get(subject, name, split.train_session)
            trials.validate()
            if channels is not None and trials.channels != channels:
                raise ValueError("Channel order differs between recordings")
            channels = trials.channels
            values = np.asarray(trials.X, dtype=np.float64)
            batch_count = values.shape[0] * values.shape[2]
            batch_mean = values.mean(axis=(0, 2))
            batch_deviation = np.sum((values - batch_mean[None, :, None]) ** 2, axis=(0, 2))
            if mean is None:
                mean = batch_mean
                squared_deviation = batch_deviation
            else:
                delta = batch_mean - mean
                squared_deviation += batch_deviation + delta**2 * count * batch_count / (count + batch_count)
                mean += delta * batch_count / (count + batch_count)
            count += batch_count
        if mean is None or squared_deviation is None or channels is None:
            raise ValueError("No training data")
        scale = np.sqrt(squared_deviation / count)
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        return cls(mean, scale, channels, tuple(sorted({subject for subject, _ in pairs})), pairs, split.train_session)

    def transform(self, values: NDArray[np.float32], channels: tuple[str, ...]) -> NDArray[np.float32]:
        if values.ndim != 3 or channels != self.channels or values.shape[1] != len(channels):
            raise ValueError("Input shape or channel order differs from training")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite EEG")
        return ((values - self.mean[None, :, None]) / self.scale[None, :, None]).astype(np.float32)