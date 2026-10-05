from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_task_fusion.data.cache import ArrayCache, Trials
from eeg_task_fusion.data.rest import RestCovariance, RestReference, rest_covariance
from eeg_task_fusion.data.splits import generate_splits
from eeg_task_fusion.data.statistics import ChannelStandardiser


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = generate_splits(tuple(range(1, 13)), 42)[1]
        self.channels = ("C3", "C4")

    def test_cache_round_trip_and_spec_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = ArrayCache(Path(directory), {"sfreq": 250})
            trials = Trials(np.ones((3, 2, 20), dtype=np.float32), np.array([0, 1, 0], dtype=np.int64), self.channels, {"units": "V"})
            cache.put(1, "MI", 1, trials)
            loaded = cache.get(1, "MI", 1)
            np.testing.assert_array_equal(loaded.X, trials.X)
            np.testing.assert_array_equal(loaded.y, trials.y)
            self.assertEqual(loaded.channels, self.channels)
            self.assertFalse(ArrayCache(Path(directory), {"sfreq": 200}).contains(1, "MI", 1))
            with self.assertRaises(FileExistsError):
                cache.put(1, "MI", 1, trials)

    def test_normalisation_only_reads_training_pairs_and_session(self) -> None:
        calls = []
        channels = self.channels

        class GuardedDataset:
            def get(inner_self, subject_id: int, paradigm_id: str, session: int) -> Trials:
                calls.append((subject_id, paradigm_id, session))
                values = np.broadcast_to(np.array([1, 3], dtype=np.float32), (2, 2, 2)).copy()
                return Trials(values, np.array([0, 1], dtype=np.int64), channels, {})

        standardiser = ChannelStandardiser.fit(GuardedDataset(), self.split, "MI")
        self.assertEqual(set(calls), {(subject, "MI", 1) for subject in self.split.train_subjects})
        np.testing.assert_array_equal(standardiser.mean, [2, 2])
        np.testing.assert_array_equal(standardiser.scale, [1, 1])
        transformed = standardiser.transform(np.array([[[1, 3], [1, 3]]], dtype=np.float32), channels)
        np.testing.assert_array_equal(transformed, [[[-1, 1], [-1, 1]]])
        with self.assertRaises(ValueError):
            standardiser.transform(np.zeros((1, 2, 2), dtype=np.float32), tuple(reversed(channels)))

    def test_rest_reference_uses_training_subjects_only(self) -> None:
        records = {(subject, 1): RestCovariance(np.eye(2), self.channels) for subject in self.split.train_subjects}
        for subject in self.split.validation_subjects + self.split.test_subjects:
            records[(subject, 1)] = RestCovariance(np.eye(2) * 1e6, self.channels)
        for subject in self.split.train_subjects:
            records[(subject, 2)] = RestCovariance(np.eye(2) * 1e9, self.channels)
        reference = RestReference.fit(records, self.split)
        self.assertEqual(reference.fit_subjects, self.split.train_subjects)
        self.assertEqual(reference.fit_session, 1)
        np.testing.assert_allclose(reference.tangent.reference_, np.eye(2), atol=1e-12)
        before = reference.tangent.reference_.copy()
        features = reference.transform(records[(self.split.test_subjects[0], 1)])
        self.assertEqual(features.shape, (3,))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(reference.tangent.reference_, before)

    def test_oas_covariance_is_positive_definite(self) -> None:
        recording = np.random.default_rng(42).normal(size=(2, 250))
        covariance = rest_covariance(recording)
        self.assertEqual(covariance.shape, (2, 2))
        self.assertTrue(np.linalg.eigvalsh(covariance).min() > 0)
        with self.assertRaises(ValueError):
            rest_covariance(np.zeros((2, 250)))

    def test_missing_training_rest_is_not_replaced_by_test_data(self) -> None:
        records = {(subject, 2): RestCovariance(np.eye(2), self.channels) for subject in self.split.train_subjects}
        with self.assertRaises(KeyError):
            RestReference.fit(records, self.split)


if __name__ == "__main__":
    unittest.main()