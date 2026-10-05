from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from eeg_task_fusion.data.splits import generate_splits, load_splits, save_splits


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = generate_splits(tuple(range(1, 55)), seed=42)

    def test_unseen_subject_folds(self) -> None:
        tested = []
        for split in self.splits[1:]:
            self.assertEqual(len(split.test_subjects), 9)
            self.assertEqual(len(split.validation_subjects), 9)
            self.assertEqual(len(split.train_subjects), 36)
            self.assertFalse(set(split.test_subjects) & set(split.train_subjects))
            self.assertFalse(set(split.validation_subjects) & set(split.train_subjects))
            tested.extend(split.test_subjects)
        self.assertEqual(sorted(tested), list(range(1, 55)))

    def test_unseen_pair_grid(self) -> None:
        split = self.splits[0]
        self.assertFalse(set(split.test_pairs) & set(split.train_pairs))
        for subject, paradigm in split.test_pairs:
            self.assertTrue(any(subject == candidate for candidate, _ in split.train_pairs))
            self.assertTrue(any(paradigm == candidate for _, candidate in split.train_pairs))
        self.assertEqual([sum(paradigm == name for _, paradigm in split.test_pairs) for name in ("MI", "ERP", "SSVEP")], [15, 15, 15])

    def test_deterministic_and_order_independent(self) -> None:
        self.assertEqual(self.splits, generate_splits(tuple(range(54, 0, -1)), seed=42))
        self.assertNotEqual(self.splits, generate_splits(tuple(range(1, 55)), seed=43))

    def test_saved_splits_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_splits(path, self.splits)
            self.assertEqual(self.splits, load_splits(path))
            with self.assertRaises(FileExistsError):
                save_splits(path, self.splits)

    def test_invalid_partitions_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.splits[1], train_subjects=self.splits[1].test_subjects).validate()
        with self.assertRaises(ValueError):
            replace(self.splits[0], train_pairs=self.splits[0].train_pairs + self.splits[0].test_pairs).validate()
        with self.assertRaises(ValueError):
            replace(self.splits[1], train_session=2).validate()

    def test_resting_reference_reads_only_training_subjects(self) -> None:
        import numpy as np
        from eeg_task_fusion.data.rest import RestCovariance, RestReference

        split = self.splits[1]
        channels = ("C3", "C4")

        class GuardedCovariances(dict):
            def __getitem__(inner_self, key: tuple[int, int]) -> RestCovariance:
                subject, session = key
                if subject not in split.train_subjects or session != 1:
                    raise AssertionError("Reference accessed non-training covariance")
                return RestCovariance(np.eye(2), channels)

        reference = RestReference.fit(GuardedCovariances(), split)
        np.testing.assert_allclose(reference.tangent.reference_, np.eye(2))
        self.assertEqual(reference.fit_subjects, split.train_subjects)


if __name__ == "__main__":
    unittest.main()