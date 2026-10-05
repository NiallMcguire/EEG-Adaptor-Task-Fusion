import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import mne
import yaml

from eeg_task_fusion.cli import main
from eeg_task_fusion.data.cache import ArrayCache, Trials
from eeg_task_fusion.data.openbmi import Preprocessing
from eeg_task_fusion.data.pipeline import FoldDataset, cache_openbmi, prepare_fold
from eeg_task_fusion.data.rest import RestCache, RestCovariance
from eeg_task_fusion.data.splits import generate_splits


class PipelineTests(unittest.TestCase):
    def test_cache_extraction_resumes_without_loading_recordings(self) -> None:
        preprocessing = Preprocessing(expected_channels=2, rest_seconds=10)
        sfreq = 250
        values = np.random.default_rng(42).normal(scale=1e-6, size=(3, 30 * sfreq))
        values[2] = 0
        for index, code in enumerate((1, 2, 3, 4)):
            values[2, (2 + index * 6) * sfreq] = code
        raw = mne.io.RawArray(values, mne.create_info(["C3", "C4", "STI 014"], sfreq, ["eeg", "eeg", "stim"]), verbose=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("eeg_task_fusion.data.pipeline.OpenBMI.load_runs", return_value={"1train": raw, "0preTrainRest": raw}) as load:
                rows = cache_openbmi(root, preprocessing, (1,), (1,), ("MI", "ERP", "SSVEP"))
                self.assertEqual(load.call_count, 3)
                self.assertEqual([row["state"] for row in rows], ["created"] * 3)
                load.reset_mock()
                rows = cache_openbmi(root, preprocessing, (1,), (1,), ("MI", "ERP", "SSVEP"))
                load.assert_not_called()
                self.assertEqual([row["state"] for row in rows], ["reused"] * 3)
            self.assertTrue(RestCache(root, preprocessing.specification()).contains(1, 1))

    def test_offline_cli_prepares_every_saved_fold(self) -> None:
        config = yaml.safe_load(Path("configs/openbmi.yaml").read_text())
        config["preprocessing"]["expected_channels"] = 2
        subjects = tuple(range(1, 13))
        channels = ("C3", "C4")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.update(subjects=list(subjects), results_root=str(root / "results"), split_file=str(root / "split.json"), cache_root=str(root / "cache"))
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            specification = Preprocessing(**config["preprocessing"]).specification()
            tasks = ArrayCache(root / "cache", specification)
            resting = RestCache(root / "cache", specification)
            for subject in subjects:
                for session in (1, 2):
                    resting.put(subject, session, RestCovariance(np.eye(2) * subject, channels))
                    for paradigm in ("MI", "ERP", "SSVEP"):
                        tasks.put(subject, paradigm, session, Trials(np.tile([1, 3], (2, 2, 1)).astype(np.float32), np.array([0, 1]), channels, {}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["splits", "--config", str(path), "--seed", "42", "--run-id", "splits"]), 0)
                self.assertEqual(main(["prepare", "--config", str(path), "--seed", "7", "--run-id", "prepared"]), 0)
            directory = root / "results" / "openbmi_data" / "prepared"
            self.assertEqual(len(list((directory / "folds").glob("*/*/statistics.json"))), 7)
            self.assertEqual(len(list((directory / "folds").glob("*/*/rest_features/*.npy"))), 7 * 12 * 2)
            self.assertEqual(json.loads((directory / "metadata.json").read_text())["seed"], 7)

    def test_fold_preparation_and_normalised_access(self) -> None:
        subjects = tuple(range(1, 13))
        split = generate_splits(subjects, 42)[0]
        channels = ("C3", "C4")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = ArrayCache(root / "cache", {"test": True})
            resting = RestCache(root / "cache", {"test": True})
            for subject in subjects:
                for session in (1, 2):
                    resting.put(subject, session, RestCovariance(np.eye(2) * (subject + session), channels))
            for subject, paradigm in split.train_pairs:
                tasks.put(subject, paradigm, 1, Trials(np.tile([1, 3], (2, 2, 1)).astype(np.float32), np.array([0, 1]), channels, {}))
            rows = prepare_fold(tasks, resting, split, root / "fold", subjects, (1, 2))
            self.assertEqual(len(rows), 24)
            provenance = json.loads((root / "fold" / "statistics.json").read_text())
            self.assertEqual(provenance["rest"]["fit_subjects"], list(split.train_subjects))
            self.assertEqual(provenance["normalisation"]["MI"]["fit_pairs"], [list(pair) for pair in split.train_pairs if pair[1] == "MI"])
            self.assertEqual(len(list((root / "fold" / "rest_features").glob("*.npy"))), 24)
            dataset = FoldDataset(tasks, root / "fold")
            subject, paradigm = split.train_pairs[0]
            normalised = dataset.get(subject, paradigm, 1)
            np.testing.assert_array_equal(normalised.X, np.tile([-1, 1], (2, 2, 1)))
            with self.assertRaises(FileExistsError):
                prepare_fold(tasks, resting, split, root / "fold", subjects, (1, 2))

    def test_cli_provenance_splits_and_download_guard(self) -> None:
        config = yaml.safe_load(Path("configs/openbmi.yaml").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config.update(subjects=list(range(1, 13)), results_root=str(root / "results"), split_file=str(root / "split.json"))
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["splits", "--config", str(path), "--seed", "42", "--run-id", "splits"]), 0)
                self.assertEqual(main(["splits", "--config", str(path), "--seed", "42", "--run-id", "repeat"]), 1)
                with patch("eeg_task_fusion.cli.cache_openbmi") as download:
                    self.assertEqual(main(["cache", "--config", str(path), "--seed", "42", "--run-id", "guard"]), 1)
                    download.assert_not_called()
            directory = root / "results" / "openbmi_data" / "splits"
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["seed"], 42)
            self.assertTrue((directory / "config.yaml").exists())
            self.assertTrue((directory / "subjects.csv").exists())
            metadata = json.loads((directory.parent / "guard" / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")


if __name__ == "__main__":
    unittest.main()