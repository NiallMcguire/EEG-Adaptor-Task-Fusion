from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np

from eeg_task_fusion.data.openbmi import CLASS_NAMES, LOADERS, Preprocessing, preprocess_rest, preprocess_trials, verify_api
from eeg_task_fusion.data.rest import RestCache


class OpenBMITests(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessing = Preprocessing(expected_channels=2, rest_seconds=10)
        self.sfreq = 500
        self.duration = 30
        self.eeg = np.random.default_rng(42).normal(scale=1e-6, size=(2, self.sfreq * self.duration))

    def raw(self, codes: list[int]) -> mne.io.RawArray:
        stim = np.zeros(self.eeg.shape[1])
        for index, code in enumerate(codes):
            stim[(2 + index * 6) * self.sfreq] = code
        info = mne.create_info(["C3", "C4", "STI 014"], self.sfreq, ["eeg", "eeg", "stim"])
        return mne.io.RawArray(np.vstack([self.eeg, stim]), info, verbose=False)

    def test_installed_loader_ids_and_rest_option(self) -> None:
        api = verify_api()
        self.assertEqual(api["subject_ids"], list(range(1, 55)))
        self.assertTrue(api["resting_state_supported"])
        self.assertFalse(api["rest_run_keys_verified_on_recording"])

    def test_all_paradigm_windows_and_labels(self) -> None:
        for paradigm, loader in LOADERS.items():
            event_id = loader().event_id
            codes = [event_id[name] for name in CLASS_NAMES[paradigm]]
            raw = self.raw(codes)
            trials = preprocess_trials(raw, event_id, paradigm, self.preprocessing)
            self.assertEqual(trials.X.shape, (len(codes), 2, round(self.preprocessing.window(paradigm) * 250)))
            np.testing.assert_array_equal(trials.y, np.arange(len(codes)))
            self.assertEqual(trials.channels, ("C3", "C4"))
            self.assertEqual(raw.info["sfreq"], self.sfreq)
            if paradigm == "ERP":
                self.assertEqual(trials.metadata["class_names"][1], "Target")

    def test_rest_excludes_stim_and_round_trips(self) -> None:
        record = preprocess_rest(self.raw([1, 2]), self.preprocessing)
        self.assertEqual(record.channels, ("C3", "C4"))
        self.assertEqual(record.covariance.shape, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            cache = RestCache(Path(directory), self.preprocessing.specification())
            cache.put(1, 1, record)
            self.assertTrue(cache.contains(1, 1))
            np.testing.assert_array_equal(cache.get(1, 1).covariance, record.covariance)
            with self.assertRaises(FileExistsError):
                cache.put(1, 1, record)

    def test_missing_events_and_channels_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_trials(self.raw([]), LOADERS["MI"]().event_id, "MI", self.preprocessing)
        with self.assertRaises(ValueError):
            preprocess_rest(self.raw([1, 2]), Preprocessing(rest_seconds=10))
        with self.assertRaises(ValueError):
            preprocess_rest(self.raw([1, 2]), Preprocessing(expected_channels=2, rest_seconds=60))


if __name__ == "__main__":
    unittest.main()