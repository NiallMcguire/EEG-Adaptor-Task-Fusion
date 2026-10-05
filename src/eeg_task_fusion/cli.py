"""Single-command, config-driven entry points for the data milestone."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any
import uuid

import numpy as np
import yaml

from eeg_task_fusion.data.cache import ArrayCache
from eeg_task_fusion.data.openbmi import Preprocessing, verify_api
from eeg_task_fusion.data.pipeline import cache_openbmi, prepare_fold
from eeg_task_fusion.data.rest import RestCache
from eeg_task_fusion.data.splits import PARADIGMS, generate_splits, load_splits, save_splits


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"experiment", "cache_root", "results_root", "split_file", "split_seed", "subjects", "sessions", "paradigms", "preprocessing"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Config must contain exactly {sorted(required)}")
    if not isinstance(config["experiment"], str) or not config["experiment"].replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid experiment name")
    if not config["subjects"] or len(set(config["subjects"])) != len(config["subjects"]) or any(type(subject) is not int or not 1 <= subject <= 54 for subject in config["subjects"]):
        raise ValueError("Subjects must be unique integer OpenBMI IDs in 1..54")
    if config["sessions"] != [1, 2] or set(config["paradigms"]) != set(PARADIGMS) or len(config["paradigms"]) != 3:
        raise ValueError("This protocol requires sessions [1, 2] and all three paradigms")
    if type(config["split_seed"]) is not int:
        raise ValueError("split_seed must be an integer")
    Preprocessing(**config["preprocessing"]).validate()
    return config


def _git_output(arguments: list[str]) -> str | None:
    process = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def _start_run(config: dict[str, Any], arguments: argparse.Namespace) -> Path:
    run_id = arguments.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    directory = Path(config["results_root"]) / config["experiment"] / run_id
    directory.mkdir(parents=True, exist_ok=False)
    resolved = {**config, "command": arguments.command, "seed": arguments.seed, "run_id": run_id, "download_allowed": arguments.download}
    (directory / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    metadata = {
        "git_hash": _git_output(["rev-parse", "HEAD"]), "git_status": _git_output(["status", "--porcelain"]),
        "seed": arguments.seed, "python": sys.version, "executable": sys.executable,
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "pyriemann", "moabb", "PyYAML")},
        "status": "running",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return directory


def _write_rows(directory: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (directory / "subjects.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _execute(config: dict[str, Any], arguments: argparse.Namespace, directory: Path) -> None:
    preprocessing = Preprocessing(**config["preprocessing"])
    subjects = tuple(config["subjects"])
    sessions = tuple(config["sessions"])
    if arguments.command == "verify-api":
        report = verify_api()
        (directory / "api.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _write_rows(directory, [{"subject_id": subject, "matching_paradigms": 3} for subject in report["subject_ids"]])
        print(json.dumps(report, indent=2))
    elif arguments.command == "splits":
        if arguments.seed != config["split_seed"]:
            raise ValueError("For split creation --seed must match the configured split_seed")
        splits = generate_splits(subjects, arguments.seed)
        save_splits(Path(config["split_file"]), splits)
        _write_rows(directory, [
            {"setting": split.setting, "fold": split.fold, "subject_id": subject, "group": group}
            for split in splits for group, ids in (("train", split.train_subjects), ("validation", split.validation_subjects), ("test", split.test_subjects)) for subject in ids
        ])
    elif arguments.command == "cache":
        if not arguments.download:
            raise ValueError("Caching may download large OpenBMI files; pass --download to explicitly permit it")
        rows = cache_openbmi(Path(config["cache_root"]), preprocessing, subjects, sessions, tuple(config["paradigms"]))
        _write_rows(directory, rows)
    else:
        splits = load_splits(Path(config["split_file"]))
        if len(splits) != 7 or any(split.seed != config["split_seed"] for split in splits):
            raise ValueError("Expected the saved partitioned grid and six subject folds with the configured split seed")
        if [(split.setting, split.fold) for split in splits] != [("unseen_pair", 0)] + [("unseen_subject", fold) for fold in range(6)]:
            raise ValueError("Unexpected split settings or fold IDs")
        specification = preprocessing.specification()
        tasks = ArrayCache(Path(config["cache_root"]), specification)
        resting = RestCache(Path(config["cache_root"]), specification)
        (directory / "cache_specification.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
        rows = []
        for split in splits:
            destination = directory / "folds" / split.setting / f"fold-{split.fold}"
            rows.extend(prepare_fold(tasks, resting, split, destination, subjects, sessions))
        _write_rows(directory, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify-api", "splits", "cache", "prepare"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args(argv)
    config = _read_config(arguments.config)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    directory = _start_run(config, arguments)
    try:
        _execute(config, arguments, directory)
    except Exception as error:
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Failed: {error}. Run record: {directory}", file=sys.stderr)
        return 1
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "completed"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Run record: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())