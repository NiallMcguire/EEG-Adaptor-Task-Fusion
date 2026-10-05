"""Explicit subject and pair partitions, independent of EEG downloads."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


PARADIGMS = ("MI", "ERP", "SSVEP")
Pair = tuple[int, str]


@dataclass(frozen=True)
class Split:
    setting: str
    seed: int
    fold: int
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    train_pairs: tuple[Pair, ...]
    validation_pairs: tuple[Pair, ...]
    test_pairs: tuple[Pair, ...]
    train_session: int = 1
    validation_session: int = 1
    test_session: int = 2

    def validate(self) -> None:
        if self.setting not in {"unseen_pair", "unseen_subject"}:
            raise ValueError("Unknown evaluation setting")
        train = set(self.train_subjects)
        validation = set(self.validation_subjects)
        test = set(self.test_subjects)
        if not train or not validation or not test:
            raise ValueError("All subject groups must be non-empty")
        for group in (self.train_subjects, self.validation_subjects, self.test_subjects):
            if len(set(group)) != len(group) or any(subject < 1 for subject in group):
                raise ValueError("Subject IDs must be unique positive integers")
        if validation & (train | test):
            raise ValueError("Validation subjects overlap another group")
        if self.setting == "unseen_subject" and train & test:
            raise ValueError("Test subjects occur in training")
        if self.setting == "unseen_pair" and train != test:
            raise ValueError("Unseen-pair subjects must be seen in training")
        if (self.train_session, self.validation_session, self.test_session) != (1, 1, 2):
            raise ValueError("The protocol requires training/validation session 1 and test session 2")
        groups = (
            (self.train_pairs, train),
            (self.validation_pairs, validation),
            (self.test_pairs, test),
        )
        for pairs, subjects in groups:
            if not pairs or len(set(pairs)) != len(pairs):
                raise ValueError("Pair groups must be non-empty and unique")
            if {subject for subject, _ in pairs} != subjects:
                raise ValueError("Pair subjects do not match their declared subject group")
            if any(paradigm not in PARADIGMS for _, paradigm in pairs):
                raise ValueError("Unknown paradigm")
        if set(self.train_pairs) & set(self.test_pairs):
            raise ValueError("Held-out pairs occur in training")
        if self.setting == "unseen_pair":
            seen_paradigms = {paradigm for _, paradigm in self.train_pairs}
            if any(paradigm not in seen_paradigms for _, paradigm in self.test_pairs):
                raise ValueError("Held-out paradigm is never seen in training")
            if any(sum(subject == candidate for candidate, _ in self.test_pairs) != 1 for subject in train):
                raise ValueError("Each training subject must lose exactly one paradigm")
            expected = {(subject, paradigm) for subject in train for paradigm in PARADIGMS}
            if set(self.train_pairs) | set(self.test_pairs) != expected:
                raise ValueError("The partitioned grid is incomplete")
            counts = [sum(paradigm == candidate for _, candidate in self.test_pairs) for paradigm in PARADIGMS]
            if max(counts) - min(counts) > 1:
                raise ValueError("Held-out paradigms are not balanced")
        else:
            for pairs, subjects in groups:
                if set(pairs) != {(subject, paradigm) for subject in subjects for paradigm in PARADIGMS}:
                    raise ValueError("Unseen-subject groups must contain all paradigms")


def _grid(subjects: tuple[int, ...]) -> tuple[Pair, ...]:
    return tuple((subject, paradigm) for subject in subjects for paradigm in PARADIGMS)


def generate_splits(subjects: tuple[int, ...], seed: int, folds: int = 6) -> tuple[Split, ...]:
    if len(set(subjects)) != len(subjects) or any(subject < 1 for subject in subjects):
        raise ValueError("Subject IDs must be unique positive integers")
    if folds < 3 or len(subjects) < folds * 2 or len(subjects) % folds:
        raise ValueError("Use at least three equally sized folds with at least two subjects each")
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    size = len(subjects) // folds
    groups = [tuple(sorted(shuffled[index * size : (index + 1) * size])) for index in range(folds)]
    validation = groups[0]
    train = tuple(sorted(set(subjects) - set(validation)))
    order = list(train)
    random.Random(seed + 1).shuffle(order)
    held_out = tuple(sorted((subject, PARADIGMS[index % 3]) for index, subject in enumerate(order)))
    unseen_pair = Split(
        "unseen_pair", seed, 0, train, validation, train,
        tuple(pair for pair in _grid(train) if pair not in set(held_out)),
        _grid(validation), held_out,
    )
    splits = [unseen_pair]
    for fold, test in enumerate(groups):
        validation = groups[(fold + 1) % folds]
        train = tuple(sorted(set(subjects) - set(test) - set(validation)))
        splits.append(Split("unseen_subject", seed, fold, train, validation, test, _grid(train), _grid(validation), _grid(test)))
    for split in splits:
        split.validate()
    return tuple(splits)


def save_splits(path: Path, splits: tuple[Split, ...]) -> None:
    for split in splits:
        split.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "splits": [asdict(split) for split in splits]}, stream, indent=2)
        stream.write("\n")


def load_splits(path: Path) -> tuple[Split, ...]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported split schema")
    splits = []
    for record in payload["splits"]:
        for field in ("train_subjects", "validation_subjects", "test_subjects"):
            record[field] = tuple(record[field])
        for field in ("train_pairs", "validation_pairs", "test_pairs"):
            record[field] = tuple(tuple(pair) for pair in record[field])
        split = Split(**record)
        split.validate()
        splits.append(split)
    return tuple(splits)