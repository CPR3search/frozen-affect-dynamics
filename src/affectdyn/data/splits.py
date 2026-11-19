"""Leave-one-subject-out splits with an overlap audit."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Fold:
    fold_id: int
    test_subject: str
    train_idx: np.ndarray
    test_idx: np.ndarray


class SubjectOverlapError(RuntimeError):
    pass


def loso_folds(subject_ids: np.ndarray) -> list[Fold]:
    """One fold per subject; that subject is the test set, the rest are train."""
    subjects = np.unique(subject_ids)
    folds: list[Fold] = []
    for k, s in enumerate(sorted(subjects.tolist())):
        test_idx = np.where(subject_ids == s)[0]
        train_idx = np.where(subject_ids != s)[0]
        folds.append(Fold(k, str(s), train_idx, test_idx))
    audit_folds(folds, subject_ids)
    return folds


def audit_folds(folds: list[Fold], subject_ids: np.ndarray) -> None:
    """Hard fail on any subject overlap or index leakage between train and test."""
    for f in folds:
        tr_subj = set(subject_ids[f.train_idx].tolist())
        te_subj = set(subject_ids[f.test_idx].tolist())
        overlap = tr_subj & te_subj
        if overlap:
            raise SubjectOverlapError(
                f"fold {f.fold_id}: subjects in both train and test: {sorted(overlap)}")
        if np.intersect1d(f.train_idx, f.test_idx).size:
            raise SubjectOverlapError(f"fold {f.fold_id}: index overlap between train and test")
        if set(f.train_idx.tolist()) | set(f.test_idx.tolist()) != set(range(len(subject_ids))):
            raise SubjectOverlapError(f"fold {f.fold_id}: folds do not partition the dataset")
    print(f"[splits] LOSO audit passed: {len(folds)} folds, zero subject overlap.")


def assert_window_safety(window_subject: np.ndarray, window_sequence: np.ndarray,
                         folds: list[Fold]) -> None:
    """Optional utility, not invoked by the released pipeline. Reject sequences shared across train and
    test; sequence identifiers must be globally unique. window_subject is currently unused.
    """
    for f in folds:
        tr_seq = set(window_sequence[f.train_idx].tolist())
        te_seq = set(window_sequence[f.test_idx].tolist())
        if tr_seq & te_seq:
            raise SubjectOverlapError(
                f"fold {f.fold_id}: sequence(s) span train and test: {sorted(tr_seq & te_seq)[:5]}")
    print("[splits] window-safety audit passed: no sequence spans a fold boundary.")
