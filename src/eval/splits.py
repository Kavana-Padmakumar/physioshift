"""
src/eval/splits.py

22 Aug 2026 -- Phase 3, Day 3 (Member 1)

Reusable Split A/B/C evaluation logic so every model variant calls the
SAME split code, avoiding leakage/inconsistency bugs across experiments.

Split A (in-device):      80/20 train/test WITHIN each dataset.
                           Easiest split -- tests whether the model works
                           at all, not whether it generalizes.

Split B (unseen device):  Train on D1-D4, test ONLY on D5 (PPG-DaLiA).
                           Tests generalization to a device never seen
                           during training.

Split C (clinical->consumer): Train ONLY on clinical-grade devices
                           (D1, D2), test ONLY on consumer/wearable
                           devices (D3, D4, D5). The hardest and most
                           realistic split -- simulates a model trained
                           on hospital ECG machines being deployed on a
                           consumer smartwatch.

Each split function returns windows tagged with their domain_id, NOT a
resolved class label -- the actual classification target (domain
identity, arrhythmia label, etc.) is decided by whatever training
script consumes this split, keeping this module reusable across future
model variants.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np


CLINICAL_DOMAINS = ["D1_MITBIH_open_chest", "D2_PTBXL_open_clinical"]
CONSUMER_DOMAINS = [
    "D3_MultisitePPG_open_wearable",
    "D4_WESAD_open_chest",
    "D5_PPGDaLiA_open_wrist",
]
UNSEEN_DEVICE_DOMAIN = "D5_PPGDaLiA_open_wrist"

ALL_DOMAINS = CLINICAL_DOMAINS + CONSUMER_DOMAINS


class SplitResult(NamedTuple):
    """Container for one train/test split."""
    train_windows: np.ndarray
    train_domain_ids: np.ndarray
    test_windows: np.ndarray
    test_domain_ids: np.ndarray


def load_harmonized_cache(cache_dir: str = "data/harmonized_cache") -> dict[str, np.ndarray]:
    """
    Loads every .npy file in the harmonized cache directory into a dict.

    Parameters
    ----------
    cache_dir : str, default="data/harmonized_cache"
        Directory containing {domain_id}.npy files produced by
        PhysioShiftDataset.save_cache().

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of domain_id -> windowed array, shape (n_windows, window_size).
    """
    cache = {}
    for fname in os.listdir(cache_dir):
        if fname.endswith(".npy"):
            domain_id = fname[:-4]
            cache[domain_id] = np.load(os.path.join(cache_dir, fname))
    return cache


def split_a_in_device(
    cache: dict[str, np.ndarray],
    test_frac: float = 0.2,
    seed: int = 0,
) -> SplitResult:
    """
    Split A: in-device. For each domain, randomly holds out test_frac of
    its windows for testing; the rest go to train. Train and test cover
    the SAME domains -- this is the easy baseline split.

    Parameters
    ----------
    cache : dict[str, np.ndarray]
        domain_id -> windowed array, from load_harmonized_cache().
    test_frac : float, default=0.2
        Fraction of each domain's windows held out for test.
    seed : int, default=0
        Random seed for reproducible shuffling.

    Returns
    -------
    SplitResult
        Train/test windows and their domain_ids. Domains overlap between
        train and test (by design -- that's what "in-device" means), but
        no individual WINDOW appears in both.
    """
    rng = np.random.default_rng(seed)
    train_windows, train_ids, test_windows, test_ids = [], [], [], []

    for domain_id, windows in cache.items():
        n = windows.shape[0]
        if n == 0:
            continue
        idx = rng.permutation(n)
        n_test = max(1, int(round(n * test_frac))) if n > 1 else 0
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        if len(train_idx) > 0:
            train_windows.append(windows[train_idx])
            train_ids.extend([domain_id] * len(train_idx))
        if len(test_idx) > 0:
            test_windows.append(windows[test_idx])
            test_ids.extend([domain_id] * len(test_idx))

    return SplitResult(
        train_windows=np.concatenate(train_windows, axis=0) if train_windows else np.empty((0,)),
        train_domain_ids=np.array(train_ids),
        test_windows=np.concatenate(test_windows, axis=0) if test_windows else np.empty((0,)),
        test_domain_ids=np.array(test_ids),
    )


def split_b_unseen_device(cache: dict[str, np.ndarray]) -> SplitResult:
    """
    Split B: unseen device. Trains on every domain EXCEPT the unseen
    device domain (D5, PPG-DaLiA); tests ONLY on that unseen domain.
    Tests generalization to a device/subject population never seen
    during training.

    Parameters
    ----------
    cache : dict[str, np.ndarray]
        domain_id -> windowed array, from load_harmonized_cache().

    Returns
    -------
    SplitResult
        Train covers all domains except UNSEEN_DEVICE_DOMAIN; test
        covers ONLY UNSEEN_DEVICE_DOMAIN. Zero domain overlap between
        train and test by construction.
    """
    train_windows, train_ids, test_windows, test_ids = [], [], [], []

    for domain_id, windows in cache.items():
        if windows.shape[0] == 0:
            continue
        if domain_id == UNSEEN_DEVICE_DOMAIN:
            test_windows.append(windows)
            test_ids.extend([domain_id] * windows.shape[0])
        else:
            train_windows.append(windows)
            train_ids.extend([domain_id] * windows.shape[0])

    return SplitResult(
        train_windows=np.concatenate(train_windows, axis=0) if train_windows else np.empty((0,)),
        train_domain_ids=np.array(train_ids),
        test_windows=np.concatenate(test_windows, axis=0) if test_windows else np.empty((0,)),
        test_domain_ids=np.array(test_ids),
    )


def split_c_clinical_to_consumer(cache: dict[str, np.ndarray]) -> SplitResult:
    """
    Split C: clinical->consumer. Trains ONLY on clinical-grade domains
    (D1, D2); tests ONLY on consumer/wearable domains (D3, D4, D5).
    The hardest and most realistic split -- simulates a model trained
    on hospital-grade equipment being deployed on consumer wearables,
    with entirely different noise characteristics and signal quality.

    Parameters
    ----------
    cache : dict[str, np.ndarray]
        domain_id -> windowed array, from load_harmonized_cache().

    Returns
    -------
    SplitResult
        Train covers ONLY CLINICAL_DOMAINS; test covers ONLY
        CONSUMER_DOMAINS. Zero domain overlap between train and test
        by construction.
    """
    train_windows, train_ids, test_windows, test_ids = [], [], [], []

    for domain_id, windows in cache.items():
        if windows.shape[0] == 0:
            continue
        if domain_id in CLINICAL_DOMAINS:
            train_windows.append(windows)
            train_ids.extend([domain_id] * windows.shape[0])
        elif domain_id in CONSUMER_DOMAINS:
            test_windows.append(windows)
            test_ids.extend([domain_id] * windows.shape[0])
        # domains not in either list (e.g. future MIMIC) are excluded from Split C

    return SplitResult(
        train_windows=np.concatenate(train_windows, axis=0) if train_windows else np.empty((0,)),
        train_domain_ids=np.array(train_ids),
        test_windows=np.concatenate(test_windows, axis=0) if test_windows else np.empty((0,)),
        test_domain_ids=np.array(test_ids),
    )


def verify_no_leakage(split: SplitResult, split_name: str, domain_level: bool = True) -> bool:
    """
    Verifies zero train/test leakage for a split.

    Parameters
    ----------
    split : SplitResult
        The split to check.
    split_name : str
        Name for logging, e.g. "Split B".
    domain_level : bool, default=True
        If True, checks that NO domain_id appears in both train and test
        (correct check for Split B/C, which split by whole domains).
        If False (Split A), domains legitimately overlap, so this check
        is skipped -- Split A's leakage check happens at window-count
        level instead (handled by construction: train/test indices are
        disjoint within each domain).

    Returns
    -------
    bool
        True if no leakage detected, False otherwise. Prints a message
        either way.
    """
    if not domain_level:
        print(f"[{split_name}] Window-level split (leakage prevented by disjoint indices per domain).")
        return True

    train_domains = set(split.train_domain_ids)
    test_domains = set(split.test_domain_ids)
    overlap = train_domains & test_domains

    if overlap:
        print(f"[{split_name}] LEAKAGE DETECTED: domains {overlap} appear in both train and test!")
        return False
    else:
        print(f"[{split_name}] OK -- zero domain overlap between train and test.")
        return True


def print_split_summary(cache: dict[str, np.ndarray]) -> None:
    """
    Runs all three splits and prints a summary table of window counts
    per split, plus a leakage check for each. This is the deliverable
    verification table for 22 Aug.

    Parameters
    ----------
    cache : dict[str, np.ndarray]
        domain_id -> windowed array, from load_harmonized_cache().

    Returns
    -------
    None
        Prints directly to stdout.
    """
    a = split_a_in_device(cache)
    b = split_b_unseen_device(cache)
    c = split_c_clinical_to_consumer(cache)

    print("\n" + "=" * 70)
    print("  SPLIT A/B/C SUMMARY -- window counts and leakage check")
    print("=" * 70)

    for name, split, domain_level in [("Split A (in-device)", a, False),
                                        ("Split B (unseen device D5)", b, True),
                                        ("Split C (clinical->consumer)", c, True)]:
        print(f"\n{name}")
        print(f"  Train: {split.train_windows.shape[0]:6} windows across "
              f"{len(set(split.train_domain_ids)):2} domain(s)")
        print(f"  Test:  {split.test_windows.shape[0]:6} windows across "
              f"{len(set(split.test_domain_ids)):2} domain(s)")
        verify_no_leakage(split, name, domain_level=domain_level)


if __name__ == "__main__":
    cache = load_harmonized_cache()
    print("Loaded domains:", list(cache.keys()))
    print_split_summary(cache)