"""
src/harmonize/dataset_fallback.py

Day 18 deliverable: a safety wrapper so the harmonization pipeline never
crashes when a dataset's raw data isn't available yet (e.g. MIMIC-III/IV
while PhysioNet credentialing is still pending).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def try_process_dataset(
    dataset,
    loader_fn: Callable[[], tuple[np.ndarray, float, str]],
    domain_id: str,
) -> Optional[np.ndarray]:
    """
    Attempt to load and process one dataset through the harmonization
    pipeline. If the raw data isn't available, skip it gracefully
    instead of crashing the whole run.

    Parameters
    ----------
    dataset : PhysioShiftDataset
        An initialized PhysioShiftDataset instance to process the signal into.
    loader_fn : Callable[[], tuple[np.ndarray, float, str]]
        Zero-argument function that performs the actual file I/O and
        returns (signal, original_fs, modality). Should raise
        FileNotFoundError if the raw data isn't present on disk.
    domain_id : str
        Clean domain-ID string for this dataset, e.g. 'D6_MIMIC_clinical_icu'.

    Returns
    -------
    np.ndarray or None
        Windowed, normalized signal array if loading succeeded.
        None if the dataset was skipped (missing file or unexpected error) --
        caller should handle None gracefully rather than assuming success.
    """
    try:
        signal, original_fs, modality = loader_fn()
    except FileNotFoundError as e:
        print(f"[SKIP] {domain_id}: data not found ({e}). "
              f"Proceeding without this dataset (fallback mode).")
        return None
    except Exception as e:
        print(f"[SKIP] {domain_id}: unexpected error while loading "
              f"({type(e).__name__}: {e}). Skipping.")
        return None

    return dataset.process_signal(
        signal,
        original_fs=original_fs,
        modality=modality,
        domain_id=domain_id,
    )