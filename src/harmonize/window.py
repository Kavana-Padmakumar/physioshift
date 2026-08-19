from __future__ import annotations
import numpy as np


def zscore_normalize(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalizes a signal window (zero mean, unit variance).

    Parameters
    ----------
    window : np.ndarray
        1D array of signal values to normalize.
    eps : float, default=1e-8
        Small constant added to the standard deviation to avoid
        division by zero on flat/constant windows.

    Returns
    -------
    np.ndarray
        Normalized window, same shape as input.
    """
    window = np.asarray(window, dtype=np.float64)
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return (window - mean) / (std + eps)


def sliding_window_indices(n_samples: int, window_size: int, step: int) -> list[tuple[int, int]]:
    """
    Computes (start, end) index pairs for sliding windows over a signal.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the signal.
    window_size : int
        Number of samples per window.
    step : int
        Number of samples to advance between consecutive windows
        (step = window_size * (1 - overlap)).

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) index pairs. Empty list if the signal is
        shorter than window_size.
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive integers")
    if n_samples < window_size:
        return []
    indices = []
    start = 0
    while start + window_size <= n_samples:
        indices.append((start, start + window