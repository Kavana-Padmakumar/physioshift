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
        indices.append((start, start + window_size))
        start += step
    return indices


def normalize_and_window(signal, fs, window_sec, overlap=0.0, eps=1e-8):
    """
    Normalizes a signal and slices it into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values.
    fs : int
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds.
    overlap : float, default=0.0
        Fractional overlap between consecutive windows (0 to 1).
    eps : float, default=1e-8
        Passed through to zscore_normalize.

    Returns
    -------
    np.ndarray
        2D array of shape (num_windows, window_size).
    """
    normalized = zscore_normalize(signal, eps=eps)
    window_size = int(window_sec * fs)
    step = max(1, int(window_size * (1 - overlap)))
    idx_pairs = sliding_window_indices(len(normalized), window_size, step)
    windows = [normalized[start:end] for start, end in idx_pairs]
    return np.array(windows)
