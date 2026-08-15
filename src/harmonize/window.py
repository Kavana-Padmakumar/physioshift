from __future__ import annotations
import numpy as np


def zscore_normalize(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    mean = np.nanmean(window)
    std = np.nanstd(window)
    return (window - mean) / (std + eps)


def sliding_window_indices(n_samples: int, window_size: int, step: int) -> list[tuple[int, int]]:
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


def normalize_and_window(
    signal: np.ndarray,
    fs: int,
    window_sec: float = 10.0,
    overlap: float = 0.5,
    drop_nan_windows: bool = True,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D, got shape {signal.shape}")
    if not (0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    window_size = int(round(window_sec * fs))
    step = max(1, int(round(window_size * (1 - overlap))))

    idx_pairs = sliding_window_indices(signal.shape[0], window_size, step)
    if not idx_pairs:
        return np.empty((0, window_size), dtype=np.float64)

    windows = []
    for start, end in idx_pairs:
        raw_window = signal[start:end]
        if drop_nan_windows and np.all(np.isnan(raw_window)):
            continue
        norm_window = zscore_normalize(raw_window)
        if drop_nan_windows and np.any(np.isnan(norm_window)):
            continue
        windows.append(norm_window)

    if not windows:
        return np.empty((0, window_size), dtype=np.float64)
    return np.stack(windows, axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 100
    t = np.linspace(0, 30, 30 * fs, endpoint=False)
    fake_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(t.shape)
    windows = normalize_and_window(fake_ecg, fs=fs, window_sec=10, overlap=0.5)
    print("shape:", windows.shape)
    print("means:", windows.mean(axis=1))
    print("stds:", windows.std(axis=1))