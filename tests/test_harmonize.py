import numpy as np
import pytest
from src.harmonize.window import normalize_and_window, sliding_window_indices, zscore_normalize


def test_zscore_normalize_mean_zero_std_one():
    rng = np.random.default_rng(42)
    window = rng.normal(loc=5.0, scale=2.0, size=1000)
    normalized = zscore_normalize(window)
    assert np.isclose(normalized.mean(), 0.0, atol=1e-8)
    assert np.isclose(normalized.std(), 1.0, atol=1e-6)


def test_zscore_normalize_constant_window_no_divide_by_zero():
    window = np.full(100, 7.0)
    normalized = zscore_normalize(window)
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0, atol=1e-3)


def test_sliding_window_indices_50pct_overlap_count():
    n_samples = 1000
    window_size = 100
    step = 50
    indices = sliding_window_indices(n_samples, window_size, step)
    assert len(indices) == 19
    assert indices[0] == (0, 100)
    assert indices[1] == (50, 150)
    assert indices[-1][1] <= n_samples


def test_sliding_window_indices_signal_shorter_than_window():
    indices = sliding_window_indices(n_samples=50, window_size=100, step=50)
    assert indices == []


def test_normalize_and_window_basic_shape():
    fs = 100
    duration_sec = 30
    signal = np.sin(np.linspace(0, 20 * np.pi, duration_sec * fs))
    windows = normalize_and_window(signal, fs=fs, window_sec=10, overlap=0.5)
    window_size = 10 * fs
    assert windows.shape == (5, window_size)


def test_normalize_and_window_short_signal_returns_empty():
    fs = 100
    short_signal = np.random.default_rng(2).normal(size=5 * fs)
    windows = normalize_and_window(short_signal, fs=fs, window_sec=10, overlap=0.5)
    assert windows.shape == (0, 10 * fs)


def test_normalize_and_window_per_window_normalization():
    fs = 100
    rng = np.random.default_rng(1)
    t = np.arange(0, 30 * fs)
    signal = (t / fs) * np.sin(2 * np.pi * 1.0 * t / fs) + rng.normal(0, 0.01, t.shape)
    windows = normalize_and_window(signal, fs=fs, window_sec=10, overlap=0.5)
    assert windows.shape[0] > 0
    for w in windows:
        assert np.isclose(w.mean(), 0.0, atol=1e-6)
        assert np.isclose(w.std(), 1.0, atol=1e-4)


def test_normalize_and_window_handles_nans_by_dropping_bad_windows():
    fs = 100
    duration_sec = 30
    signal = np.sin(np.linspace(0, 20 * np.pi, duration_sec * fs))
    signal = signal.copy()
    signal[1000:2000] = np.nan
    windows = normalize_and_window(signal, fs=fs, window_sec=10, overlap=0.5, drop_nan_windows=True)
    assert windows.shape[0] > 0
    assert not np.any(np.isnan(windows))


def test_normalize_and_window_invalid_overlap_raises():
    fs = 100
    signal = np.zeros(50 * fs)
    with pytest.raises(ValueError):
        normalize_and_window(signal, fs=fs, window_sec=10, overlap=1.0)
    with pytest.raises(ValueError):
        normalize_and_window(signal, fs=fs, window_sec=10, overlap=-0.1)


def test_normalize_and_window_non_1d_signal_raises():
    fs = 100
    signal_2d = np.zeros((100, 2))
    with pytest.raises(ValueError):
        normalize_and_window(signal_2d, fs=fs, window_sec=10, overlap=0.5)