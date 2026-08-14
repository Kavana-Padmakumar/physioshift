import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(raw_signal, fs, modality="ecg", order=4):
    """
    Applies a modality-aware zero-phase bandpass filter to a physiological signal.

    Parameters
    ----------
    raw_signal : array-like
        The raw 1D signal to filter.
    fs : float
        Sampling rate of the signal, in Hz.
    modality : str
        One of "ecg", "ppg", or "eda". Determines the frequency band kept.
    order : int
        Butterworth filter order (default 4).

    Returns
    -------
    np.ndarray
        The zero-phase filtered signal, same length as the input.
    """
    cutoffs = {
        "ecg": (0.5, 40),
        "ppg": (0.5, 8),
        "eda": (0.001, 5),
    }

    if modality not in cutoffs:
        raise ValueError(f"Unknown modality: {modality}")

    low, high = cutoffs[modality]

    # Nyquist safety check: high cutoff must stay below fs/2
    nyquist = fs / 2
    if high >= nyquist:
        high = nyquist * 0.95

    b, a = butter(N=order, Wn=[low, high], btype="band", fs=fs)
    return filtfilt(b, a, raw_signal)
