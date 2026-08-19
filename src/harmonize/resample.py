import numpy as np
from scipy.signal import resample_poly
from math import gcd


def harmonize_resample(signal: np.ndarray, original_fs: float, target_fs: float = 100.0) -> np.ndarray:
    """
    Resample a 1D physiological signal from original_fs to target_fs
    using polyphase filtering (anti-aliased).

    Parameters
    ----------
    signal : np.ndarray
        The raw 1D signal to resample.
    original_fs : float
        The signal's native sampling rate, in Hz.
    target_fs : float, default=100.0
        The desired output sampling rate, in Hz. 100Hz is used as the
        common target across all 5 project datasets.

    Returns
    -------
    np.ndarray
        The resampled signal, at target_fs.

    Notes
    -----
    Uses scipy.signal.resample_poly rather than naive slicing/padding
    because it applies an anti-aliasing lowpass filter before changing
    the sample rate, preventing high-frequency content from folding
    back as spurious low-frequency noise.
    """
    orig = int(round(original_fs))
    targ = int(round(target_fs))
    g = gcd(orig, targ)
    up = targ // g
    down = orig // g
    return resample_poly(signal, up, down)