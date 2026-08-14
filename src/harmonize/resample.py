import numpy as np
from scipy.signal import resample_poly
from math import gcd

def harmonize_resample(signal: np.ndarray, original_fs: float, target_fs: float = 100.0) -> np.ndarray:
    """Resample a 1D physiological signal from original_fs to target_fs using polyphase filtering ^(anti-aliased^)."""
    orig = int(round(original_fs))
    targ = int(round(target_fs))
    g = gcd(orig, targ)
    up = targ // g
    down = orig // g
    return resample_poly(signal, up, down)
