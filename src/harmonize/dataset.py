"""
src/harmonize/dataset.py

PhysioShiftDataset: assembles the full harmonization pipeline
resample -> filter -> normalize_and_window
chaining Member 1 (resample.py), Member 2 (filter.py), and Member 3 (window.py).
"""

import os
import numpy as np

from src.harmonize.resample import harmonize_resample
from src.harmonize.filter import bandpass_filter
from src.harmonize.window import normalize_and_window


class PhysioShiftDataset:
    """
    Domain ID schema: {dataset}_{device}_{placement}
    e.g. "D1_MIMIC_clinical_chest", "D4_WESAD_wearable_chest"
    """

    def __init__(self, target_fs: float = 100.0, window_sec: float = 10.0, overlap: float = 0.5):
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.overlap = overlap
        self.cache = {}

    def process_signal(self, signal, original_fs, modality="ecg", domain_id="unknown"):
        resampled = harmonize_resample(signal, original_fs=original_fs, target_fs=self.target_fs)
        filtered = bandpass_filter(resampled, fs=self.target_fs, modality=modality)
        windowed = normalize_and_window(
            filtered,
            fs=int(self.target_fs),
            window_sec=self.window_sec,
            overlap=self.overlap,
        )
        self.cache[domain_id] = windowed
        return windowed

    def save_cache(self, out_dir="data/harmonized_cache"):
        """
    Saves each cached domain's windowed array as a .npy file.

    Parameters
    ----------
    out_dir : str, default="data/harmonized_cache"
        Directory to save the .npy files into. Created if it
        doesn't already exist.

    Returns
    -------
    None
        Writes one {domain_id}.npy file per cached domain and
        prints a confirmation line for each.
    """
        os.makedirs(out_dir, exist_ok=True)

        for domain_id, arr in self.cache.items():
            path = os.path.join(out_dir, f"{domain_id}.npy")
            np.save(path, arr)
            print(f"Saved {domain_id}: shape={arr.shape} -> {path}")


    def summary(self):
        """
        Prints a quick summary of every domain currently in the cache,
        showing each domain_id alongside its windowed array shape.

        Returns
        -------
        None
            Prints directly to stdout; does not return a value.
        """
        print(f"\n{'='*50}")   

        print(f"  PhysioShiftDataset cache summary")
        print(f"{'='*50}")
        for domain_id, arr in self.cache.items():
            print(f"  {domain_id:35}: shape={arr.shape}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 250
    t = np.linspace(0, 30, 30 * fs, endpoint=False)
    fake_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(t.shape)

    ds = PhysioShiftDataset(target_fs=100.0, window_sec=10.0, overlap=0.5)
    out = ds.process_signal(fake_ecg, original_fs=fs, modality="ecg", domain_id="TEST_synthetic_ecg")
    ds.summary()