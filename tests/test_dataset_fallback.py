import numpy as np
import pytest
from src.harmonize.dataset_fallback import try_process_dataset


class MockDataset:
    def __init__(self):
        self.calls = []

    def process_signal(self, signal, original_fs, modality, domain_id):
        self.calls.append(domain_id)
        return np.asarray(signal).reshape(1, -1)


def test_try_process_dataset_skips_missing_data_gracefully():
    ds = MockDataset()

    def missing_loader():
        raise FileNotFoundError("data/mimic directory not found")

    result = try_process_dataset(ds, missing_loader, domain_id="D6_MIMIC_clinical_icu")
    assert result is None
    assert ds.calls == []


def test_try_process_dataset_processes_available_data_normally():
    ds = MockDataset()

    def working_loader():
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        return signal, 100.0, "ecg"

    result = try_process_dataset(ds, working_loader, domain_id="D1_MITBIH_open_chest")
    assert result is not None
    assert ds.calls == ["D1_MITBIH_open_chest"]
    assert result.shape == (1, 4)


def test_try_process_dataset_does_not_stop_other_datasets():
    ds = MockDataset()

    def working_loader_a():
        return np.array([1.0, 2.0]), 100.0, "ecg"

    def missing_loader():
        raise FileNotFoundError("MIMIC not available")

    def working_loader_b():
        return np.array([3.0, 4.0, 5.0]), 64.0, "ppg"

    results = {}
    results["D1"] = try_process_dataset(ds, working_loader_a, domain_id="D1")
    results["D6_MIMIC"] = try_process_dataset(ds, missing_loader, domain_id="D6_MIMIC")
    results["D5"] = try_process_dataset(ds, working_loader_b, domain_id="D5")

    assert results["D1"] is not None
    assert results["D6_MIMIC"] is None
    assert results["D5"] is not None
    assert ds.calls == ["D1", "D5"]


def test_try_process_dataset_handles_unexpected_errors_without_crashing():
    ds = MockDataset()

    def corrupt_loader():
        raise ValueError("unexpected header format in .dat file")

    result = try_process_dataset(ds, corrupt_loader, domain_id="D6_MIMIC_corrupt_test")
    assert result is None
    assert ds.calls == []