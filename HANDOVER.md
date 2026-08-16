# PhysioShift Handover Notes

## Phase 2 Day 4 (16 Aug) — COMPLETE
`PhysioShiftDataset` class built, chaining resample → filter → normalize_and_window.
Run end-to-end on all 5 open datasets. Harmonized cache generated and saved.

**Pipeline code:** `src/harmonize/dataset.py`
**Reference notebook:** `notebooks/02_full_pipeline.ipynb`

## Harmonized Cache — ready to use
Location: `data/harmonized_cache/*.npy`

Each file: windowed, resampled (100Hz), filtered, z-score normalized signal.
Shape: `(n_windows, 1000)` — 1000 samples = 10 sec @ 100Hz.

| Domain ID | File | Shape |
|---|---|---|
| D1_MITBIH_open_chest | D1_MITBIH_open_chest.npy | (360, 1000) |
| D2_PTBXL_open_clinical | D2_PTBXL_open_clinical.npy | (1, 1000) |
| D3_MultisitePPG_open_wearable | D3_MultisitePPG_open_wearable.npy | (96, 1000) |
| D4_WESAD_open_chest | D4_WESAD_open_chest.npy | (1, 1000) |
| D5_PPGDaLiA_open_wrist | D5_PPGDaLiA_open_wrist.npy | (14, 1000) |

**Load example:**
```python
import numpy as np
d1 = np.load('data/harmonized_cache/D1_MITBIH_open_chest.npy')
```

⚠️ **Caveat:** D2 (PTB-XL) and D4 (WESAD) currently have only 1 window each — the loaded source slices were short. If downstream analysis needs more samples per dataset for meaningful statistics, re-run `PhysioShiftDataset.process_signal()` with longer input signals first.

## Next up: Member 2, 17 Aug
Implement `domain_shift_severity()` — compute the full 6×6 pairwise severity matrix across datasets (5 open datasets + 1 placeholder slot for MIMIC, pending credentialing approval).

Combine 4 mismatch signals into one 0–1 severity score per dataset pair.

**Deliverable:** `src/harmonize/severity.py` + severity matrix as a heatmap figure (with labels + colorbar — this is a required paper figure, build it properly).