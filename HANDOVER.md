# PhysioShift Handover Notes

## Phase 2 (13–19 Aug) — COMPLETE
Full data harmonization pipeline built, tested, and documented across 7 days:
- `resample.py`, `filter.py`, `window.py` (13–15 Aug)
- `dataset.py` — PhysioShiftDataset class chaining all three (16 Aug)
- `severity.py` — pairwise domain-shift severity matrix (17 Aug)
- `dataset_fallback.py` — safety wrapper for missing MIMIC access (18 Aug)
- Full docstrings across the entire module + `data/README.md` regeneration
  guide + frozen `requirements.txt` (19 Aug)

**Full regeneration guide:** `data/README.md` — covers raw data download
commands for all 5 datasets, folder structure, pipeline commands, and
design rationale.

## Harmonized Cache — ready to use as model input
Location: `data/harmonized_cache/*.npy`

Each file: windowed, resampled (100Hz), filtered, z-score normalized.
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

⚠️ **Caveat:** D2 (PTB-XL) and D4 (WESAD) currently have only 1 window
each — too few samples for a meaningful train/val split alone. If
Phase 3's training script needs more data per domain, either re-run
`PhysioShiftDataset.process_signal()` with longer input slices, or
treat this as a known limitation for the baseline pass.

## Next up: Member 2, 20 Aug — Phase 3 begins
Implement `LightweightResNet1D` — a ~500K parameter 1D-ResNet in PyTorch
(6 lightweight residual blocks, 1D convolutions, batch norm, ReLU).

**Deliverable:** `src/models/resnet1d.py` + a notebook cell showing
parameter count and a successful forward-pass sanity check (~500K
param range).

**Input shape expectation:** model's forward() should accept the
harmonized cache format above, likely reshaped to (batch, 1, 1000)
for the 1D-CNN channel dimension.