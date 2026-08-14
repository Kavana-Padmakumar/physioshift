\# Handover: Phase 2 Day 1 → Day 2



\## What's done (Member 1, 13 Aug)

`harmonize\_resample()` implemented and tested on real signals from all 5 datasets.



\*\*Function location:\*\* `src/harmonize/resample.py`



\*\*Signature:\*\*

```python

harmonize\_resample(signal: np.ndarray, original\_fs: float, target\_fs: float = 100.0) -> np.ndarray

```

Resamples any 1D signal to a target rate (default 100Hz) using `scipy.signal.resample\_poly` (anti-aliased, polyphase filtering).



\*\*Key fact:\*\* All 5 datasets are now harmonized to a common \*\*100Hz\*\* — downstream steps (filtering, windowing) can assume this fixed rate.



\## Dataset reference table



| Dataset | Original Fs | Format | Local Path | Loading Notes |

|---|---|---|---|---|

| MIT-BIH | 360 Hz | WFDB | `data/mitdb/` | `wfdb.rdrecord()` |

| PTB-XL | 100 Hz | WFDB | `data/ptbxl/` | `wfdb.rdrecord()` |

| Multi-site PPG | 500 Hz | WFDB | `data/multisite\_ppg/` | `wfdb.rdrecord()` |

| WESAD | 700 Hz | Pickle | `data/wesad/S{n}/S{n}.pkl` | `pickle.load()` → `\['signal']\['chest']\['ECG']` |

| PPG-DaLiA | 64 Hz | Pickle | `data/ppgdalia/PPG\_FieldStudy/S{n}/S{n}.pkl` | `pickle.load()` → `\['signal']\['wrist']\['BVP']` |



\## Reference materials

\- Test/verification notebook: `notebooks/01\_resample\_test.ipynb`

\- Before/after plots: `notebooks/plots/\*.png`



\## Next up (Member 2, 14 Aug)

Implement `bandpass\_filter()` using `scipy.signal.butter` + `filtfilt`, with modality-aware cutoffs:

\- ECG: 0.5–40 Hz

\- PPG: 0.5–8 Hz

\- EDA/GSR: 0–5 Hz



Apply to all 5 resampled datasets, visually inspect for filtering artifacts.

Deliverable: `src/harmonize/filter.py` + comparison plots for ECG, PPG, EDA signals.

