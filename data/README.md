\# PhysioShift — Data Directory



This document explains how to regenerate every raw dataset and the

harmonized cache from scratch. Written for a stranger or mentor

reviewer with no prior context on this project.



\---



\## 1. Raw Data Sources



Five open-access datasets are used. A sixth (MIMIC-III/IV) is

credentialed-access and handled separately (see Section 4).



\### D1: MIT-BIH Arrhythmia Database

\- Format: WFDB (PhysioNet native format)

\- Source: PhysioNet, open access

\- Download:

```python

&#x20; import wfdb

&#x20; wfdb.dl\_database('mitdb', dl\_dir='data/mitdb')

```



\### D2: PTB-XL

\- Format: WFDB

\- Source: PhysioNet, open access

\- Download:

```python

&#x20; import wfdb

&#x20; wfdb.dl\_database('ptb-xl', dl\_dir='data/ptbxl')

```



\### D3: Multi-site PPG (2026)

\- Format: WFDB

\- Source: PhysioNet, open access (verify access type before download —

&#x20; search "multi-site PPG" at https://physionet.org/content/)

\- Download: same wfdb.dl\_database() pattern as above



\### D4: WESAD (Wearable Stress and Affect Detection)

\- Format: Python pickle (.pkl), NOT WFDB

\- Source: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+affect+detection

\- Download: manual — click Download on the UCI page, extract into

&#x20; `data/wesad/`

\- Expected structure after extraction: `data/wesad/S{n}/S{n}.pkl`

&#x20; (subjects S2–S17, some subject numbers may be missing)



\### D5: PPG-DaLiA

\- Format: Python pickle (.pkl), NOT WFDB

\- Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia

\- Download (scriptable):

```python

&#x20; import urllib.request, zipfile

&#x20; url = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"

&#x20; urllib.request.urlretrieve(url, "data/ppgdalia/ppg\_dalia.zip")

&#x20; with zipfile.ZipFile("data/ppgdalia/ppg\_dalia.zip", 'r') as z:

&#x20;     z.extractall("data/ppgdalia")

&#x20; # NOTE: the outer zip contains an INNER data.zip — extract that too:

&#x20; with zipfile.ZipFile("data/ppgdalia/data.zip", 'r') as z:

&#x20;     z.extractall("data/ppgdalia")

```

\- Expected structure after extraction:

&#x20; `data/ppgdalia/PPG\_FieldStudy/S{n}/S{n}.pkl`



\---



\## 2. Expected Folder Structure


Raw data files are excluded from git via `.gitignore` — they must be

downloaded fresh using the commands above. Only the harmonized cache

(small, \~3.7MB total) is committed directly.



\---



\## 3. Regenerating the Harmonized Cache



Once all 5 raw datasets are downloaded and placed as above, run:



```bash

python -m src.harmonize.dataset

```



or open and run `notebooks/02\_full\_pipeline.ipynb` top to bottom.



This runs each dataset through the full pipeline:

`harmonize\_resample()` → `bandpass\_filter()` → `normalize\_and\_window()`



producing windowed, normalized `.npy` arrays saved to

`data/harmonized\_cache/`. Each array has shape `(n\_windows, 1000)`

— 1000 samples = 10 seconds at the common target rate of 100Hz.



\*\*Pipeline design choices:\*\*

\- \*\*Target rate: 100Hz\*\* — chosen as it's achievable via a clean

&#x20; rational resampling ratio from all 5 datasets' native rates

&#x20; (360Hz, 100Hz, 500Hz, 700Hz, 64Hz).

\- \*\*Window length: 10 seconds, 50% overlap\*\* — TODO: confirm exact

&#x20; rationale with Member 2/3.

\- \*\*Z-score normalization\*\* — standard approach for cross-dataset

&#x20; comparability, removes per-signal amplitude/offset differences.

\- \*\*Deterministic\*\*: no randomness in the resample/filter/window

&#x20; functions themselves. The smoke test in `dataset.py`'s

&#x20; `\_\_main\_\_` block uses a fixed seed (`np.random.default\_rng(0)`)

&#x20; for reproducibility.



\---



\## 4. MIMIC-III/IV Fallback Behavior



MIMIC requires separate PhysioNet credentialing (in progress as of

19 Aug 2026 — see HANDOVER.md for status). The pipeline is designed

to run cleanly with or without MIMIC access, using

`src/harmonize/dataset\_fallback.py`'s `try\_process\_dataset()`:



\- If MIMIC's raw data files aren't present, the loader raises

&#x20; `FileNotFoundError`, which is caught and logged as a `\[SKIP]`

&#x20; message rather than crashing the run.

\- The severity matrix (`src/harmonize/severity.py`) reserves a 6th

&#x20; row/column for MIMIC, shown as placeholder cells until access

&#x20; is approved.



\---



\## 5. Reproducing Exact Environment



```bash

python -m venv venv

venv\\Scripts\\activate.bat        # Windows

pip install -r requirements.txt  # exact frozen versions

```



`requirements.txt` is frozen with exact package versions as of

19 Aug 2026 to ensure numerical reproducibility (e.g. scipy version

changes could alter `resample\_poly`'s internal behavior).

