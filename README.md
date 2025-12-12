# debiasing-contextual-embeddings

Project code + scripts for experimenting with debiasing contextual embeddings / language model representations (Baseline, CDA, INLP) and evaluating on SST-2, StereoSet, and gender-probe style tests.

## Repository layout
- `src/` – core implementation (data loading, CDA utilities, INLP, task logic)
- `scripts/` – runnable entry-point scripts
- `requirements.txt` – dependencies

## Setup
```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Run (from repo root)

### SST-2
Baseline:
```bash
python scripts/run_sst2_baseline.py
```

CDA baseline:
```bash
python scripts/run_sst2_cda_baseline.py
```

INLP probe on SST-2:
```bash
python scripts/run_sst2_inlp_probe.py
```

SST-2 INLP probe with CDA:
```bash
python scripts/run_sst2_inlp_probe_cda_true.py
```

### Gender probe (INLP)
INLP gender probe:
```bash
python scripts/run_inlp_gender_probe.py
```

INLP gender probe with CDA:
```bash
python scripts/run_inlp_gender_probe_cda.py
```

### StereoSet
INLP on StereoSet:
```bash
python scripts/run_stereoset_inlp.py
```

### Stats / reporting
Compute summary stats:
```bash
python scripts/run_compute_stats.py
```

## Notes
- Some scripts may download datasets/models on first run and cache them locally.
- If you get import errors, reinstall deps: `pip install -r requirements.txt`
