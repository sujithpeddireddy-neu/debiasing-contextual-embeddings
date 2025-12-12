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
