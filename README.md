# debiasing-contextual-embeddings

Code + scripts for a project exploring **debiasing contextual embeddings / language model representations**, with an emphasis on running reproducible experiments and evaluations. :contentReference[oaicite:0]{index=0}

## Repository layout

- `src/` – core Python source code (models / training / evaluation utilities). :contentReference[oaicite:1]{index=1}  
- `scripts/` – runnable scripts to reproduce experiments (preprocess/train/eval). :contentReference[oaicite:2]{index=2}  
- `requirements.txt` – Python dependencies. :contentReference[oaicite:3]{index=3}  
- `ProjectProposal.pdf` – project proposal / design context. :contentReference[oaicite:4]{index=4}  

## Setup

```bash
# (recommended) create a virtual environment
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows:
# .venv\Scripts\activate

pip install -r requirements.txt
