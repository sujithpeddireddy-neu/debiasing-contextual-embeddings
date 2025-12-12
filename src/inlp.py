"""
Iterative Nullspace Projection (INLP) implementation for debiasing contextual embeddings.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from transformers import AutoTokenizer, AutoModel

def extract_cls_embeddings(
    model_name: str,
    texts: List[str],
    batch_size: int = 32,
    device: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Extract CLS token embeddings from a HuggingFace Transformer model.
    Returns numpy array of shape (N, d).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_kwargs = tokenizer_kwargs or {"truncation": True, "padding": True, "max_length": 128, "return_tensors": "pt"}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, **tokenizer_kwargs)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls)

    X = np.vstack(all_embs)
    return X


DEFAULT_MALE_PRONOUNS = {"he", "him", "his"}
DEFAULT_FEMALE_PRONOUNS = {"she", "her", "hers"}


def build_pronoun_gender_dataset(
    texts: List[str],
    male_pronouns: Optional[set] = None,
    female_pronouns: Optional[set] = None,
    ignore_ambiguous: bool = True,
) -> Tuple[List[str], np.ndarray]:
    male_pronouns = male_pronouns or DEFAULT_MALE_PRONOUNS
    female_pronouns = female_pronouns or DEFAULT_FEMALE_PRONOUNS

    selected = []
    labels = []
    for t in texts:
        t_lower = t.lower()
        has_m = any(p in t_lower.split() for p in male_pronouns)
        has_f = any(p in t_lower.split() for p in female_pronouns)

        if has_m and not has_f:
            selected.append(t)
            labels.append(0)
        elif has_f and not has_m:
            selected.append(t)
            labels.append(1)
        else:
            if not ignore_ambiguous:
                selected.append(t)
                labels.append(-1)

    if len(selected) == 0:
        raise ValueError("No pronoun-labeled examples found. Provide different texts or lower ignore_ambiguous.")
    return selected, np.array(labels)

def orthonormalize_rows(W: np.ndarray) -> np.ndarray:
    if W.size == 0:
        return np.zeros((W.shape[1], 0))
    Q, _ = np.linalg.qr(W.T)
    k = W.shape[0]
    return Q[:, :k]


def learn_inlp_projection(
    X: np.ndarray,
    y: np.ndarray,
    n_iters: int = 10,
    clf_penalty: str = "l2",
    clf_C: float = 1.0,
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    assert X.ndim == 2, "X should be (N, d)"
    N, d = X.shape
    assert y.shape[0] == N, "y must have same first dimension as X"

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    P = np.eye(d, dtype=float)
    W_rows = []

    for it in range(n_iters):
        X_proj = Xz @ P  # (N, d)
        clf = LogisticRegression(penalty=clf_penalty, C=clf_C, solver="liblinear", random_state=random_state, max_iter=1000)
        clf.fit(X_proj, y)
        w = clf.coef_.reshape(-1)  # shape (d,)
        norm_w = np.linalg.norm(w)
        if verbose:
            acc = clf.score(X_proj, y)
            print(f"[INLP] Iter {it+1}/{n_iters} — classifier acc on projected space: {acc:.4f} — ||w||={norm_w:.4e}")
        if norm_w < 1e-8:
            if verbose:
                print("[INLP] Terminating early: weight norm too small.")
            break

        W_rows.append(w.copy())
        W_mat = np.vstack(W_rows)
        Q = orthonormalize_rows(W_mat)
        P = np.eye(d) - Q @ Q.T
    info = {"W_rows": np.array(W_rows), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "n_iters": len(W_rows)}
    return P, info


def apply_projection(X: np.ndarray, P: np.ndarray, scaler_mean: Optional[np.ndarray] = None, scaler_scale: Optional[np.ndarray] = None) -> np.ndarray:
    if scaler_mean is not None and scaler_scale is not None:
        Xz = (X - scaler_mean) / (scaler_scale + 1e-12)
        return Xz @ P
    else:
        return X @ P


def save_projection(path: str, P: np.ndarray, info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"P": P, "info": info}, path)


def load_projection(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = joblib.load(path)
    return data["P"], data["info"]

if __name__ == "__main__":

    from datasets import load_dataset

    print("Loading a small text pool (GLUE SST-2 train subset) for demo ...")
    glue = load_dataset("glue", "sst2", split="train[:2000]")
    texts = [s for s in glue["sentence"]]

    selected_texts, labels = build_pronoun_gender_dataset(texts, ignore_ambiguous=True)
    print(f"Pronoun-labeled examples: {len(selected_texts)} (male/female counts: {np.bincount(labels) if len(labels)>0 else None})")

    print("Extracting CLS embeddings (this may take a while)...")
    X = extract_cls_embeddings(model_name="bert-base-uncased", texts=selected_texts, batch_size=32)
    print("Embedding shape:", X.shape)

    print("Learning INLP projection ...")
    P, info = learn_inlp_projection(X, labels, n_iters=10, verbose=True)
    print("Learned projection P shape:", P.shape, "iters used:", info["n_iters"])

    X_proj = apply_projection(X, P, scaler_mean=info["scaler_mean"], scaler_scale=info["scaler_scale"])
    print("Projected embeddings shape:", X_proj.shape)

    save_projection("checkpoints/inlp_projection.joblib", P, info)
    print("Saved projection to checkpoints/inlp_projection.joblib")
