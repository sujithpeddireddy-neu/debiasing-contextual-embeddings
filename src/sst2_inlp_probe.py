from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.inlp import extract_cls_embeddings, load_projection, apply_projection

def get_sst2_split_embeddings(
    model_name: str = "bert-base-uncased",
    split: str = "train[:10000]",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an SST-2 split and return:
      X: CLS embeddings (N, d)
      y: labels (N,)
    We use extract_cls_embeddings from src.inlp.
    """
    print(f"Loading SST-2 split: {split}")
    ds = load_dataset("glue", "sst2", split=split)
    texts = list(ds["sentence"])
    labels = np.array(ds["label"], dtype=int)

    print(f"Extracting CLS embeddings for {len(texts)} examples...")
    X = extract_cls_embeddings(
        model_name=model_name,
        texts=texts,
        batch_size=batch_size,
    )
    print("Embeddings shape:", X.shape)
    return X, labels


def train_logreg_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Train a simple Logistic Regression classifier and evaluate on val set.
    """
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }


def main(
    model_name: str = "bert-base-uncased",
    train_split: str = "train[:10000]",
    val_split: str = "validation[:2000]",
    projection_path: str = "checkpoints/inlp_projection.joblib",
    output_path: str = "outputs/sst2_inlp_probe_results.json",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    1. Get CLS embeddings for SST-2 train/val.
    2. Train & evaluate a logistic regression on original embeddings.
    3. Load INLP projection and apply it to train/val embeddings.
    4. Train & evaluate another logistic regression on projected embeddings.
    5. Save results to JSON.
    """
    from pathlib import Path
    import json
    import os

    X_train, y_train = get_sst2_split_embeddings(
        model_name=model_name,
        split=train_split,
        batch_size=batch_size,
    )
    X_val, y_val = get_sst2_split_embeddings(
        model_name=model_name,
        split=val_split,
        batch_size=batch_size,
    )

    print("\n=== Training baseline probe on ORIGINAL embeddings ===")
    baseline_metrics = train_logreg_probe(X_train, y_train, X_val, y_val)
    print(
        f"Baseline probe - "
        f"accuracy: {baseline_metrics['accuracy']:.4f}, "
        f"F1-weighted: {baseline_metrics['f1_weighted']:.4f}"
    )


    print(f"\nLoading INLP projection from {projection_path} ...")
    P, info = load_projection(projection_path)
    print("Projection shape:", P.shape, "| iters used:", info.get("n_iters", "N/A"))


    print("Applying INLP projection to train/val embeddings ...")
    X_train_proj = apply_projection(
        X_train,
        P,
        scaler_mean=info.get("scaler_mean"),
        scaler_scale=info.get("scaler_scale"),
    )
    X_val_proj = apply_projection(
        X_val,
        P,
        scaler_mean=info.get("scaler_mean"),
        scaler_scale=info.get("scaler_scale"),
    )


    print("\n=== Training probe on INLP-PROJECTED embeddings ===")
    inlp_metrics = train_logreg_probe(X_train_proj, y_train, X_val_proj, y_val)
    print(
        f"INLP probe - "
        f"accuracy: {inlp_metrics['accuracy']:.4f}, "
        f"F1-weighted: {inlp_metrics['f1_weighted']:.4f}"
    )


    results = {
        "model": model_name,
        "train_split": train_split,
        "val_split": val_split,
        "projection_path": projection_path,
        "baseline_probe": baseline_metrics,
        "inlp_probe": inlp_metrics,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nSaved SST-2 INLP probe results to {out_path}")
    return results
