from pathlib import Path
import json

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.inlp import (
    build_pronoun_gender_dataset,
    extract_cls_embeddings,
    learn_inlp_projection,
    apply_projection,
    save_projection,
)


def train_and_eval_probe(X, y, random_state=0):
    """Train a simple logistic-regression gender probe and report metrics."""
    # simple 80/20 split
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)

    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Grab a pool of sentences
    print("Loading SST-2 training subset for pronoun probe ...")
    glue_train = load_dataset("glue", "sst2", split="train[:5000]")  # 5k for speed
    texts = list(glue_train["sentence"])

    # 2. Build a labeled pronoun dataset (0 = male, 1 = female)
    print("Building pronoun-labeled dataset ...")
    selected_texts, labels = build_pronoun_gender_dataset(
        texts,
        ignore_ambiguous=True,
    )
    y = labels
    print(
        f"Selected {len(selected_texts)} sentences "
        f"(label distribution: {np.bincount(y)})"
    )

    # 3. Extract CLS embeddings from *CDA-finetuned* BERT model
    cda_model_name = "checkpoints/bert-base-sst2-cda/checkpoint-6972"
    print(f"Extracting CLS embeddings from CDA model at: {cda_model_name} ...")
    X = extract_cls_embeddings(
        model_name=cda_model_name,
        texts=selected_texts,
        batch_size=32,
    )
    print("Embedding matrix shape:", X.shape)

    # 4. Baseline probe on original CDA embeddings
    print("\n=== Gender probe on ORIGINAL CDA embeddings ===")
    metrics_orig = train_and_eval_probe(X, y)
    print(
        f"Accuracy: {metrics_orig['accuracy']:.4f}, "
        f"F1-weighted: {metrics_orig['f1_weighted']:.4f}"
    )

    # 5. Learn INLP projection on CDA embeddings
    print("\nLearning INLP projection on CDA embeddings ...")
    P, info = learn_inlp_projection(
        X,
        y,
        n_iters=10,
        verbose=True,
    )
    print("Projection P shape:", P.shape, "iters used:", info["n_iters"])

    # Save CDA-specific projection for reuse
    proj_path = Path("checkpoints/inlp_projection_cda.joblib")
    proj_path.parent.mkdir(exist_ok=True)
    save_projection(str(proj_path), P, info)
    print(f"Saved CDA projection to {proj_path}")

    # 6. Apply CDA projection and re-evaluate probe
    print("\nApplying CDA INLP projection and re-evaluating probe ...")
    X_proj = apply_projection(
        X,
        P,
        scaler_mean=info["scaler_mean"],
        scaler_scale=info["scaler_scale"],
    )
    print("Projected embedding matrix shape:", X_proj.shape)

    metrics_proj = train_and_eval_probe(X_proj, y)
    print(
        f"Accuracy (projected): {metrics_proj['accuracy']:.4f}, "
        f"F1-weighted (projected): {metrics_proj['f1_weighted']:.4f}"
    )

    # 7. Save everything to JSON.
    results = {
        "model": cda_model_name,
        "n_examples": int(len(selected_texts)),
        "baseline_probe": metrics_orig,
        "inlp_probe": metrics_proj,
        "inlp_info": {
            "n_iters": int(info["n_iters"]),
        },
    }

    out_path = out_dir / "inlp_gender_probe_results_cda.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nSaved CDA probe results to {out_path}")


if __name__ == "__main__":
    main()
