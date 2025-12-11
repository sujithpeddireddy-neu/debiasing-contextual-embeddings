"""
Entry point for training the CDA-based SST-2 baseline model.
"""
from src.sst2_cda_baseline import train_sst2_cda_baseline

def main():
    metrics = train_sst2_cda_baseline(
        max_train_samples= 35000,
        max_eval_samples= 872,
    )
    print("\n=== SST-2 validation metrics (CDA baseline) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
