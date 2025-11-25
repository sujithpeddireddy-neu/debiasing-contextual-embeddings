from src.baseline_sst2 import train_sst2_baseline


def main():
    # sanity check:
    # metrics = train_sst2_baseline(max_train_samples=2000, max_eval_samples=1000)

    # For a proper baseline, use full data:
    metrics = train_sst2_baseline()

    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
