from src.sst2_cda_baseline import train_sst2_cda_baseline


def main():
    # You can pass max_train_samples / max_eval_samples here if you want to
    metrics = train_sst2_cda_baseline(
        max_train_samples=None,
        max_eval_samples=None,
    )
    print("\n=== SST-2 validation metrics (CDA baseline) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
