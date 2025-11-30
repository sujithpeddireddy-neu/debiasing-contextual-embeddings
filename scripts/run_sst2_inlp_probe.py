from src.sst2_inlp_probe import main


if __name__ == "__main__":
    main(
         model_name="bert-base-uncased",
        train_split="train",          # or "train[:]" – full train
        val_split="validation",       # full dev set (872 ex)
        projection_path="checkpoints/inlp_projection.joblib",
        output_path="outputs/sst2_inlp_probe_results_full.json",
        batch_size=32,
    )