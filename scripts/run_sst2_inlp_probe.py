"""
Evaluate how INLP affects SST-2 sentiment classification when we use
the CDA-finetuned SST-2 model but the projection learned on
vanilla BERT.
"""
from src.sst2_inlp_probe import main

if __name__ == "__main__":
    main(
        model_name = "checkpoints/bert-base-sst2-cda/checkpoint-6972",
        train_split="train",
        val_split="validation",       # full dev set (872 ex)
        projection_path="checkpoints/inlp_projection.joblib",
        output_path="outputs/sst2_inlp_probe_results_CDA.json",
        batch_size=32,
    )
