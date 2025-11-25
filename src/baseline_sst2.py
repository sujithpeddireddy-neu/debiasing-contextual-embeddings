from typing import Optional, Dict, Any

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate


def train_sst2_baseline(
    model_name: str = "bert-base-uncased",
    output_dir: str = "checkpoints/bert-base-sst2",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fine-tune a BERT-base model on SST-2 and return eval metrics.

    Args:
        model_name: HF model name to use (baseline: 'bert-base-uncased').
        output_dir: Directory for checkpoints/logs.
        max_train_samples: If set, subsample training set for quick runs.
        max_eval_samples: If set, subsample validation set.

    Returns:
        A dict of evaluation metrics on the SST-2 validation set.
    """
    raw_datasets = load_dataset("glue", "sst2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess(examples):
        return tokenizer(examples["sentence"], truncation=True)

    encoded = raw_datasets.map(preprocess, batched=True)

    train_dataset = encoded["train"]
    eval_dataset = encoded["validation"]

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        metrics = {}
        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics["f1_weighted"] = f1_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["f1"]
        return metrics

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_steps=50,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    print("\n=== SST-2 validation metrics (baseline) ===")
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return eval_metrics
