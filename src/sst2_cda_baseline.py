from typing import Optional, Dict, Any

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from .cda_utils import swap_gender_terms


def build_cda_sst2_train(max_train_samples: Optional[int] = None) -> Dataset:
    """
    Load SST-2 train split and apply CDA:
    for each sentence, add a gender-swapped version (if it changes).
    """
    raw = load_dataset("glue", "sst2")
    train = raw["train"]

    if max_train_samples is not None:
        train = train.select(range(max_train_samples))

    orig_texts = []
    orig_labels = []

    for ex in train:
        s = ex["sentence"]
        y = ex["label"]
        orig_texts.append(s)
        orig_labels.append(y)

        swapped = swap_gender_terms(s)
        # Only add if something actually changed
        if swapped != s:
            orig_texts.append(swapped)
            orig_labels.append(y)

    augmented = Dataset.from_dict({"sentence": orig_texts, "label": orig_labels})
    return augmented


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def train_sst2_cda_baseline(
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    output_dir: str = "checkpoints/bert-base-sst2-cda",
) -> Dict[str, Any]:
    """
    Fine-tune bert-base-uncased on CDA-augmented SST-2.
    Returns the evaluation metrics dict.
    """
    raw = load_dataset("glue", "sst2")

    # Build CDA-augmented train set
    train_dataset = build_cda_sst2_train(max_train_samples=max_train_samples)

    eval_dataset = raw["validation"]
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )
    tokenized_eval = eval_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
)

    def compute_metrics(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics