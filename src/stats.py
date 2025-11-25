from collections import Counter
from typing import Dict, Any

from datasets import Dataset, DatasetDict

from .data_loading import load_stereoset, load_crows_pairs, load_sst2

def _avg_token_length(texts) -> float:
    # Very simple whitespace tokenization for rough stats
    total_tokens = 0
    for t in texts:
        total_tokens += len(str(t).split())
    return total_tokens / max(len(texts), 1)


def stereoset_stats() -> Dict[str, Any]:
    """
    Compute basic stats for StereoSet (per subset).

    Returns:
        {
          "intrasentence": {...},
          "intersentence": {...},
          "overall": {...}
        }
    """
    stats: Dict[str, Any] = {}
    stereo_all = load_stereoset("all")
    assert isinstance(stereo_all, DatasetDict)

    total_examples = 0
    bias_type_global = Counter()

    for subset_name, ds in stereo_all.items():
        assert isinstance(ds, Dataset)
        n = len(ds)
        total_examples += n

        # bias_type is one of {gender, race, religion, profession}
        bias_counts = Counter(ds["bias_type"])

        # context is the prompt; sentences is the candidate completions
        avg_context_len = _avg_token_length(ds["context"])
        # Flatten sentences list-of-lists into individual sentence strings
        all_sentences = [s for seq in ds["sentences"] for s in seq["sentence"]]
        avg_sentence_len = _avg_token_length(all_sentences)

        bias_type_global.update(bias_counts)

        stats[subset_name] = {
            "num_examples": n,
            "bias_type_counts": dict(bias_counts),
            "avg_context_len_tokens": avg_context_len,
            "avg_candidate_sentence_len_tokens": avg_sentence_len,
        }

    stats["overall"] = {
        "num_examples": total_examples,
        "bias_type_counts": dict(bias_type_global),
    }

    return stats


def crows_pairs_stats() -> Dict[str, Any]:
    """
    Compute basic stats for CrowS-Pairs.

    Returns:
        {
          "num_examples": int,
          "bias_type_counts": {...},
          "stereo_vs_antistereo": {...},
          "avg_sent_more_len_tokens": float,
          "avg_sent_less_len_tokens": float,
        }
    """
    ds = load_crows_pairs()

    n = len(ds)
    bias_counts = Counter(ds["bias_type"])
    direction_counts = Counter(ds["stereo_antistereo"])

    avg_more_len = _avg_token_length(ds["sent_more"])
    avg_less_len = _avg_token_length(ds["sent_less"])

    return {
        "num_examples": n,
        "bias_type_counts": dict(bias_counts),
        "stereo_vs_antistereo": dict(direction_counts),
        "avg_sent_more_len_tokens": avg_more_len,
        "avg_sent_less_len_tokens": avg_less_len,
    }


def sst2_stats() -> Dict[str, Any]:
    """
    Compute basic stats for SST-2 (GLUE).

    Returns:
        {
          "train": {...},
          "validation": {...},
          "test": {...},
          "label_names": [...],
        }
    """
    ds_dict = load_sst2()
    assert isinstance(ds_dict, DatasetDict)

    label_names = ds_dict["train"].features["label"].names

    split_stats: Dict[str, Any] = {}

    for split in ds_dict.keys():
        ds = ds_dict[split]
        labels = ds["label"]
        label_counts = Counter(labels)

        avg_sentence_len = _avg_token_length(ds["sentence"])

        # Map integer labels to human-readable class names
        label_counts_named = {label_names[k]: v for k, v in label_counts.items()}

        split_stats[split] = {
            "num_examples": len(ds),
            "label_counts": dict(label_counts_named),
            "avg_sentence_len_tokens": avg_sentence_len,
        }

    split_stats["label_names"] = list(label_names)
    return split_stats
