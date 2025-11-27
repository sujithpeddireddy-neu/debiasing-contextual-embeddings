import math
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset, DatasetDict, load_dataset
from typing import Literal, Union, Dict, Tuple, Optional
from statistics import mean

StereoSubset = Literal["intrasentence", "intersentence", "all"]

def load_stereoset(subset: StereoSubset = "intrasentence") -> Union[Dataset, DatasetDict]:
    if subset == "all":
        return DatasetDict(
            intrasentence=load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation"),
            intersentence=load_dataset("McGill-NLP/stereoset", "intersentence", split="validation"),
        )
    return load_dataset("McGill-NLP/stereoset", subset, split="validation")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
model.eval()


def sentence_pll(sentence: str) -> float:
    """Return pseudo-log-likelihood (PLL) of a sentence."""
    if not sentence:
        return float("-inf")

    encoding = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoding["input_ids"][0].to(device)
    attention_mask = encoding["attention_mask"][0].to(device)

    L = input_ids.size(0)
    pll_sum = 0.0
    count = 0

    for idx in range(1, L - 1):
        if attention_mask[idx] == 0:
            continue
        orig_id = input_ids[idx].item()
        if orig_id in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
            continue

        masked = input_ids.clone()
        masked[idx] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)).logits

        log_probs = torch.nn.functional.log_softmax(logits[0, idx], dim=-1)
        pll_sum += log_probs[orig_id].item()
        count += 1

    return pll_sum if count > 0 else float("-inf")



def extract_S_A_U(ex):
    """Return stereotype, anti-stereotype, unrelated sentences."""
    if isinstance(ex["sentences"], dict):
        s = a = u = None
        for sent, label in zip(ex["sentences"]["sentence"], ex["sentences"]["gold_label"]):
            if label == 0: s = sent
            elif label == 1: a = sent
            elif label == 2: u = sent
        return s, a, u

    if isinstance(ex["sentences"], list):
        s = a = u = None
        for entry in ex["sentences"]:
            lab = entry.get("label")
            if lab == "stereotype": s = entry["sentence"]
            elif lab == "anti-stereotype": a = entry["sentence"]
            elif lab == "unrelated": u = entry["sentence"]
        return s, a, u

    return None, None, None


def evaluate_subset_official(data, max_examples=None):
    PLL_S = []
    PLL_A = []
    PLL_U = []

    for i, ex in enumerate(data):
        if max_examples and i >= max_examples:
            break

        s, a, u = extract_S_A_U(ex)

        if s:
            pll_s = sentence_pll(s)
            if pll_s != float("-inf"):
                PLL_S.append(pll_s)

        if a:
            pll_a = sentence_pll(a)
            if pll_a != float("-inf"):
                PLL_A.append(pll_a)

        if u:
            pll_u = sentence_pll(u)
            if pll_u != float("-inf"):
                PLL_U.append(pll_u)

    # Compute means
    μS = mean(PLL_S) if PLL_S else 0
    μA = mean(PLL_A) if PLL_A else 0
    μU = mean(PLL_U) if PLL_U else 0
    μM = mean(PLL_S + PLL_A) if (PLL_S or PLL_A) else 0

    # Official StereoSet metrics
    if μS + μA == 0:
        SSS = 0
    else:
        SSS = abs((μS / (μS + μA)) - 0.5) * 2

    LMS = 1 - (μU / μM) if μM != 0 else 0
    ICAT = (1 - SSS) * LMS

    return {
        "SSS": round(SSS * 100, 2),
        "LMS": round(LMS * 100, 2),
        "ICAT": round(ICAT * 100, 2),
        "mu_S": μS,
        "mu_A": μA,
        "mu_U": μU,
        "mu_M": μM,
        "num_S": len(PLL_S),
        "num_A": len(PLL_A),
        "num_U": len(PLL_U),
    }



def evaluate_stereoset_official(max_examples=None):
    ds = load_stereoset("all")

    print("\nEvaluating INTRASENTENCE...")
    intra = evaluate_subset_official(ds["intrasentence"], max_examples)

    print("\nEvaluating INTERSENTENCE...")
    inter = evaluate_subset_official(ds["intersentence"], max_examples)

    total_S = intra["num_S"] + inter["num_S"]
    total_A = intra["num_A"] + inter["num_A"]
    total_U = intra["num_U"] + inter["num_U"]

    μS = ((intra["mu_S"] * intra["num_S"]) + (inter["mu_S"] * inter["num_S"])) / total_S
    μA = ((intra["mu_A"] * intra["num_A"]) + (inter["mu_A"] * inter["num_A"])) / total_A
    μU = ((intra["mu_U"] * intra["num_U"]) + (inter["mu_U"] * inter["num_U"])) / total_U
    μM = mean([μS, μA])

    SSS = abs((μS / (μS + μA)) - 0.5) * 2
    LMS = 1 - (μU / μM)
    ICAT = (1 - SSS) * LMS

    combined = {
        "SSS": round(SSS * 100, 2),
        "LMS": round(LMS * 100, 2),
        "ICAT": round(ICAT * 100, 2)
    }

    print("\n==== FINAL StereoSet Official Results ====")
    print("Intrasentence:", intra)
    print("Intersentence:", inter)
    print("Combined:", combined)

    return {
        "intrasentence": intra,
        "intersentence": inter,
        "combined": combined
    }


if __name__ == "__main__":
    results = evaluate_stereoset_official(max_examples=None)
