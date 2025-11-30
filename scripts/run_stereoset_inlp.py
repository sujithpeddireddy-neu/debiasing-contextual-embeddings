# scripts/run_stereoset_inlp.py
"""
Evaluate BERT-base + INLP on StereoSet using the official-style PLL metrics.

This script:
  - loads the StereoSet validation splits (intrasentence + intersentence),
  - wraps BertForMaskedLM with an INLP projection applied to hidden states,
  - computes pseudo-log-likelihood (PLL) for stereotype/anti/unrelated sentences,
  - computes SSS, LMS, ICAT for each subset and the combined score.
"""

from statistics import mean
from typing import Literal, Union, Dict, Any, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, BertForMaskedLM

from src.inlp import load_projection, apply_projection


StereoSubset = Literal["intrasentence", "intersentence", "all"]


def load_stereoset(subset: StereoSubset = "intrasentence") -> Union[Dataset, DatasetDict]:
    """
    Load the StereoSet validation split.
    """
    if subset == "all":
        return DatasetDict(
            intrasentence=load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation"),
            intersentence=load_dataset("McGill-NLP/stereoset", "intersentence", split="validation"),
        )
    return load_dataset("McGill-NLP/stereoset", subset, split="validation")


def extract_S_A_U(ex):
    """
    Extract stereotype (S), anti-stereotype (A), and unrelated (U) sentences
    from a StereoSet example.
    """
    # HF version sometimes stores as a dict-of-lists:
    if isinstance(ex["sentences"], dict):
        s = a = u = None
        for sent, label in zip(ex["sentences"]["sentence"], ex["sentences"]["gold_label"]):
            if label == 0:
                s = sent
            elif label == 1:
                a = sent
            elif label == 2:
                u = sent
        return s, a, u

    # or as a list of dicts:
    if isinstance(ex["sentences"], list):
        s = a = u = None
        for entry in ex["sentences"]:
            lab = entry.get("label")
            if lab == "stereotype":
                s = entry["sentence"]
            elif lab == "anti-stereotype":
                a = entry["sentence"]
            elif lab == "unrelated":
                u = entry["sentence"]
        return s, a, u

    return None, None, None


class INLPBertForMaskedLM:
    """
    Wrapper around BertForMaskedLM that:
      - runs the BERT encoder,
      - standardizes + applies INLP projection to last hidden states,
      - then applies the MLM head.
    Used only for PLL scoring (no training).
    """

    def __init__(self, model_name: str, proj_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Load tokenizer + MLM model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Load INLP projection
        print(f"Loading INLP projection from {proj_path} ...")
        P, info = load_projection(proj_path)
        self.P = torch.tensor(P, dtype=torch.float32, device=self.device)  # (d, d)
        self.scaler_mean = torch.tensor(info["scaler_mean"], dtype=torch.float32, device=self.device)
        self.scaler_scale = torch.tensor(info["scaler_scale"], dtype=torch.float32, device=self.device)
        self.n_iters = info.get("n_iters", None)
        print("Projection shape:", self.P.shape, "| n_iters:", self.n_iters)

    @torch.no_grad()
    def sentence_pll(self, sentence: str) -> float:
        """
        Pseudo log-likelihood for a sentence, but:
          - pass through BERT encoder,
          - standardize + project last hidden states via INLP,
          - run MLM head on projected states.
        """
        if not sentence:
            return float("-inf")

        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)       # (1, L)
        attention_mask = enc["attention_mask"].to(self.device)

        L = input_ids.size(1)
        pll_sum = 0.0
        count = 0

        # Skip [CLS] and [SEP] positions
        for idx in range(1, L - 1):
            orig_id = input_ids[0, idx].item()
            if orig_id == self.tokenizer.pad_token_id:
                continue

            masked_ids = input_ids.clone()
            masked_ids[0, idx] = self.tokenizer.mask_token_id

            # Run encoder to get hidden states
            outputs = self.model.bert(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state  # (1, L, d)

            # Standardize + INLP project
            h_norm = (hidden - self.scaler_mean) / (self.scaler_scale + 1e-8)
            # (1, L, d) @ (d, d) -> (1, L, d)
            h_proj = torch.matmul(h_norm, self.P)

            # MLM head on projected states
            logits = self.model.cls(h_proj)  # (1, L, vocab)
            log_probs = torch.nn.functional.log_softmax(logits[0, idx], dim=-1)
            pll_sum += log_probs[orig_id].item()
            count += 1

        return pll_sum if count > 0 else float("-inf")


def evaluate_subset_inlp(
    data,
    model: INLPBertForMaskedLM,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate one StereoSet subset (intra/inter) using INLP-projected hidden states.
    """
    PLL_S = []
    PLL_A = []
    PLL_U = []

    for i, ex in enumerate(data):
        if max_examples is not None and i >= max_examples:
            break

        s, a, u = extract_S_A_U(ex)

        if s:
            pll_s = model.sentence_pll(s)
            if pll_s != float("-inf"):
                PLL_S.append(pll_s)

        if a:
            pll_a = model.sentence_pll(a)
            if pll_a != float("-inf"):
                PLL_A.append(pll_a)

        if u:
            pll_u = model.sentence_pll(u)
            if pll_u != float("-inf"):
                PLL_U.append(pll_u)

    mu_S = mean(PLL_S) if PLL_S else 0.0
    mu_A = mean(PLL_A) if PLL_A else 0.0
    mu_U = mean(PLL_U) if PLL_U else 0.0
    mu_M = mean(PLL_S + PLL_A) if (PLL_S or PLL_A) else 0.0

    if mu_S + mu_A == 0:
        SSS = 0.0
    else:
        SSS = abs((mu_S / (mu_S + mu_A)) - 0.5) * 2.0

    LMS = 1 - (mu_U / mu_M) if mu_M != 0 else 0.0
    ICAT = (1 - SSS) * LMS

    return {
        "SSS": round(SSS * 100, 2),
        "LMS": round(LMS * 100, 2),
        "ICAT": round(ICAT * 100, 2),
        "mu_S": mu_S,
        "mu_A": mu_A,
        "mu_U": mu_U,
        "mu_M": mu_M,
        "num_S": len(PLL_S),
        "num_A": len(PLL_A),
        "num_U": len(PLL_U),
    }


def evaluate_stereoset_inlp(
    proj_path: str = "checkpoints/inlp_projection.joblib",
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full evaluation:
      - load intrasentence + intersentence,
      - run PLL scoring with INLP,
      - compute StereoSet metrics for each + combined.
    """
    ds = load_stereoset("all")
    inlp_mlm = INLPBertForMaskedLM("bert-base-uncased", proj_path=proj_path)

    print("\nEvaluating INTRASENTENCE with INLP...")
    intra = evaluate_subset_inlp(ds["intrasentence"], inlp_mlm, max_examples=max_examples)

    print("\nEvaluating INTERSENTENCE with INLP...")
    inter = evaluate_subset_inlp(ds["intersentence"], inlp_mlm, max_examples=max_examples)

    total_S = intra["num_S"] + inter["num_S"]
    total_A = intra["num_A"] + inter["num_A"]
    total_U = intra["num_U"] + inter["num_U"]

    mu_S = ((intra["mu_S"] * intra["num_S"]) + (inter["mu_S"] * inter["num_S"])) / total_S
    mu_A = ((intra["mu_A"] * intra["num_A"]) + (inter["mu_A"] * inter["num_A"])) / total_A
    mu_U = ((intra["mu_U"] * intra["num_U"]) + (inter["mu_U"] * inter["num_U"])) / total_U
    mu_M = mean([mu_S, mu_A])

    SSS = abs((mu_S / (mu_S + mu_A)) - 0.5) * 2.0 if (mu_S + mu_A) != 0 else 0.0
    LMS = 1 - (mu_U / mu_M) if mu_M != 0 else 0.0
    ICAT = (1 - SSS) * LMS

    combined = {
        "SSS": round(SSS * 100, 2),
        "LMS": round(LMS * 100, 2),
        "ICAT": round(ICAT * 100, 2),
    }

    print("\n==== StereoSet Results WITH INLP ====")
    print("Intrasentence (INLP):", intra)
    print("Intersentence (INLP):", inter)
    print("Combined (INLP):", combined)

    return {
        "intrasentence": intra,
        "intersentence": inter,
        "combined": combined,
    }


if __name__ == "__main__":
    # Set max_examples to 100 for a quick run, or None for full StereoSet
    evaluate_stereoset_inlp(
        proj_path="checkpoints/inlp_projection.joblib",
        max_examples=None,
    )
