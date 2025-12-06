import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from statistics import mean
from cda_utils import swap_gender_terms

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################
# Load BERT MLM model
###############################################
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
model.eval()

###############################################
# PLL computation
###############################################
def sentence_pll(sentence):
    encoding = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoding["input_ids"][0].to(device)
    attention = encoding["attention_mask"][0].to(device)

    L = input_ids.size(0)
    pll_sum, count = 0.0, 0

    for i in range(1, L-1):
        if attention[i] == 0:
            continue
        token_id = input_ids[i].item()
        if token_id in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
            continue

        masked = input_ids.clone()
        masked[i] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked.unsqueeze(0), attention_mask=attention.unsqueeze(0)).logits[0, i]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        pll_sum += log_probs[token_id].item()
        count += 1

    return pll_sum if count > 0 else float("-inf")

###############################################
# Extract S / A / U and apply CDA
###############################################
def extract_sau_cda(example):
    S = A = U = None

    for sent, label in zip(example["sentences"]["sentence"], example["sentences"]["gold_label"]):

        swapped = swap_gender_terms(sent)

        if label == 0:
            S = swapped
        elif label == 1:
            A = swapped
        elif label == 2:
            U = swapped

    return S, A, U

###############################################
# Evaluate split
###############################################
def evaluate_split(dataset, max_examples=None):
    PLL_S, PLL_A, PLL_U = [], [], []

    for i, ex in enumerate(dataset):
        if max_examples and i >= max_examples:
            break

        s, a, u = extract_sau_cda(ex)

        if s: PLL_S.append(sentence_pll(s))
        if a: PLL_A.append(sentence_pll(a))
        if u: PLL_U.append(sentence_pll(u))

    μS = mean(PLL_S)
    μA = mean(PLL_A)
    μU = mean(PLL_U)
    μM = mean([μS, μA])

    SSS = abs((μS / (μS + μA)) - 0.5) * 2
    LMS = 1 - (μU / μM)
    ICAT = (1 - SSS) * LMS

    return {
        "SSS": round(SSS * 100, 2),
        "LMS": round(LMS * 100, 2),
        "ICAT": round(ICAT * 100, 2),
        "mu_S": μS,
        "mu_A": μA,
        "mu_U": μU,
        "num_S": len(PLL_S)
    }

###############################################
# Main StereoSet Evaluation (CDA version)
###############################################
def evaluate_stereoset_cda(max_examples=None):
    intra = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    inter = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")

    print("\nRunning CDA on INTRASENTENCE...")
    intra_scores = evaluate_split(intra, max_examples)

    print("\nRunning CDA on INTERSENTENCE...")
    inter_scores = evaluate_split(inter, max_examples)

    print("\n===== CDA StereoSet Scores =====")
    print("Intrasentence:", intra_scores)
    print("Intersentence:", inter_scores)

    return {"intrasentence": intra_scores, "intersentence": inter_scores}


if __name__ == "__main__":
    results = evaluate_stereoset_cda(max_examples=None)
