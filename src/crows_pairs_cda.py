import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_loading import load_crows_pairs
from cda_utils import swap_gender_terms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def sentence_log_likelihood(model, tokenizer, sentence):
    """
    Computes pseudo-log-likelihood for a sentence using MLM masking.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    input_ids = inputs["input_ids"]

    log_likelihood = 0.0
    seq_len = input_ids.size(1)

    with torch.no_grad():
        for i in range(1, seq_len - 1):
            masked = input_ids.clone()
            masked[0, i] = tokenizer.mask_token_id

            outputs = model(masked)
            logits = outputs.logits[0, i]
            true_token = input_ids[0, i]

            log_likelihood += torch.log_softmax(logits, dim=0)[true_token].item()

    return log_likelihood


def evaluate_crows_pairs_cda(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    ds = load_crows_pairs()

    total = 0
    stereotype_pref = 0

    for ex in ds:
        sent_more = swap_gender_terms(ex["sent_more"])
        sent_less = swap_gender_terms(ex["sent_less"])

        ll_more = sentence_log_likelihood(model, tokenizer, sent_more)
        ll_less = sentence_log_likelihood(model, tokenizer, sent_less)

        if ll_more > ll_less:
            stereotype_pref += 1

        total += 1

        # Progress every 100 examples
        if total % 100 == 0:
            print(f"Processed: {total}/{len(ds)}")

    percent = stereotype_pref / total * 100
    anti_percent = 100 - percent

    print("=== CrowS-Pairs CDA Evaluation Completed ===")
    print(f"Stereotypical preference score: {percent:.2f}%")
    print(f"Anti-stereotypical score      : {anti_percent:.2f}%")
    print(f"Total examples: {total}")

    return percent

evaluate_crows_pairs_cda()
