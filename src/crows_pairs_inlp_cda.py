import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from data_loading import load_crows_pairs
from cda_utils import swap_gender_terms
from inlp import load_projection, apply_projection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def add_inlp_hook(model, P, info):
    scaler_mean = info.get("scaler_mean")
    scaler_scale = info.get("scaler_scale")

    def hook(module, inputs, output):
        # output: (batch_size, seq_len, hidden_dim)
        hs = output
        bsz, seqlen, dim = hs.shape

        # (B * T, D)
        X = hs.detach().cpu().numpy().reshape(-1, dim)

        # Apply INLP projection (with standardization)
        X_proj = apply_projection(
            X,
            P,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )
        # Back to tensor
        X_proj_t = torch.tensor(X_proj, dtype=hs.dtype, device=hs.device)
        X_proj_t = X_proj_t.view(bsz, seqlen, dim)
        return X_proj_t

    # Register hook on the base BERT model
    handle = model.bert.embeddings.register_forward_hook(hook)
    return handle


def sentence_log_likelihood(model, tokenizer, sentence: str) -> float:
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = enc["input_ids"].to(device)

        seq_len = input_ids.size(1)
        total_ll = 0.0

        # Mask each token (except [CLS], [SEP]) and sum log-probs
        for i in range(1, seq_len - 1):
            masked = input_ids.clone()
            masked[0, i] = tokenizer.mask_token_id

            outputs = model(masked)
            logits = outputs.logits[0, i]
            true_id = input_ids[0, i]

            total_ll += torch.log_softmax(logits, dim=-1)[true_id].item()

        return total_ll


def evaluate_crows_pairs_inlp_cda(
    model_name: str = "bert-base-uncased",
    inlp_path: str = "checkpoints/inlp_projection.joblib",
):
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Load INLP projection
    print(f"Loading INLP projection from: {inlp_path}")
    P, info = load_projection(inlp_path)
    print("Projection shape:", P.shape, "| iters used:", info.get("n_iters", "N/A"))

    # Attach INLP hook
    hook_handle = add_inlp_hook(model, P, info)

    # Load CrowS-Pairs data
    ds = load_crows_pairs()

    total = 0
    stereotype_pref = 0

    for ex in ds:
        sent_more = ex["sent_more"]
        sent_less = ex["sent_less"]

        sent_more_cda = swap_gender_terms(sent_more)
        sent_less_cda = swap_gender_terms(sent_less)

        ll_more = sentence_log_likelihood(model, tokenizer, sent_more_cda)
        ll_less = sentence_log_likelihood(model, tokenizer, sent_less_cda)

        total += 1
        if ll_more > ll_less:
            stereotype_pref += 1

        if total % 100 == 0:
            print(f"Processed {total}/{len(ds)} examples...")

    # Remove the hook to avoid side effects if model is reused
    hook_handle.remove()

    percent = stereotype_pref / total * 100.0
    anti_percent = 100.0 - percent

    print("\n=== CrowS-Pairs INLP + CDA (BERT-base) ===")
    print(f"Stereotypical preference score: {percent:.2f}%")
    print(f"Anti-stereotypical score      : {anti_percent:.2f}%")
    print(f"Total examples: {total}")

    return {
        "method": "INLP + CDA",
        "stereotype_preference": percent,
        "anti_preference": anti_percent,
        "total_examples": total,
    }


if __name__ == "__main__":
    evaluate_crows_pairs_inlp_cda()
