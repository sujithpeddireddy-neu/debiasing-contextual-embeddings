import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from data_loading import load_crows_pairs

# INLP
from inlp import load_projection, apply_projection


# Modified log-likelihood WITH INLP
def sentence_log_likelihood_inlp(model, tokenizer, sentence, P, info):
    model.eval()
    with torch.no_grad():
        # Tokenize and forward
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        log_likelihood = 0.0

        for i in range(1, len(input_ids) - 1):
            masked = input_ids.clone()
            masked[i] = tokenizer.mask_token_id

            outputs = model(masked.unsqueeze(0), output_hidden_states=True)

            # Get hidden state for token i  (shape: 1, seq_len, hidden_size)
            hidden = outputs.hidden_states[-1][0, i, :].cpu().numpy()  # (768,)

            # Apply projection
            hidden_proj = apply_projection(
                hidden.reshape(1, -1),
                P,
                scaler_mean=info["scaler_mean"],
                scaler_scale=info["scaler_scale"],
            )  # shape: (1,768)

            # Back to torch for scoring
            hidden_proj_torch = torch.tensor(hidden_proj, dtype=torch.float32)

            # Replace the hidden state of token i with projected version
            outputs.logits[0, i, :] = torch.matmul(
                hidden_proj_torch,
                model.cls.predictions.decoder.weight.T
            ) + model.cls.predictions.bias

            # Compute log-probability of the true token
            logits = outputs.logits[0, i]
            true_token = input_ids[i]
            log_likelihood += torch.log_softmax(logits, dim=0)[true_token].item()

        return log_likelihood


# Main INLP Debiasing Evaluation
def evaluate_crows_pairs_inlp(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # INLP load projection 
    P, info = load_projection("checkpoints/inlp_projection.joblib")

    ds = load_crows_pairs()

    total = 0
    stereotypical_pref = 0

    for example in ds:
        sent_more = example["sent_more"]
        sent_less = example["sent_less"]

        ll_more = sentence_log_likelihood_inlp(model, tokenizer, sent_more, P, info)
        ll_less = sentence_log_likelihood_inlp(model, tokenizer, sent_less, P, info)

        total += 1
        if ll_more > ll_less:
            stereotypical_pref += 1

    percentage = (stereotypical_pref / total) * 100
    anti_percentage = 100 - percentage

    print("=== CrowS-Pairs INLP-Debiased (BERT-base) ===")
    print(f"Stereotypical preference score: {percentage:.2f}%")
    print(f"Anti-stereotypical score      : {anti_percentage:.2f}%")
    print(f"Total examples: {total}")

    return {
        "stereotype_preference_percentage": percentage,
        "anti_stereotype_percentage": anti_percentage,
        "total_examples": total,
    }
    
if __name__ == "__main__":
    evaluate_crows_pairs_inlp()
