import torch
import json
from datetime import datetime
from pathlib import Path
from coconut import Coconut
from run_logits import setup_model, create_coconut_input, extract_generated_only, test_question_with_branching

# ========================== CONFIG ==========================
model_name = "Qwen/Qwen2.5-7B-Instruct"
custom_question = """
What is the sum of the first 50 positive integers? 
Explain your reasoning step by step and put the final answer in \\boxed{}.
"""

noise_scale = 0.2
num_branches = 5
max_new_tokens = 1024
temperature = 0.7
top_p = 0.9

output_dir = Path("~/results/custom_tests").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading model...")
tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model(model_name)

if "gpt-oss" in model_name:
    hidden_layer_idx = 1
else:
    hidden_layer_idx = -1

coconut_model = Coconut(
    base_causallm=base_model,
    latent_token_id=latent_token_id,
    start_latent_id=start_latent_id,
    end_latent_id=end_latent_id,
    eos_token_id=eos_token_id,
    hidden_layer_idx=hidden_layer_idx
).to(device).eval()

input_ids = create_coconut_input(
    tokenizer, 
    custom_question, 
    start_latent_id, 
    end_latent_id, 
    model_name=model_name,
    benchmark="gsm8k"
)

print(f"\nQuestion:\n{custom_question.strip()}\n")
print(f"Running {num_branches} branches with noise={noise_scale}...\n")

result = test_question_with_branching(
    coconut_model=coconut_model,
    tokenizer=tokenizer,
    question=custom_question,
    noise_scale=noise_scale,
    start_latent_id=start_latent_id,
    end_latent_id=end_latent_id,
    num_branches=num_branches,
    temperature=temperature,
    top_p=top_p,
    max_new_tokens=max_new_tokens,
    device=device,
    noise_type="gaussian_scaled",
    noise_direction=None,
    benchmark="gsm8k",
    model_name=model_name
)

# ====================== SAVE LOGITS JSON ======================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logits_file = output_dir / f"logits_custom_{timestamp}.json"

logits_record = {
    "question": custom_question.strip(),
    "noise_scale": noise_scale,
    "num_branches": num_branches,
    "temperature": temperature,
    "top_p": top_p,
    "majority_answer": result["majority_answer"],
    "branches": []
}

for i, (branch_text, branch_answer, log_prob_stats) in enumerate(zip(
        result["branch_texts"], 
        result["branch_answers"], 
        result["branch_log_prob_stats"])):

    logits_record["branches"].append({
        "branch_index": i,
        "branch_answer": branch_answer,
        "branch_weight": result["branch_weights"][i] if i < len(result["branch_weights"]) else None,
        "generated_text": branch_text,
        "num_tokens": log_prob_stats["num_tokens"],
        "mean_log_prob": log_prob_stats["mean_log_prob"],
        "sum_log_prob": log_prob_stats["sum_log_prob"],
        "min_log_prob": log_prob_stats["min_log_prob"],
        "max_log_prob": log_prob_stats["max_log_prob"],
        "raw_log_probs": log_prob_stats["raw_log_probs"],
    })

with open(logits_file, 'w', encoding='utf-8') as f:
    json.dump(logits_record, f, indent=2, ensure_ascii=False)

print("="*80)
print("FINAL AGGREGATED ANSWER:", result["majority_answer"])
print("="*80)
print(f" Logits JSON saved to: {logits_file}")
print(f"   ({len(result['branch_texts'])} branches, full per-token log probs)")