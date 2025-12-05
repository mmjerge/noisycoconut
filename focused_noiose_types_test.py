"""
Focused test script: Compare different noise types with fixed parameters.

This tests Gaussian, Uniform, Orthogonal, and Structured noise to understand
which noise distribution leads to better diversity and accuracy.

Usage:
    python focused_noise_types_test.py [num_questions]
"""

import torch
import json
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any
import re
from collections import Counter

# ============================================================================
# FOCUSED TEST CONFIGURATION - Compare Noise Types
# ============================================================================

BENCHMARK = "gsm8k"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_QUESTIONS = 100

# FOCUSED: Test different noise types with fixed other parameters
NOISE_TYPES = ["gaussian_scaled", "uniform", "orthogonal", "structured"]
NOISE_SCALE = 0.1  # Use optimal from noise scale ablation
NOISE_AT_STEP = 1  # Use optimal from step ablation
NUM_BRANCHES = 5   # Use optimal from paths ablation

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# ============================================================================


def setup_model(model_name: str = MODEL_NAME):
    """Initialize model and add special tokens."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if model_name == "/scratch/mj6ux/.cache/hf_models/gpt-oss-20b":
        dtype = torch.bfloat16
        print("Using bfloat16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        dtype = torch.float16
        print("Using float16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    
    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    base_model.resize_token_embeddings(len(tokenizer))
    
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded!")
    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_benchmark_dataset(benchmark: str = "gsm8k", num_questions: int = None) -> List[Dict[str, str]]:
    """Load benchmark dataset."""
    print(f"\nLoading {benchmark.upper()} test set...")

    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
        questions = []
        for i in range(min(num_questions or len(dataset), len(dataset))):
            item = dataset[i]
            questions.append({
                "question": item["question"],
                "answer": item["answer"],
                "question_id": i,
                "benchmark": "gsm8k"
            })

    elif benchmark == "gsm-symbolic":
        dataset = load_dataset("apple/gsm-symbolic", split="test", trust_remote_code=True)
        questions = []
        for i in range(min(num_questions or len(dataset), len(dataset))):
            item = dataset[i]
            questions.append({
                "question": item["question"],
                "answer": item["answer"],
                "question_id": i,
                "benchmark": "gsm-symbolic"
            })

    elif benchmark == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        questions = []
        answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        for i in range(min(num_questions or len(dataset), len(dataset))):
            item = dataset[i]
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(item["choices"])])
            formatted_question = f"{item['question']}\n\n{choices_text}"
            correct_answer = answer_mapping[item["answer"]]

            questions.append({
                "question": formatted_question,
                "answer": f"#### {correct_answer}",
                "question_id": i,
                "benchmark": "mmlu",
                "subject": item.get("subject", "unknown"),
                "choices": item["choices"],
                "correct_choice_idx": item["answer"]
            })

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"Loaded {len(questions)} questions from {benchmark.upper()}")
    return questions


def extract_answer(text: str, benchmark: str = "gsm8k") -> str:
    """Extract answer from generated text."""
    if benchmark == "mmlu":
        patterns = [
            r'####\s*([A-D])',
            r'[Tt]he answer is\s*([A-D])',
            r'[Aa]nswer[:\s]+([A-D])',
            r'^([A-D])\.',
            r'\b([A-D])\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()

        letters = re.findall(r'\b([A-D])\b', text)
        if letters:
            return letters[-1].upper()

        return "NO_ANSWER_FOUND"

    else:
        def clean_number(num_str: str) -> str:
            return num_str.replace(',', '').replace('$', '').strip()

        patterns = [
            r'\\boxed\{([\d,]+)\}',
            r'####\s*([\d,]+)',
            r'[Tt]he final answer is:?\s*\$?\s*([\d,]+)',
            r'[Ff]inal answer:?\s*\$?\s*([\d,]+)',
            r'[Aa]nswer:?\s*\$?\s*([\d,]+)',
            r'is:?\s*\$?\s*([\d,]+)\s*\.?\s*$',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                return clean_number(matches[-1].group(1))

        numbers = re.findall(r'[\d,]+', text)
        if numbers:
            cleaned_numbers = [clean_number(n) for n in numbers if clean_number(n)]
            if cleaned_numbers:
                return cleaned_numbers[-1]

        return "NO_ANSWER_FOUND"


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int):
    """Create input for Coconut model."""
    messages = [
        {"role": "user", "content": f"{question}\n\nPlease solve this step by step."}
    ]

    question_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    start_token = torch.tensor([[start_latent_id]])
    end_token = torch.tensor([[end_latent_id]])
    input_ids = torch.cat([question_ids, start_token, end_token], dim=1)

    return input_ids


def extract_generated_only(tokenizer, full_ids: torch.Tensor, original_input_ids: torch.Tensor, end_latent_id: int) -> str:
    """Extract only newly generated tokens."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]

    if len(end_latent_positions) > 0:
        end_latent_pos_expanded = end_latent_positions[0].item() + 8
        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    original_len = original_input_ids.shape[1]
    if full_ids.shape[1] > original_len:
        generated_ids = full_ids[0, original_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return ""


def majority_vote(answers: List[str]) -> str:
    """Perform majority voting."""
    if not answers:
        return "NO_ANSWER_FOUND"
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def calculate_path_diversity(branch_answers: List[str]) -> float:
    """Calculate path diversity as the ratio of unique answers to total branches."""
    if not branch_answers:
        return 0.0
    unique_answers = len(set(branch_answers))
    return unique_answers / len(branch_answers)


def test_question_with_branching(
    coconut_model, tokenizer, question: str, noise_type: str,
    noise_scale: float, noise_at_step: int, num_branches: int,
    start_latent_id: int, end_latent_id: int, temperature: float,
    top_p: float, max_new_tokens: int, device: str, benchmark: str = "gsm8k"
) -> Dict[str, Any]:
    """Test a single question using branching generation."""
    coconut_model.eval()
    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        try:
            all_branches = coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                num_branches=num_branches,
                temperature=temperature,
                top_p=top_p,
                noise_scale=noise_scale,
                noise_type=noise_type,
                noise_direction=None,
                noise_at_step=noise_at_step
            )

            branch_texts = []
            branch_answers = []

            for branch_ids in all_branches:
                branch_text = extract_generated_only(tokenizer, branch_ids, original_input_ids, end_latent_id)
                branch_answer = extract_answer(branch_text, benchmark)
                branch_texts.append(branch_text)
                branch_answers.append(branch_answer)

            majority_answer = majority_vote(branch_answers)
            vote_distribution = dict(Counter(branch_answers))
            diversity = calculate_path_diversity(branch_answers)

            return {
                "success": True,
                "branch_answers": branch_answers,
                "majority_answer": majority_answer,
                "vote_distribution": vote_distribution,
                "path_diversity": diversity,
                "noise_type": noise_type,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "branch_answers": [],
                "majority_answer": "NO_ANSWER_FOUND",
                "vote_distribution": {},
                "path_diversity": 0.0,
                "noise_type": noise_type,
                "error": str(e)
            }


def run_focused_test():
    """Run focused test comparing different numbers of paths."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("FOCUSED TEST: Comparing Noise Types")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Testing noise types: {NOISE_TYPES}")
    print(f"Fixed noise scale: {NOISE_SCALE}")
    print(f"Fixed noise at step: {NOISE_AT_STEP}")
    print(f"Fixed num branches: {NUM_BRANCHES}")
    print(f"Device: {device}")
    print("="*70)
    print(f"\n→ This will run {len(NOISE_TYPES)} experiments per question")
    print(f"→ Total: {NUM_QUESTIONS * len(NOISE_TYPES)} experiments\n")

    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model(MODEL_NAME)

    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )

    questions = load_benchmark_dataset(BENCHMARK, NUM_QUESTIONS)
    results = []

    for q_idx, q_data in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {q_idx + 1}/{len(questions)}")
        print(f"{'='*70}")
        print(f"Q: {q_data['question'][:80]}...")
        print(f"Expected Answer: {extract_answer(q_data['answer'], BENCHMARK)}")

        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_type_tests": []
        }

        # Test each noise type
        for noise_type in NOISE_TYPES:
            print(f"\n  → Testing {noise_type}...", end=" ")

            result = test_question_with_branching(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_type=noise_type,
                noise_scale=NOISE_SCALE,
                noise_at_step=NOISE_AT_STEP,
                num_branches=NUM_BRANCHES,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                device=device,
                benchmark=BENCHMARK
            )

            if result["success"]:
                expected_answer = extract_answer(q_data["answer"], BENCHMARK)
                is_correct = result["majority_answer"] == expected_answer

                print(f"Majority: {result['majority_answer']}, Diversity: {result['path_diversity']:.2f}, {'✓' if is_correct else '✗'}")

                question_results["noise_type_tests"].append({
                    "noise_type": noise_type,
                    "branch_answers": result["branch_answers"],
                    "majority_answer": result["majority_answer"],
                    "vote_distribution": result["vote_distribution"],
                    "path_diversity": result["path_diversity"],
                    "is_correct": is_correct
                })
            else:
                print(f"✗ Error: {result['error']}")
                question_results["noise_type_tests"].append({
                    "noise_type": noise_type,
                    "error": result["error"]
                })

        results.append(question_results)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Compute accuracy and diversity by noise type
    noise_stats = {}
    for noise_type in NOISE_TYPES:
        noise_stats[noise_type] = {
            "correct": 0,
            "total": 0,
            "diversity_sum": 0.0,
            "diversity_count": 0
        }

    for q_result in results:
        for test in q_result["noise_type_tests"]:
            if "is_correct" in test:
                noise_type = test["noise_type"]
                noise_stats[noise_type]["total"] += 1
                if test["is_correct"]:
                    noise_stats[noise_type]["correct"] += 1
                noise_stats[noise_type]["diversity_sum"] += test["path_diversity"]
                noise_stats[noise_type]["diversity_count"] += 1

    print(f"\nResults by Noise Type (noise_scale={NOISE_SCALE}):\n")
    print(f"{'Noise Type':<20}{'Accuracy':<15}{'Correct/Total':<18}{'Avg Diversity':<15}")
    print("-" * 68)

    for noise_type in NOISE_TYPES:
        stats = noise_stats[noise_type]
        if stats["total"] > 0:
            accuracy = 100 * stats["correct"] / stats["total"]
            avg_diversity = stats["diversity_sum"] / stats["diversity_count"]
            print(f"{noise_type:<20}{accuracy:>6.1f}%{'':<8}{stats['correct']}/{stats['total']}{'':<10}{avg_diversity:.3f}")

    # Find best noise type
    best_noise_type = max(NOISE_TYPES, key=lambda nt: noise_stats[nt]["correct"] / max(noise_stats[nt]["total"], 1))
    best_accuracy = 100 * noise_stats[best_noise_type]["correct"] / noise_stats[best_noise_type]["total"]
    best_diversity = noise_stats[best_noise_type]["diversity_sum"] / noise_stats[best_noise_type]["diversity_count"]
    
    print(f"\n→ Best performing noise type: {best_noise_type} ({best_accuracy:.1f}% accuracy, {best_diversity:.3f} diversity)")

    # Save results
    model_name_safe = MODEL_NAME.replace('/', '_')
    output_file = f"focused_noise_types_comparison_{BENCHMARK}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "num_questions": NUM_QUESTIONS,
                "noise_types": NOISE_TYPES,
                "noise_scale": NOISE_SCALE,
                "noise_at_step": NOISE_AT_STEP,
                "num_branches": NUM_BRANCHES,
                "temperature": TEMPERATURE,
                "top_p": TOP_P
            },
            "results": results,
            "noise_stats": noise_stats
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        NUM_QUESTIONS = int(sys.argv[1])
        print(f"Overriding NUM_QUESTIONS to {NUM_QUESTIONS} from command line")

    run_focused_test()