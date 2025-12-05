"""
Focused test script: Compare different aggregation methods with fixed parameters.

This tests Majority Voting, Noise-weighted, and Accuracy-weighted aggregation
to understand which method best combines multiple reasoning paths.

Note: For Accuracy-weighted, we use a simple heuristic based on answer confidence
(vote count) as we don't have per-path accuracy during inference.

Usage:
    python focused_aggregation_test.py [num_questions]
"""

import torch
import json
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any, Tuple
import re
from collections import Counter
import numpy as np

# ============================================================================
# FOCUSED TEST CONFIGURATION - Compare Aggregation Methods
# ============================================================================

BENCHMARK = "gsm8k"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_QUESTIONS = 100

# FOCUSED: Test different aggregation methods with fixed other parameters
AGGREGATION_METHODS = ["majority_voting", "noise_weighted", "confidence_weighted"]
NOISE_SCALE = 0.1  # Use optimal from noise scale ablation
NOISE_AT_STEP = 1  # Use optimal from step ablation
NUM_BRANCHES = 5   # Use optimal from paths ablation
NOISE_TYPE = "gaussian_scaled"  # Use optimal from noise type ablation

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
    """Perform simple majority voting."""
    if not answers:
        return "NO_ANSWER_FOUND"
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]


def noise_weighted_aggregation(answers: List[str], noise_scale: float) -> str:
    """
    Aggregate answers with inverse noise weighting.
    Lower noise branches get higher weight.
    
    For this implementation, we assume branches created later have slightly 
    higher effective noise due to accumulation, so we weight earlier branches more.
    """
    if not answers:
        return "NO_ANSWER_FOUND"
    
    # Weight branches inversely by their implicit noise level
    # Earlier branches get higher weight
    weights = [1.0 / (1.0 + i * noise_scale) for i in range(len(answers))]
    
    # Weighted voting
    weighted_votes = {}
    for answer, weight in zip(answers, weights):
        weighted_votes[answer] = weighted_votes.get(answer, 0.0) + weight
    
    # Return answer with highest weighted vote
    if weighted_votes:
        return max(weighted_votes.items(), key=lambda x: x[1])[0]
    return "NO_ANSWER_FOUND"


def confidence_weighted_aggregation(answers: List[str]) -> str:
    """
    Aggregate using confidence weighting based on vote counts.
    Answers that appear more frequently get exponentially higher weight.
    """
    if not answers:
        return "NO_ANSWER_FOUND"
    
    vote_counts = Counter(answers)
    
    # Apply confidence weighting: weight = count^2
    # This gives much higher weight to answers with strong agreement
    weighted_votes = {answer: count ** 2 for answer, count in vote_counts.items()}
    
    # Return answer with highest weighted score
    # (In practice, this often reduces to majority voting, but handles ties differently)
    return max(weighted_votes.items(), key=lambda x: x[1])[0]


def apply_aggregation(answers: List[str], method: str, noise_scale: float = 0.1) -> str:
    """Apply the specified aggregation method."""
    if method == "majority_voting":
        return majority_vote(answers)
    elif method == "noise_weighted":
        return noise_weighted_aggregation(answers, noise_scale)
    elif method == "confidence_weighted":
        return confidence_weighted_aggregation(answers)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def test_question_with_branching(
    coconut_model, tokenizer, question: str, num_branches: int,
    noise_scale: float, noise_at_step: int, noise_type: str,
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

            branch_answers = []
            for branch_ids in all_branches:
                branch_text = extract_generated_only(tokenizer, branch_ids, original_input_ids, end_latent_id)
                branch_answer = extract_answer(branch_text, benchmark)
                branch_answers.append(branch_answer)

            return {
                "success": True,
                "branch_answers": branch_answers,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "branch_answers": [],
                "error": str(e)
            }


def run_focused_test():
    """Run focused test comparing different aggregation methods."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("FOCUSED TEST: Comparing Aggregation Methods")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Testing aggregation methods: {AGGREGATION_METHODS}")
    print(f"Fixed noise scale: {NOISE_SCALE}")
    print(f"Fixed noise type: {NOISE_TYPE}")
    print(f"Fixed noise at step: {NOISE_AT_STEP}")
    print(f"Fixed num branches: {NUM_BRANCHES}")
    print(f"Device: {device}")
    print("="*70)
    print(f"\n→ Generating {NUM_BRANCHES} branches per question")
    print(f"→ Testing {len(AGGREGATION_METHODS)} aggregation methods")
    print(f"→ Total: {NUM_QUESTIONS} questions\n")

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
        expected_answer = extract_answer(q_data["answer"], BENCHMARK)
        print(f"Expected Answer: {expected_answer}")

        # Generate branches once
        print(f"\n  → Generating {NUM_BRANCHES} branches...")
        result = test_question_with_branching(
            coconut_model=coconut_model,
            tokenizer=tokenizer,
            question=q_data["question"],
            num_branches=NUM_BRANCHES,
            noise_scale=NOISE_SCALE,
            noise_at_step=NOISE_AT_STEP,
            noise_type=NOISE_TYPE,
            start_latent_id=start_latent_id,
            end_latent_id=end_latent_id,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
            benchmark=BENCHMARK
        )

        if not result["success"]:
            print(f"  ✗ Error generating branches: {result['error']}")
            results.append({
                "question_id": q_data["question_id"],
                "question": q_data["question"],
                "expected_answer": q_data["answer"],
                "error": result["error"]
            })
            continue

        branch_answers = result["branch_answers"]
        print(f"  Branch answers: {branch_answers}")

        # Test each aggregation method on the same branches
        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "branch_answers": branch_answers,
            "aggregation_tests": []
        }

        print(f"\n  Testing aggregation methods:")
        for method in AGGREGATION_METHODS:
            aggregated_answer = apply_aggregation(branch_answers, method, NOISE_SCALE)
            is_correct = aggregated_answer == expected_answer
            
            print(f"    {method:<25} → {aggregated_answer:<15} {'✓' if is_correct else '✗'}")

            question_results["aggregation_tests"].append({
                "method": method,
                "aggregated_answer": aggregated_answer,
                "is_correct": is_correct
            })

        results.append(question_results)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Compute accuracy by aggregation method
    agg_stats = {}
    for method in AGGREGATION_METHODS:
        agg_stats[method] = {"correct": 0, "total": 0}

    for q_result in results:
        if "aggregation_tests" in q_result:
            for test in q_result["aggregation_tests"]:
                method = test["method"]
                agg_stats[method]["total"] += 1
                if test["is_correct"]:
                    agg_stats[method]["correct"] += 1

    print(f"\nResults by Aggregation Method:\n")
    print(f"{'Method':<25}{'Accuracy':<15}{'Correct/Total':<20}")
    print("-" * 60)

    for method in AGGREGATION_METHODS:
        stats = agg_stats[method]
        if stats["total"] > 0:
            accuracy = 100 * stats["correct"] / stats["total"]
            print(f"{method:<25}{accuracy:>6.1f}%{'':<8}{stats['correct']}/{stats['total']}")

    # Find best method
    best_method = max(AGGREGATION_METHODS, key=lambda m: agg_stats[m]["correct"] / max(agg_stats[m]["total"], 1))
    best_accuracy = 100 * agg_stats[best_method]["correct"] / agg_stats[best_method]["total"]
    
    print(f"\n→ Best performing aggregation method: {best_method} ({best_accuracy:.1f}% accuracy)")

    # Save results
    model_name_safe = MODEL_NAME.replace('/', '_')
    output_file = f"focused_aggregation_comparison_{BENCHMARK}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "num_questions": NUM_QUESTIONS,
                "aggregation_methods": AGGREGATION_METHODS,
                "num_branches": NUM_BRANCHES,
                "noise_scale": NOISE_SCALE,
                "noise_type": NOISE_TYPE,
                "noise_at_step": NOISE_AT_STEP,
                "temperature": TEMPERATURE,
                "top_p": TOP_P
            },
            "results": results,
            "agg_stats": agg_stats
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