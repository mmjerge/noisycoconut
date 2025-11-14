"""
Quick test script for Coconut branching with majority voting.

This is a simplified version for quick testing and debugging of the branching
and majority voting functionality. Use this to verify the system works before
running larger experiments.

Usage:
    python quick_branch_test.py
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
# QUICK TEST CONFIGURATION
# ============================================================================

BENCHMARK = "mmlu"  # Options: "gsm8k", "gsm-symbolic", "mmlu"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Small, fast model for testing
NUM_QUESTIONS = 50  # Number of questions to test
NUM_BRANCHES = 5   # Use 5 branches for testing
MAX_NEW_TOKENS = 512  # Shorter for faster testing

# Test just a few noise scales
NOISE_SCALES = [0.0, 0.2, 0.5, 1.0]  # No noise, medium noise, high noise, very high noise
NOISE_TYPE = "gaussian_scaled"
NOISE_DIRECTION = None

# Sampling parameters
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

    dtype = torch.float16
    print(f"Using float16")
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
    print(f"  <|latent|>: {latent_token_id}")
    print(f"  <|start-latent|>: {start_latent_id}")
    print(f"  <|end-latent|>: {end_latent_id}")

    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_benchmark_dataset(benchmark: str = "gsm8k", num_questions: int = None) -> List[Dict[str, str]]:
    """
    Load benchmark dataset (GSM8K, GSM-Symbolic, or MMLU).

    Args:
        benchmark: One of "gsm8k", "gsm-symbolic", or "mmlu"
        num_questions: Number of questions to load (None = all)

    Returns:
        List of dictionaries with "question", "answer", "question_id" fields
    """
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
            # Format multiple choice question
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(item["choices"])])
            formatted_question = f"{item['question']}\n\n{choices_text}"

            # Get correct answer letter
            correct_answer = answer_mapping[item["answer"]]

            questions.append({
                "question": formatted_question,
                "answer": f"#### {correct_answer}",  # Format consistent with GSM8K
                "question_id": i,
                "benchmark": "mmlu",
                "subject": item.get("subject", "unknown"),
                "choices": item["choices"],
                "correct_choice_idx": item["answer"]
            })

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of: gsm8k, gsm-symbolic, mmlu")

    print(f"Loaded {len(questions)} questions from {benchmark.upper()}")
    return questions


def extract_answer(text: str, benchmark: str = "gsm8k") -> str:
    """
    Extract answer from generated text based on benchmark type.

    Args:
        text: Generated text to extract answer from
        benchmark: Type of benchmark (gsm8k, gsm-symbolic, or mmlu)

    Returns:
        Extracted answer string
    """
    if benchmark == "mmlu":
        # For MMLU, look for A, B, C, or D
        # Try various patterns for multiple choice answers
        patterns = [
            r'####\s*([A-D])',  # GSM8K style format
            r'[Tt]he answer is\s*([A-D])',
            r'[Aa]nswer[:\s]+([A-D])',
            r'^([A-D])\.',  # Answer starts with letter
            r'\b([A-D])\b',  # Single letter answer
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()

        # Fallback: find any single capital letter A-D
        letters = re.findall(r'\b([A-D])\b', text)
        if letters:
            return letters[-1].upper()

        return "NO_ANSWER_FOUND"

    else:
        # For GSM8K and GSM-Symbolic, extract numerical answers
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
    """Create input for Coconut model with latent reasoning markers."""
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
    """Extract only the newly generated tokens after <end-latent>."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]

    if len(end_latent_positions) > 0:
        end_latent_pos_expanded = end_latent_positions[0].item() + 8

        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()

    original_len = original_input_ids.shape[1]
    if full_ids.shape[1] > original_len:
        generated_ids = full_ids[0, original_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

    return ""


def majority_vote(answers: List[str]) -> str:
    """Perform majority voting on a list of answers."""
    if not answers:
        return "NO_ANSWER_FOUND"

    vote_counts = Counter(answers)
    most_common = vote_counts.most_common(1)[0][0]
    return most_common


def test_question_with_branching(
    coconut_model,
    tokenizer,
    question: str,
    noise_scale: float,
    start_latent_id: int,
    end_latent_id: int,
    num_branches: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
    noise_type: str,
    noise_direction: str,
    benchmark: str = "gsm8k"
) -> Dict[str, Any]:
    """Test a single question using branching generation with majority voting."""
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
                noise_direction=noise_direction
            )

            branch_texts = []
            branch_answers = []

            for branch_ids in all_branches:
                branch_text = extract_generated_only(
                    tokenizer,
                    branch_ids,
                    original_input_ids,
                    end_latent_id
                )
                branch_answer = extract_answer(branch_text, benchmark)

                branch_texts.append(branch_text)
                branch_answers.append(branch_answer)

            majority_answer = majority_vote(branch_answers)
            vote_distribution = dict(Counter(branch_answers))

            return {
                "success": True,
                "branch_texts": branch_texts,
                "branch_answers": branch_answers,
                "majority_answer": majority_answer,
                "vote_distribution": vote_distribution,
                "num_branches": num_branches,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "branch_texts": [],
                "branch_answers": [],
                "majority_answer": "NO_ANSWER_FOUND",
                "vote_distribution": {},
                "num_branches": num_branches,
                "error": str(e)
            }


def run_quick_test():
    """Run a quick test of branching and majority voting."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("QUICK BRANCHING + MAJORITY VOTING TEST")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Branches: {NUM_BRANCHES}")
    print(f"Noise scales: {NOISE_SCALES}")
    print(f"Noise type: {NOISE_TYPE}")
    print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}")
    print(f"Device: {device}")
    print("="*70)

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
        print(f"Q: {q_data['question']}")
        print(f"\nExpected Answer: {extract_answer(q_data['answer'], BENCHMARK)}")

        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_tests": []
        }

        for noise_scale in NOISE_SCALES:
            print(f"\n{'-'*70}")
            print(f"Testing noise={noise_scale} with {NUM_BRANCHES} branches...")
            print(f"{'-'*70}")

            result = test_question_with_branching(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_scale=noise_scale,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                num_branches=NUM_BRANCHES,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                device=device,
                noise_type=NOISE_TYPE,
                noise_direction=NOISE_DIRECTION,
                benchmark=BENCHMARK
            )

            if result["success"]:
                expected_answer = extract_answer(q_data["answer"], BENCHMARK)
                is_correct = result["majority_answer"] == expected_answer

                print(f"\n✓ Generation successful!")
                print(f"\nBranch answers: {result['branch_answers']}")
                print(f"Vote distribution: {result['vote_distribution']}")
                print(f"\nMajority vote: {result['majority_answer']}")
                print(f"Expected: {expected_answer}")
                print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

                # Show first branch output as example
                if result['branch_texts']:
                    print(f"\nExample output (Branch 1):")
                    print(f"  {result['branch_texts'][0][:200]}...")

                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "branch_texts": result["branch_texts"],  # Full generated text from each branch
                    "branch_answers": result["branch_answers"],
                    "majority_answer": result["majority_answer"],
                    "vote_distribution": result["vote_distribution"],
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "num_branches": result["num_branches"]
                })
            else:
                print(f"\n✗ Error: {result['error']}")
                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "error": result["error"]
                })

        results.append(question_results)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for q_idx, q_result in enumerate(results):
        print(f"\nQuestion {q_idx + 1}: {q_result['question'][:60]}...")
        print(f"Expected: {extract_answer(q_result['expected_answer'], BENCHMARK)}")

        for test in q_result["noise_tests"]:
            noise = test.get("noise_scale", "?")
            if "majority_answer" in test:
                majority = test["majority_answer"]
                correct = "✓" if test.get("is_correct") else "✗"
                print(f"  Noise={noise}: {correct} Majority={majority}, Votes={test['vote_distribution']}")
            else:
                print(f"  Noise={noise}: ✗ Error")

    # Save results
    model_name_safe = MODEL_NAME.replace('/', '_')
    output_file = f"noisy_coconut_{BENCHMARK}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "num_questions": NUM_QUESTIONS,
                "num_branches": NUM_BRANCHES,
                "noise_scales": NOISE_SCALES,
                "noise_type": NOISE_TYPE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P
            },
            "results": results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Allow overriding NUM_QUESTIONS from command line
    if len(sys.argv) > 1:
        NUM_QUESTIONS = int(sys.argv[1])
        print(f"Overriding NUM_QUESTIONS to {NUM_QUESTIONS} from command line")

    run_quick_test()
