"""
Focused test script: Compare noise injection at different steps with FIXED noise scale.

This simplified version tests steps 1-4 with a single noise scale (0.2) to quickly
understand how the timing of noise injection affects performance.

Usage:
    python focused_step_test.py [num_questions]
    
Examples:
    python focused_step_test.py       # Run on 100 questions
    python focused_step_test.py 5     # Quick test on 5 questions
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

BENCHMARK = "gsm8k" 
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_QUESTIONS = 100 

NOISE_AT_STEPS = [1, 2, 3, 4]  
NOISE_SCALE = 0.2  
NOISE_TYPE = "gaussian_scaled"

NUM_BRANCHES = 5
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9


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
    noise_at_step: int,
    start_latent_id: int,
    end_latent_id: int,
    num_branches: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
    noise_type: str,
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
                noise_direction=None,
                noise_at_step=noise_at_step
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
                "noise_at_step": noise_at_step,
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
                "noise_at_step": noise_at_step,
                "error": str(e)
            }


def run_focused_test():
    """Run focused test comparing different noise injection steps."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("FOCUSED TEST: Comparing Noise Injection Steps")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Branches: {NUM_BRANCHES}")
    print(f"Testing steps: {NOISE_AT_STEPS}")
    print(f"Fixed noise scale: {NOISE_SCALE}")
    print(f"Noise type: {NOISE_TYPE}")
    print(f"Device: {device}")
    print("="*70)
    print(f"\n→ This will run {len(NOISE_AT_STEPS)} experiments per question")
    print(f"→ Total: {NUM_QUESTIONS * len(NOISE_AT_STEPS)} experiments\n")

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
            "step_tests": []
        }

        # Test each noise_at_step with fixed noise_scale
        for noise_at_step in NOISE_AT_STEPS:
            print(f"\n  → Testing step {noise_at_step}...", end=" ")

            result = test_question_with_branching(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_scale=NOISE_SCALE,
                noise_at_step=noise_at_step,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                num_branches=NUM_BRANCHES,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                device=device,
                noise_type=NOISE_TYPE,
                benchmark=BENCHMARK
            )

            if result["success"]:
                expected_answer = extract_answer(q_data["answer"], BENCHMARK)
                is_correct = result["majority_answer"] == expected_answer

                print(f"Majority: {result['majority_answer']}, {'✓' if is_correct else '✗'}")

                question_results["step_tests"].append({
                    "noise_at_step": noise_at_step,
                    "branch_answers": result["branch_answers"],
                    "majority_answer": result["majority_answer"],
                    "vote_distribution": result["vote_distribution"],
                    "is_correct": is_correct
                })
            else:
                print(f"✗ Error: {result['error']}")
                question_results["step_tests"].append({
                    "noise_at_step": noise_at_step,
                    "error": result["error"]
                })

        results.append(question_results)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Compute accuracy by step
    step_accuracy = {}
    for step in NOISE_AT_STEPS:
        step_accuracy[step] = {"correct": 0, "total": 0}

    for q_result in results:
        for test in q_result["step_tests"]:
            if "is_correct" in test:
                step = test["noise_at_step"]
                step_accuracy[step]["total"] += 1
                if test["is_correct"]:
                    step_accuracy[step]["correct"] += 1

    print(f"\nAccuracy by Noise Injection Step (noise_scale={NOISE_SCALE}):\n")
    print(f"{'Step':<10}{'Accuracy':<15}{'Correct/Total':<20}")
    print("-" * 45)

    for step in NOISE_AT_STEPS:
        stats = step_accuracy[step]
        if stats["total"] > 0:
            accuracy = 100 * stats["correct"] / stats["total"]
            print(f"Step {step:<6}{accuracy:>6.1f}%{'':<8}{stats['correct']}/{stats['total']}")

    # Find best step
    best_step = max(NOISE_AT_STEPS, key=lambda s: step_accuracy[s]["correct"] / max(step_accuracy[s]["total"], 1))
    best_accuracy = 100 * step_accuracy[best_step]["correct"] / step_accuracy[best_step]["total"]
    print(f"\n→ Best performing step: {best_step} ({best_accuracy:.1f}% accuracy)")

    # Save results
    model_name_safe = MODEL_NAME.replace('/', '_')
    output_file = f"focused_step_comparison_{BENCHMARK}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "num_questions": NUM_QUESTIONS,
                "num_branches": NUM_BRANCHES,
                "noise_at_steps": NOISE_AT_STEPS,
                "noise_scale": NOISE_SCALE,
                "noise_type": NOISE_TYPE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P
            },
            "results": results,
            "step_accuracy": step_accuracy
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

    run_focused_test()