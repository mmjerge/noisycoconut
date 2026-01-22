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
import random
import os
import signal
import random

# ============================================================================
# QUICK TEST CONFIGURATION
# ============================================================================

BENCHMARK = "mmlu"  # Options: "gsm8k", "gsm-symbolic", "mmlu"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_QUESTIONS = 1000
NUM_BRANCHES = 5
MAX_NEW_TOKENS = 2056

NOISE_SCALES = [0.0, 0.2]
NOISE_TYPE = "gaussian_scaled"
NOISE_DIRECTION = None

RANDOM_SEED = 42

# Sampling parameters
TEMPERATURE = 0.7
TOP_P = 0.9

CHECKPOINT_INTERVAL = 100  # Save every N questions

CHECKPOINT_FILE = f"checkpoint_{BENCHMARK}_{MODEL_NAME.replace('/', '_')}.json"

# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"✓ Loaded checkpoint: {len(checkpoint['results'])} questions completed")
        return checkpoint
    return None

def save_checkpoint(checkpoint_file: str, results: List[Dict], completed_ids: set, config: Dict):
    """Save checkpoint to disk."""
    checkpoint = {
        "config": config,
        "completed_question_ids": list(completed_ids),
        "results": results,
        "last_saved": datetime.now().isoformat()
    }
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.replace(temp_file, checkpoint_file)
    print(f"✓ Checkpoint saved: {len(results)} questions completed")

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
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        dtype = torch.float16
        print("Using float16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
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

def load_benchmark_dataset(benchmark: str = "gsm8k", num_questions: int = None, random_seed: int = None) -> List[Dict[str, str]]:
    """
    Load benchmark dataset (GSM8K, GSM-Symbolic, or MMLU).

    Args:
        benchmark: One of "gsm8k", "gsm-symbolic", or "mmlu"
        num_questions: Number of questions to load (None = all)
        random_seed: Seed for random sampling (None = no randomization, uses first N)

    Returns:
        List of dictionaries with "question", "answer", "question_id" fields
    """
    print(f"\nLoading {benchmark.upper()} test set...")

    if benchmark == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif benchmark == "gsm-symbolic":
        dataset = load_dataset("apple/gsm-symbolic", split="test", trust_remote_code=True)
    elif benchmark == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of: gsm8k, gsm-symbolic, mmlu")

    # Determine which indices to use
    total = len(dataset)
    n = min(num_questions or total, total)
    
    if random_seed is not None:
        random.seed(random_seed)
        indices = random.sample(range(total), n)
        print(f"Randomly sampling {n} questions (seed={random_seed})")
    else:
        indices = list(range(n))

    # Build questions list
    questions = []
    answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i in indices:
        item = dataset[i]
        
        if benchmark in ("gsm8k", "gsm-symbolic"):
            questions.append({
                "question": item["question"],
                "answer": item["answer"],
                "question_id": i,
                "benchmark": benchmark
            })
        else:  # mmlu
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

    print(f"Loaded {len(questions)} questions from {benchmark.upper()}")
    return questions


def extract_answer(text: str, benchmark: str = "gsm8k") -> str:
    if benchmark == "mmlu":
        patterns = [
            r'####\s*([A-D])',
            r'[Tt]he answer is\s*([A-D])',
            r'[Ss]o,?\s*the answer is\s*([A-D])',
            r'[Hh]ence,?\s*the answer is\s*([A-D])',
            r'[Tt]herefore,?\s*the answer is\s*([A-D])',
            r'[Ff]inal answer[:\s]+([A-D])',
            r'[Aa]nswer[:\s]+([A-D])',
            r'[Cc]orrect answer is\s*([A-D])',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()

        last_portion = text[-500:] if len(text) > 500 else text
        
        final_patterns = [
            r'is\s+([A-D])\s*[.,]',
            r'\b([A-D])\s*\.\s*$', 
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, last_portion)
            if match:
                return match.group(1).upper()

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


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int, model_name: str = "", benchmark: str = "gsm8k"):
    """Create input for Coconut model with latent reasoning markers."""
    
    if benchmark == "mmlu":
        instruction = f"{question}\n\nPlease solve this step by step. At the end, clearly state your final answer as 'The answer is X' where X is A, B, C, or D."
    else:
        instruction = f"{question}\n\nPlease solve this step by step."
    
    messages = [
        {"role": "user", "content": instruction}
    ]

    question_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    if "gpt-oss" in model_name:
        # Model expects: <|start|>assistant<|channel|>...<|message|>
        # Insert latent markers after adding the channel structure
        channel_token = tokenizer.convert_tokens_to_ids('<|channel|>')
        message_token = tokenizer.convert_tokens_to_ids('<|message|>')
        
        # Encode "analysis" as that's the first channel it expects
        analysis_ids = tokenizer.encode('analysis', add_special_tokens=False)
        
        # Build: question_ids + <|channel|> + analysis + <|message|> + <|start-latent|> + <|end-latent|>
        channel_prefix = torch.tensor([[channel_token] + analysis_ids + [message_token]])
        start_token = torch.tensor([[start_latent_id]])
        end_token = torch.tensor([[end_latent_id]])
        
        input_ids = torch.cat([question_ids, channel_prefix, start_token, end_token], dim=1)
    else:
        # Standard format for other models
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
    benchmark: str = "gsm8k",
    model_name=MODEL_NAME 
) -> Dict[str, Any]:
    """Test a single question using branching generation with majority voting."""
    coconut_model.eval()

    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id, 
                                  model_name=MODEL_NAME, benchmark=BENCHMARK)
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
                noise_direction=noise_direction,
                model_name=model_name
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

    config = {
        "benchmark": BENCHMARK,
        "model": MODEL_NAME,
        "num_questions": NUM_QUESTIONS,
        "num_branches": NUM_BRANCHES,
        "noise_scales": NOISE_SCALES,
        "noise_type": NOISE_TYPE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "random_seed": RANDOM_SEED
    }

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

    # Check for existing checkpoint
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint:
        results = checkpoint["results"]
        completed_ids = set(checkpoint["completed_question_ids"])
        print(f"Resuming from checkpoint: {len(completed_ids)} questions already done")
    else:
        results = []
        completed_ids = set()

    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model(MODEL_NAME)

    if "gpt-oss" in MODEL_NAME:
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
    )

    questions = load_benchmark_dataset(BENCHMARK, NUM_QUESTIONS, random_seed=RANDOM_SEED)

    print(f"Loaded {len(questions)} questions")
    print(f"Completed IDs from checkpoint: {len(completed_ids)}")
    questions_to_process = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Questions to process: {len(questions_to_process)}")
    print(f"First few completed IDs: {list(completed_ids)[:5]}")
    print(f"First few question IDs to process: {[q['question_id'] for q in questions_to_process[:5]]}")

    questions_to_process = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Questions remaining: {len(questions_to_process)}/{len(questions)}")

    def sigterm_handler(signum, frame):
        print("\n⚠ Received SIGTERM, saving checkpoint...")
        save_checkpoint(CHECKPOINT_FILE, results, completed_ids, config)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        for q_idx, q_data in enumerate(questions_to_process):
            global_idx = len(completed_ids) + q_idx + 1
            print(f"\n{'='*70}")
            print(f"Question {global_idx}/{len(questions)} (ID: {q_data['question_id']})")
            print(f"{'='*70}")
            print(f"Q: {q_data['question'][:200]}...")
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

                    if result['branch_texts']:
                        print(f"\nExample output (Branch 1):")
                        print(f"  {result['branch_texts'][0][:200]}...")

                    question_results["noise_tests"].append({
                        "noise_scale": noise_scale,
                        "branch_texts": result["branch_texts"],
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
            completed_ids.add(q_data["question_id"])

            # Checkpoint every N questions
            if len(completed_ids) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(CHECKPOINT_FILE, results, completed_ids, config)

    except KeyboardInterrupt:
        print("\n⚠ Interrupted, saving checkpoint...")
        save_checkpoint(CHECKPOINT_FILE, results, completed_ids, config)
        sys.exit(0)

    # Final checkpoint
    save_checkpoint(CHECKPOINT_FILE, results, completed_ids, config)

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

    # Save final results
    model_name_safe = MODEL_NAME.replace('/', '_')
    output_file = f"noisy_coconut_{BENCHMARK}_{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": config,
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
