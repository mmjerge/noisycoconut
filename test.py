"""
Test Coconut model with varying noise types and levels on GSM8K.

This script tests different noise injection strategies to understand system breakdown:

NOISE TYPES:
1. gaussian: Standard Gaussian noise N(0, scale^2) - baseline
2. gaussian_scaled: Gaussian noise scaled relative to hidden state norm
3. uniform: Uniform random noise in [-scale, scale]
4. orthogonal: Noise perpendicular to hidden state direction
5. targeted: Noise along (or opposite to) hidden state direction

USAGE:
Edit the main() function to choose noise type, direction, and scales.
Simply uncomment the desired configuration option.

Results are saved to JSON for analysis.
"""

import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from datasets import load_dataset
from typing import List, Dict, Any
import re

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen3-0.6B"):
    """Initialize model and add special tokens."""
    print(f"Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16
        torch_dtype=torch.bfloat16,  # Use FP16 for efficiency
        device_map="auto",  # Automatically handle device placement
        trust_remote_code=True
    )
    
    special_tokens = {
        'additional_special_tokens': ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    
    # Resize model embeddings to accommodate new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Get token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    start_latent_id = tokenizer.convert_tokens_to_ids('<|start-latent|>')
    end_latent_id = tokenizer.convert_tokens_to_ids('<|end-latent|>')
    eos_token_id = tokenizer.eos_token_id
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Special token IDs:")
    print(f"  <|latent|>: {latent_token_id}")
    print(f"  <|start-latent|>: {start_latent_id}")
    print(f"  <|end-latent|>: {end_latent_id}")
    print(f"  EOS: {eos_token_id}")
    
    return tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id


def load_gsm8k_questions(num_questions: int = 5) -> List[Dict[str, str]]:
    """Load first N questions from GSM8K dataset."""
    print(f"\nLoading first {num_questions} questions from GSM8K...")
    
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        questions = []
        
        for i in range(min(num_questions, len(dataset))):
            item = dataset[i]
            questions.append({
                "question": item["question"],
                "answer": item["answer"],
                "question_id": i
            })
        
        print(f"Loaded {len(questions)} questions")
        return questions
    
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Using fallback example questions...")
        
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "#### 18",
                "question_id": 0
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "#### 3",
                "question_id": 1
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": "#### 70000",
                "question_id": 2
            },
            {
                "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                "answer": "#### 540",
                "question_id": 3
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives her flock of chickens 15 cups in the morning and 25 cups in the afternoon. How many cups does she need in the final meal if she has 20 chickens that each eat 3 cups per day?",
                "answer": "#### 20",
                "question_id": 4
            }
        ]


def extract_numerical_answer(text: str) -> str:
    """Extract numerical answer from generated text."""

    # Helper to clean number strings (remove commas, dollar signs, etc.)
    def clean_number(num_str: str) -> str:
        return num_str.replace(',', '').replace('$', '').strip()

    # Patterns in priority order
    # Using [\d,]+ to match numbers with commas (e.g., "70,000")
    patterns = [
        r'\\boxed\{([\d,]+)\}',  # LaTeX boxed format
        r'####\s*([\d,]+)',  # GSM8K format
        r'[Tt]he final answer is:?\s*\$?\s*([\d,]+)',  # "The final answer is: 70,000"
        r'[Ff]inal answer:?\s*\$?\s*([\d,]+)',  # "Final answer: 70,000"
        r'[Aa]nswer:?\s*\$?\s*([\d,]+)',  # "Answer: 70,000"
        r'profit of \$?\s*([\d,]+)',  # "profit of $70,000"
        r'made.*?\$?\s*([\d,]+)\s*\.?\s*$',  # "made a profit of $70,000" at end
        r'is:?\s*\$?\s*([\d,]+)\s*\.?\s*$',  # "is $70,000" at end of text
        r'total.*?is:?\s*\$?\s*([\d,]+)',  # "total is $70,000"
        r'=\s*\$?\s*([\d,]+)',  # "= $70,000"
    ]

    # Try each pattern, taking the LAST match to prefer final answers
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Take the last (rightmost) match as it's most likely the final answer
            last_match = matches[-1]
            return clean_number(last_match.group(1))

    # Fallback: extract all numbers (with commas) and return the last substantial one
    # Match sequences of digits with optional commas
    numbers = re.findall(r'[\d,]+', text)
    if numbers:
        # Clean all numbers and filter out fragments
        cleaned_numbers = [clean_number(n) for n in numbers if clean_number(n)]
        if cleaned_numbers:
            # Return the last number found
            return cleaned_numbers[-1]

    return "NO_ANSWER_FOUND"


def create_coconut_input(tokenizer, question: str, start_latent_id: int, end_latent_id: int):
    """Create input for Coconut model with latent reasoning markers."""
    # Use proper chat template for instruct models
    messages = [
        {"role": "user", "content": f"{question}\n\nPlease solve this step by step."}
    ]

    # Apply chat template to get properly formatted prompt
    # add_generation_prompt=True adds the assistant prefix
    question_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
        add_special_tokens=True
    )

    # Add latent markers for continuous thought
    start_token = torch.tensor([[start_latent_id]])
    end_token = torch.tensor([[end_latent_id]])

    input_ids = torch.cat([question_ids, start_token, end_token], dim=1)

    return input_ids


def extract_generated_only(tokenizer, full_ids: torch.Tensor, original_input_ids: torch.Tensor, end_latent_id: int) -> str:
    """Extract only the newly generated tokens after <end-latent>."""
    end_latent_positions = (original_input_ids[0] == end_latent_id).nonzero(as_tuple=True)[0]
    
    if len(end_latent_positions) > 0:
        # Account for 8 virtual latent tokens inserted during TRUE METHOD
        end_latent_pos_expanded = end_latent_positions[0].item() + 8
        
        if full_ids.shape[1] > end_latent_pos_expanded + 1:
            generated_ids = full_ids[0, end_latent_pos_expanded + 1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text.strip()
    
    # Fallback
    original_len = original_input_ids.shape[1]
    if full_ids.shape[1] > original_len:
        generated_ids = full_ids[0, original_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    
    return ""


def majority_vote(answers: List[str]) -> str:
    """
    Perform majority voting on a list of answers.

    Args:
        answers: List of answer strings

    Returns:
        The most common answer, or the first answer if there's a tie
    """
    from collections import Counter

    if not answers:
        return "NO_ANSWER_FOUND"

    # Count occurrences of each answer
    vote_counts = Counter(answers)

    # Get the most common answer
    most_common = vote_counts.most_common(1)[0][0]

    return most_common


def test_question_with_noise(
    coconut_model,
    tokenizer,
    question: str,
    noise_scale: float,
    start_latent_id: int,
    end_latent_id: int,
    max_new_tokens: int = 1024,
    device: str = 'cuda',
    noise_type: str = 'gaussian',
    noise_direction: str = None,
    apply_noise_to_all_passes: bool = False
) -> Dict[str, Any]:
    """Test a single question with specific noise configuration."""

    coconut_model.eval()

    # Create input
    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)

    # Generate response WITH NOISE
    with torch.no_grad():
        try:
            generated_ids = coconut_model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                noise_scale=noise_scale,
                noise_type=noise_type,
                noise_direction=noise_direction,
                apply_noise_to_all_passes=apply_noise_to_all_passes
            )

            # Extract ONLY the generated portion
            generated_text = extract_generated_only(
                tokenizer,
                generated_ids,
                original_input_ids,
                end_latent_id
            )

            return {
                "success": True,
                "generated_text": generated_text,
                "num_tokens": generated_ids.shape[1],
                "num_new_tokens": generated_ids.shape[1] - original_input_ids.shape[1],
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "generated_text": "",
                "num_tokens": 0,
                "num_new_tokens": 0,
                "error": str(e)
            }


def test_question_with_branching(
    coconut_model,
    tokenizer,
    question: str,
    noise_scale: float,
    start_latent_id: int,
    end_latent_id: int,
    num_branches: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    device: str = 'cuda',
    noise_type: str = 'gaussian',
    noise_direction: str = None
) -> Dict[str, Any]:
    """
    Test a single question using branching generation with majority voting.

    Args:
        coconut_model: The Coconut model
        tokenizer: Tokenizer
        question: Question text
        noise_scale: Noise scale to apply
        start_latent_id: Start latent token ID
        end_latent_id: End latent token ID
        num_branches: Number of branches to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_new_tokens: Max tokens to generate
        device: Device to use
        noise_type: Type of noise
        noise_direction: Direction for directional noise

    Returns:
        Dictionary with results including all branch answers and majority vote
    """
    coconut_model.eval()

    # Create input
    input_ids = create_coconut_input(tokenizer, question, start_latent_id, end_latent_id)
    original_input_ids = input_ids.clone()
    input_ids = input_ids.to(device)

    # Generate multiple branches WITH NOISE
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

            # Extract text and answers from all branches
            branch_texts = []
            branch_answers = []

            for branch_ids in all_branches:
                branch_text = extract_generated_only(
                    tokenizer,
                    branch_ids,
                    original_input_ids,
                    end_latent_id
                )
                branch_answer = extract_numerical_answer(branch_text)

                branch_texts.append(branch_text)
                branch_answers.append(branch_answer)

            # Perform majority voting
            majority_answer = majority_vote(branch_answers)

            # Calculate vote distribution
            from collections import Counter
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


def run_noise_experiment(
    num_questions: int = 5,
    noise_scales: List[float] = None,
    max_new_tokens: int = 1024,
    model_name: str = "Qwen/Qwen3-0.6B",
    noise_type: str = "gaussian",
    noise_direction: str = None,
    apply_noise_to_all_passes: bool = False
) -> Dict[str, Any]:
    """
    Run complete noise experiment on GSM8K questions.

    For each question at each noise level, runs:
    1. Baseline run 1 (standard autoregressive, no noise, no Coconut)
    2. Baseline run 2 (standard autoregressive, no noise, no Coconut)
    3. Coconut run with noise (latent space reasoning with noise)
    """

    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print(f"COCONUT NOISE ROBUSTNESS EXPERIMENT - {model_name}")
    print("="*70)
    print(f"Testing {len(noise_scales)} noise scales on {num_questions} GSM8K questions")
    print(f"Per question per noise level: 2 baseline + 1 coconut")
    print(f"Noise type: {noise_type}")
    if noise_direction:
        print(f"Noise direction: {noise_direction}")
    print(f"Noise scales: {noise_scales}")
    print(f"Device: {device}")
    print(f"Max new tokens: {max_new_tokens}")

    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model_and_tokenizer(model_name)

    # Keep base_model for baseline runs
    # Create coconut_model for coconut runs
    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )

    # Load questions
    questions = load_gsm8k_questions(num_questions)

    # Run experiments
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "noise_scales": noise_scales,
            "noise_type": noise_type,
            "noise_direction": noise_direction,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "model": model_name,
            "latent_passes": 8
        },
        "questions": []
    }

    for q_idx, q_data in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {q_idx + 1}/{len(questions)}")
        print(f"{'='*70}")
        print(f"Q: {q_data['question'][:100]}...")
        print(f"Expected Answer: {q_data['answer']}")

        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_tests": []
        }

        for noise_scale in noise_scales:
            print(f"\n  Testing noise={noise_scale}...")
            expected_answer = extract_numerical_answer(q_data["answer"])

            # Prepare standard input for baseline runs (no latent markers)
            messages = [{"role": "user", "content": f"{q_data['question']}\n\nPlease solve this step by step."}]
            baseline_input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt',
                add_special_tokens=True
            ).to(device)

            # BASELINE RUN 1 - Standard autoregressive (no noise, no Coconut)
            print(f"    Baseline 1...", end=" ")
            with torch.no_grad():
                baseline_outputs1 = base_model.generate(
                    input_ids=baseline_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            baseline_text1 = tokenizer.decode(baseline_outputs1[0, baseline_input_ids.shape[1]:], skip_special_tokens=True)
            baseline_answer1 = extract_numerical_answer(baseline_text1)
            baseline_correct1 = baseline_answer1 == expected_answer
            print(f"[{baseline_answer1}] {'✓' if baseline_correct1 else '✗'}")

            # BASELINE RUN 2 - Standard autoregressive (no noise, no Coconut)
            print(f"    Baseline 2...", end=" ")
            with torch.no_grad():
                baseline_outputs2 = base_model.generate(
                    input_ids=baseline_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            baseline_text2 = tokenizer.decode(baseline_outputs2[0, baseline_input_ids.shape[1]:], skip_special_tokens=True)
            baseline_answer2 = extract_numerical_answer(baseline_text2)
            baseline_correct2 = baseline_answer2 == expected_answer
            print(f"[{baseline_answer2}] {'✓' if baseline_correct2 else '✗'}")

            # COCONUT RUN - With latent space reasoning and noise
            print(f"    Coconut...", end=" ")
            result = test_question_with_noise(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_scale=noise_scale,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                max_new_tokens=max_new_tokens,
                device=device,
                noise_type=noise_type,
                noise_direction=noise_direction,
                apply_noise_to_all_passes=apply_noise_to_all_passes
            )

            if result["success"]:
                coconut_answer = extract_numerical_answer(result["generated_text"])
                coconut_correct = coconut_answer == expected_answer
                print(f"[{coconut_answer}] {'✓' if coconut_correct else '✗'}")

                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    # Baseline 1
                    "baseline1_text": baseline_text1,
                    "baseline1_answer": baseline_answer1,
                    "baseline1_correct": baseline_correct1,
                    # Baseline 2
                    "baseline2_text": baseline_text2,
                    "baseline2_answer": baseline_answer2,
                    "baseline2_correct": baseline_correct2,
                    # Coconut
                    "coconut_text": result["generated_text"],
                    "coconut_answer": coconut_answer,
                    "coconut_correct": coconut_correct,
                    # Common fields
                    "expected_answer": expected_answer,
                    "num_tokens": result["num_tokens"],
                    "num_new_tokens": result["num_new_tokens"],
                    "success": True,
                    "error": None
                })
            else:
                print(f"✗ Error: {result['error']}")

                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    # Baseline 1
                    "baseline1_text": baseline_text1,
                    "baseline1_answer": baseline_answer1,
                    "baseline1_correct": baseline_correct1,
                    # Baseline 2
                    "baseline2_text": baseline_text2,
                    "baseline2_answer": baseline_answer2,
                    "baseline2_correct": baseline_correct2,
                    # Coconut (failed)
                    "coconut_text": "",
                    "coconut_answer": None,
                    "coconut_correct": False,
                    # Common fields
                    "expected_answer": expected_answer,
                    "num_tokens": 0,
                    "num_new_tokens": 0,
                    "success": False,
                    "error": result["error"]
                })

        results["questions"].append(question_results)

    return results


def run_branching_experiment(
    num_questions: int = 5,
    noise_scales: List[float] = None,
    num_branches: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    model_name: str = "Qwen/Qwen3-0.6B",
    noise_type: str = "gaussian",
    noise_direction: str = None
) -> Dict[str, Any]:
    """
    Run complete noise experiment using branching and majority voting.

    Args:
        num_questions: Number of GSM8K questions to test
        noise_scales: List of noise scales to test
        num_branches: Number of branches for each question
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        max_new_tokens: Max tokens to generate per branch
        model_name: Model name to load
        noise_type: Type of noise to apply
        noise_direction: Direction for directional noise

    Returns:
        Dictionary with experiment results
    """
    if noise_scales is None:
        noise_scales = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*70)
    print(f"COCONUT BRANCHING + MAJORITY VOTING EXPERIMENT - {model_name}")
    print("="*70)
    print(f"Testing {len(noise_scales)} noise scales on {num_questions} GSM8K questions")
    print(f"Branching: {num_branches} branches per question")
    print(f"Noise type: {noise_type}")
    if noise_direction:
        print(f"Noise direction: {noise_direction}")
    print(f"Noise scales: {noise_scales}")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Device: {device}")
    print(f"Max new tokens: {max_new_tokens}")

    # Setup
    tokenizer, base_model, latent_token_id, start_latent_id, end_latent_id, eos_token_id = setup_model_and_tokenizer(model_name)

    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )

    # Load questions
    questions = load_gsm8k_questions(num_questions)

    # Run experiments
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "noise_scales": noise_scales,
            "noise_type": noise_type,
            "noise_direction": noise_direction,
            "num_branches": num_branches,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "model": model_name,
            "latent_passes": 8,
            "method": "branching_with_majority_vote"
        },
        "questions": []
    }

    for q_idx, q_data in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {q_idx + 1}/{len(questions)}")
        print(f"{'='*70}")
        print(f"Q: {q_data['question'][:100]}...")
        print(f"Expected Answer: {q_data['answer']}")

        question_results = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "expected_answer": q_data["answer"],
            "noise_tests": []
        }

        for noise_scale in noise_scales:
            print(f"\n  Testing noise={noise_scale} with {num_branches} branches...", end=" ")

            result = test_question_with_branching(
                coconut_model=coconut_model,
                tokenizer=tokenizer,
                question=q_data["question"],
                noise_scale=noise_scale,
                start_latent_id=start_latent_id,
                end_latent_id=end_latent_id,
                num_branches=num_branches,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                device=device,
                noise_type=noise_type,
                noise_direction=noise_direction
            )

            if result["success"]:
                expected_answer = extract_numerical_answer(q_data["answer"])
                is_correct = result["majority_answer"] == expected_answer

                print(f"✓")
                print(f"    Branch answers: {result['branch_answers']}")
                print(f"    Vote distribution: {result['vote_distribution']}")
                print(f"    Majority answer: {result['majority_answer']} | Expected: {expected_answer} | {'✓' if is_correct else '✗'}")

                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "branch_texts": result["branch_texts"],
                    "branch_answers": result["branch_answers"],
                    "majority_answer": result["majority_answer"],
                    "vote_distribution": result["vote_distribution"],
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "num_branches": result["num_branches"],
                    "success": True,
                    "error": None
                })
            else:
                print(f"✗ Error: {result['error']}")

                question_results["noise_tests"].append({
                    "noise_scale": noise_scale,
                    "branch_texts": [],
                    "branch_answers": [],
                    "majority_answer": None,
                    "vote_distribution": {},
                    "expected_answer": extract_numerical_answer(q_data["answer"]),
                    "is_correct": False,
                    "num_branches": num_branches,
                    "success": False,
                    "error": result["error"]
                })

        results["questions"].append(question_results)

    return results


def analyze_results(results: Dict[str, Any]) -> None:
    """Print analysis of the results."""
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)

    noise_scales = results["experiment_info"]["noise_scales"]
    is_branching = results["experiment_info"].get("method") == "branching_with_majority_vote"

    print("\nAccuracy by Noise Scale:")
    print("-" * 70)

    # Check if we have baseline results (new format)
    has_baselines = False
    if results["questions"] and results["questions"][0]["noise_tests"]:
        first_test = results["questions"][0]["noise_tests"][0]
        has_baselines = "baseline1_correct" in first_test

    if has_baselines:
        print(f"{'Noise':<8} {'Baseline1':<12} {'Baseline2':<12} {'Coconut':<12} {'Status'}")
    elif is_branching:
        print(f"{'Noise':<8} {'Correct':<8} {'Total':<8} {'Accuracy':<10} {'Status'} (Using Majority Vote)")
    else:
        print(f"{'Noise':<8} {'Correct':<8} {'Total':<8} {'Accuracy':<10} {'Status'}")
    print("-" * 70)

    for noise_scale in noise_scales:
        if has_baselines:
            # New format with baselines
            baseline1_correct = 0
            baseline2_correct = 0
            coconut_correct = 0
            total = 0

            for question in results["questions"]:
                for test in question["noise_tests"]:
                    if test["noise_scale"] == noise_scale:
                        total += 1
                        if test.get("baseline1_correct"):
                            baseline1_correct += 1
                        if test.get("baseline2_correct"):
                            baseline2_correct += 1
                        if test.get("coconut_correct"):
                            coconut_correct += 1

            baseline1_acc = (baseline1_correct / total * 100) if total > 0 else 0
            baseline2_acc = (baseline2_correct / total * 100) if total > 0 else 0
            coconut_acc = (coconut_correct / total * 100) if total > 0 else 0

            if coconut_acc >= 60:
                status = "✓ Good"
            elif coconut_acc >= 20:
                status = "⚠ Degraded"
            else:
                status = "✗ Broken"

            print(f"{noise_scale:<8.2f} {baseline1_acc:>5.1f}% ({baseline1_correct}/{total})  {baseline2_acc:>5.1f}% ({baseline2_correct}/{total})  {coconut_acc:>5.1f}% ({coconut_correct}/{total})  {status}")

        else:
            # Old format
            correct = 0
            total = 0

            for question in results["questions"]:
                for test in question["noise_tests"]:
                    if test["noise_scale"] == noise_scale and test["success"]:
                        total += 1
                        if test["is_correct"]:
                            correct += 1

            accuracy = (correct / total * 100) if total > 0 else 0

            if accuracy >= 60:
                status = "✓ Good"
            elif accuracy >= 20:
                status = "⚠ Degraded"
            else:
                status = "✗ Broken"

            print(f"{noise_scale:<8.2f} {correct:<8} {total:<8} {accuracy:>6.1f}%     {status}")

    # Show example outputs
    print("\n" + "="*70)
    print("SAMPLE OUTPUTS (First Question)")
    print("="*70)

    if results["questions"]:
        first_q = results["questions"][0]
        print(f"\nQuestion: {first_q['question'][:120]}...")
        print(f"Expected: {first_q['expected_answer']}")
        print("\n" + "-" * 70)

        for test in first_q["noise_tests"]:
            noise = test["noise_scale"]

            if has_baselines:
                # New format with baselines
                b1_correct = "✓" if test.get("baseline1_correct") else "✗"
                b2_correct = "✓" if test.get("baseline2_correct") else "✗"
                c_correct = "✓" if test.get("coconut_correct") else "✗"

                b1_ans = test.get("baseline1_answer", "NO_ANSWER")
                b2_ans = test.get("baseline2_answer", "NO_ANSWER")
                c_ans = test.get("coconut_answer", "NO_ANSWER")

                print(f"\nNoise={noise:>5.2f}:")
                print(f"  Baseline 1: {b1_correct} [{b1_ans}]")
                print(f"  Baseline 2: {b2_correct} [{b2_ans}]")
                print(f"  Coconut:    {c_correct} [{c_ans}]")

            elif is_branching:
                # Branching mode - show majority vote info
                correct = "✓" if test["is_correct"] else "✗"
                answer = test.get("majority_answer", "NO_ANSWER")
                vote_dist = test.get("vote_distribution", {})
                print(f"\nNoise={noise:>5.2f}: {correct} [Majority: {answer}]")
                print(f"  Vote distribution: {vote_dist}")
                if test.get("branch_answers"):
                    print(f"  Individual branches: {test['branch_answers']}")
            else:
                # Single-path mode - show generated answer
                correct = "✓" if test["is_correct"] else "✗"
                answer = test.get("generated_answer", "NO_ANSWER")
                output = test.get("generated_text", "")[:100]
                print(f"\nNoise={noise:>5.2f}: {correct} [{answer}]")
                print(f"  {output}...")


def save_results(results: Dict[str, Any], filename: str = "coconut_llama_noise_results.json") -> None:
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def main():
    """Main experiment runner."""

    # ============================================================================
    # CONFIGURATION - Modify these to test different noise types
    # ============================================================================

    # EXPERIMENT MODE - Choose one:
    # -----------------------------------------------------------------------
    # BRANCHING + MAJORITY VOTING:
    #   - Applies noise after first latent pass, then branches into N paths
    #   - Each branch completes remaining 7 latent passes independently
    #   - Generates N diverse answers and takes majority vote
    #   - Good for testing noise robustness with voting redundancy
    #
    # SINGLE-PATH GENERATION:
    #   - Standard single-path generation with optional noise
    #   - Can apply noise to first pass only or all 8 passes
    #   - Good for baseline testing and understanding direct noise effects
    # -----------------------------------------------------------------------
    USE_BRANCHING = False  # Set to True to use branching + majority voting
                          # Set to False to use standard single-path generation

    NUM_QUESTIONS = 10  # Larger sample for clearer trends
    MAX_NEW_TOKENS = 1024
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

    # Branching configuration (only used if USE_BRANCHING = True)
    NUM_BRANCHES = 5  # Number of branches to generate per question
    TEMPERATURE = 0.7  # Sampling temperature for diversity
    TOP_P = 0.9  # Nucleus sampling parameter

    # Noise configuration - Choose one:

    # Option 1: Standard Gaussian noise (baseline)
    # NOISE_TYPE = "gaussian"
    # NOISE_DIRECTION = None
    # NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    # OUTPUT_FILE = "results_gaussian.json"

    # Option 2: Gaussian noise scaled to hidden state norm (FINER SCALE)
    # NOISE_TYPE = "gaussian_scaled"
    # NOISE_DIRECTION = None
    # # Finer noise scale focused on transition region (2.0-5.0)
    # NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
    # OUTPUT_FILE = "results_gaussian_scaled.json"

    # Option 3: Orthogonal noise (perpendicular to hidden state direction)
    # Tests if the model is sensitive to directional corruption while preserving magnitude
    # NOISE_TYPE = "orthogonal"
    # NOISE_DIRECTION = None
    # Gradually increase noise from 0 (no corruption) to 10 (high directional corruption)
    # NOISE_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    # OUTPUT_FILE = "results_orthogonal.json"

    # Option 4: Targeted noise - same direction (amplify)
    # NOISE_TYPE = "targeted"
    # NOISE_DIRECTION = "same"
    # NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    # OUTPUT_FILE = "results_targeted_amplify.json"

    # Option 5: Targeted noise - opposite direction (dampen) - ACTIVE
    # This directly reduces signal strength and should show clear degradation
    NOISE_TYPE = "targeted"
    NOISE_DIRECTION = "opposite"
    # Scale from 0.0 (no change) to 1.0 (complete signal elimination)
    NOISE_SCALES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    OUTPUT_FILE = "results_targeted_opposite_all_passes_50q.json"

    # Noise application strategy
    APPLY_NOISE_TO_ALL_PASSES = True  # Set to True for compounding degradation across all 8 passes
                                       # Set to False to only apply noise to first pass (default)

    # Option 6: Uniform noise
    # NOISE_TYPE = "uniform"
    # NOISE_DIRECTION = None
    # NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    # OUTPUT_FILE = "results_uniform.json"

    # ============================================================================

    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Mode: {'BRANCHING + MAJORITY VOTING' if USE_BRANCHING else 'SINGLE-PATH GENERATION'}")
    if USE_BRANCHING:
        print(f"Branches: {NUM_BRANCHES}")
        print(f"Temperature: {TEMPERATURE}")
        print(f"Top-p: {TOP_P}")
    else:
        print(f"Apply noise to all passes: {APPLY_NOISE_TO_ALL_PASSES}")
    print(f"\nNoise Configuration:")
    print(f"  Type: {NOISE_TYPE}")
    print(f"  Direction: {NOISE_DIRECTION if NOISE_DIRECTION else 'N/A'}")
    print(f"  Scales: {NOISE_SCALES}")
    print("="*70 + "\n")

    # Run experiment based on mode
    if USE_BRANCHING:
        results = run_branching_experiment(
            num_questions=NUM_QUESTIONS,
            noise_scales=NOISE_SCALES,
            num_branches=NUM_BRANCHES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            model_name=MODEL_NAME,
            noise_type=NOISE_TYPE,
            noise_direction=NOISE_DIRECTION
        )
    else:
        results = run_noise_experiment(
            num_questions=NUM_QUESTIONS,
            noise_scales=NOISE_SCALES,
            max_new_tokens=MAX_NEW_TOKENS,
            model_name=MODEL_NAME,
            noise_type=NOISE_TYPE,
            noise_direction=NOISE_DIRECTION,
            apply_noise_to_all_passes=APPLY_NOISE_TO_ALL_PASSES
        )

    # Analyze
    analyze_results(results)

    # Save
    save_results(results, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()