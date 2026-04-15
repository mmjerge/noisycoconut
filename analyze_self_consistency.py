"""
Analyze self-consistency results and produce a majority-vote breakdown table.

Usage:
    python analyze_self_consistency.py <results.json>
"""

import json
import sys
import re
from collections import Counter
from pathlib import Path


def extract_answer(text: str) -> str:
    """Extract numeric answer from text (GSM8K)."""
    def clean(s):
        return s.replace(',', '').replace('$', '').strip()

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
            return clean(matches[-1].group(1))
    numbers = re.findall(r'[\d,]+', text)
    if numbers:
        cleaned = [clean(n) for n in numbers if clean(n)]
        if cleaned:
            return cleaned[-1]
    return "NO_ANSWER_FOUND"


def get_expected_answer(expected_raw: str) -> str:
    return extract_answer(expected_raw)


def classify_vote(vote_distribution: dict) -> str:
    """
    Classify agreement level:
      5/5  -> Unanimous
      4/5  -> Strong Majority
      3/5  -> Moderate Majority
      2/5  -> Minimal Majority (one answer with 2 votes, rest with 1)
      2/2  -> Split Vote (two answers tied with 2 votes each)
    """
    counts = sorted(vote_distribution.values(), reverse=True)
    top = counts[0]
    if top == 5:
        return "Unanimous (5/5)"
    elif top == 4:
        return "Strong Majority (4/5)"
    elif top == 3:
        return "Moderate Majority (3/5)"
    elif top == 2:
        # Check if there's a tie: two answers each with 2 votes
        if len(counts) >= 2 and counts[1] == 2:
            return "Split Vote (2/2)"
        else:
            return "Minimal Majority (2/5)"
    else:
        return "No Consensus (1/5)"


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)

    config = data["config"]
    results = data["results"]

    # -------------------------------------------------------------------------
    # Baseline: first sample answer, no majority voting
    # -------------------------------------------------------------------------
    baseline = {"correct": 0, "no_answer": 0, "incorrect": 0, "total": 0}
    for item in results:
        sc = item.get("self_consistency")
        if not sc or "sample_answers" not in sc:
            continue
        expected = get_expected_answer(item["expected_answer"])
        first_answer = sc["sample_answers"][0] if sc["sample_answers"] else "NO_ANSWER_FOUND"
        baseline["total"] += 1
        if first_answer == "NO_ANSWER_FOUND":
            baseline["no_answer"] += 1
        elif first_answer == expected:
            baseline["correct"] += 1
        else:
            baseline["incorrect"] += 1

    # -------------------------------------------------------------------------
    # Majority vote breakdown by agreement level
    # -------------------------------------------------------------------------
    categories = [
        "No Consensus (1/5)",
        "Minimal Majority (2/5)",
        "Split Vote (2/2)",
        "Moderate Majority (3/5)",
        "Strong Majority (4/5)",
        "Unanimous (5/5)",
    ]
    buckets = {cat: {"correct": 0, "no_answer": 0, "incorrect": 0, "total": 0}
               for cat in categories}

    for item in results:
        sc = item.get("self_consistency")
        if not sc or "vote_distribution" not in sc:
            continue
        expected = get_expected_answer(item["expected_answer"])
        majority = sc["majority_answer"]
        cat = classify_vote(sc["vote_distribution"])
        buckets[cat]["total"] += 1
        if majority == "NO_ANSWER_FOUND":
            buckets[cat]["no_answer"] += 1
        elif majority == expected:
            buckets[cat]["correct"] += 1
        else:
            buckets[cat]["incorrect"] += 1

    # -------------------------------------------------------------------------
    # Print table
    # -------------------------------------------------------------------------
    col_w = [28, 16, 9, 11, 11, 13]
    header = ["", "Question Count", "Correct", "No Answer", "Incorrect", "Accuracy (%)"]

    def row(cells):
        return "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))

    sep = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print(f"\nFile:  {path}")
    print(f"Model: {config['model']}")
    print(f"Benchmark: {config['benchmark']}  |  Questions: {config['num_questions']}  |  Samples: {config['num_branches']}")
    print()
    print(sep)
    print(row(header))
    print(sep)

    def acc(b):
        valid = b["correct"] + b["incorrect"]
        return f"{b['correct'] / valid * 100:.2f}" if valid > 0 else "—"

    b = baseline
    print(row(["Baseline (No majority)",
               b["total"], b["correct"], b["no_answer"], b["incorrect"], acc(b)]))
    print(sep)

    for cat in categories:
        b = buckets[cat]
        if b["total"] == 0:
            continue
        print(row([cat, b["total"], b["correct"], b["no_answer"], b["incorrect"], acc(b)]))


    print(sep)

    # Overall majority-vote accuracy
    total_c = sum(buckets[c]["correct"] for c in categories)
    total_i = sum(buckets[c]["incorrect"] for c in categories)
    total_na = sum(buckets[c]["no_answer"] for c in categories)
    total_all = total_c + total_i + total_na
    print(row(["Overall (majority vote)",
               total_all, total_c, total_na, total_i,
               f"{total_c / (total_c + total_i) * 100:.2f}" if (total_c + total_i) > 0 else "—"]))
    print(sep)
    print()


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "noisy_coconut_gsm8k_Qwen_Qwen2.5-7B-Instruct_20260318_145707.json"
    ]
    for p in paths:
        analyze(p)
