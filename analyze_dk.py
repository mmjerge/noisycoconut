"""
Analyze D_K path diversity by agreement level and correctness.
Produces a table suitable for the paper's Section 3.2 / 4 response.

Usage:
    python analyze_dk.py [results.json ...]              # print tables only
    python analyze_dk.py --export results.json           # also write distributions to
                                                         # results/dk_distributions.json
                                                         # (feed into plot_dk.py)
"""

import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path

def get_agreement_level(vote_distribution):
    return max(vote_distribution.values())

def bucket_data(data):
    """Return buckets: agreement_level -> {"correct": [d_k_mean, ...], "incorrect": [...],
                                           "correct_pairwise": [float, ...], "incorrect_pairwise": [...]}."""
    buckets = defaultdict(lambda: {
        "correct": [], "incorrect": [],
        "correct_pairwise": [], "incorrect_pairwise": [],
    })
    for item in data["results"]:
        for test in item["noise_tests"]:
            if "d_k" not in test:
                continue
            agreement = get_agreement_level(test["vote_distribution"])
            label = "correct" if test["is_correct"] else "incorrect"
            buckets[agreement][label].append(test["d_k"])
            # Also pool the individual pair distances (10 values for K=5)
            pairwise = test.get("d_k_pairwise", [])
            buckets[agreement][f"{label}_pairwise"].extend(pairwise)
    return buckets

def summarize(values):
    if not values:
        return None
    return {
        "n":      len(values),
        "mean":   statistics.mean(values),
        "median": statistics.median(values),
        "std":    statistics.stdev(values) if len(values) > 1 else 0.0,
        "q1":     statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
        "q3":     statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values),
        "min":    min(values),
        "max":    max(values),
        "values": values,   # raw list — used by plot_dk.py
    }

def print_tables(path, data, buckets):
    print(f"\nResults from: {path}")
    print(f"Noise scale: {data['config']['noise']['scales']}")
    print(f"Model: {data['config']['model']['name']}")
    print(f"Benchmark: {data['config']['benchmark']}")
    print()

    # Summary table by agreement level
    print("=" * 75)
    print(f"{'Agreement':>12} | {'N':>5} | {'Mean D_K':>10} | {'Median':>8} | {'Std':>7} | {'Correct %':>10}")
    print("-" * 75)
    for level in sorted(buckets.keys(), reverse=True):
        correct_dks = buckets[level]["correct"]
        incorrect_dks = buckets[level]["incorrect"]
        all_dks = correct_dks + incorrect_dks
        n = len(all_dks)
        s = summarize(all_dks)
        acc = len(correct_dks) / n * 100 if n > 0 else 0
        print(f"{level}/5{' ':>9} | {n:>5} | {s['mean']:>10.2f} | {s['median']:>8.2f} | {s['std']:>7.2f} | {acc:>9.1f}%")
    print("=" * 75)

    # Overall correct vs incorrect
    all_correct   = [dk for b in buckets.values() for dk in b["correct"]]
    all_incorrect = [dk for b in buckets.values() for dk in b["incorrect"]]
    sc = summarize(all_correct)
    si = summarize(all_incorrect)
    print()
    print(f"Overall D_K  correct:   mean={sc['mean']:.2f}  median={sc['median']:.2f}  std={sc['std']:.2f}  (n={sc['n']})")
    print(f"Overall D_K incorrect:  mean={si['mean']:.2f}  median={si['median']:.2f}  std={si['std']:.2f}  (n={si['n']})")

    # Detailed breakdown: agreement x correctness
    print()
    print("Detailed breakdown (D_K by agreement × correctness):")
    print("=" * 80)
    print(f"{'Agreement':>12} | {'Correct mean (med, std, n)':>28} | {'Incorrect mean (med, std, n)':>30}")
    print("-" * 80)
    for level in sorted(buckets.keys(), reverse=True):
        sc = summarize(buckets[level]["correct"])
        si = summarize(buckets[level]["incorrect"])
        c_str = f"{sc['mean']:.2f} ({sc['median']:.2f}, {sc['std']:.2f}, n={sc['n']})" if sc else "—"
        i_str = f"{si['mean']:.2f} ({si['median']:.2f}, {si['std']:.2f}, n={si['n']})" if si else "—"
        print(f"{level}/5{' ':>9} | {c_str:>28} | {i_str:>30}")
    print("=" * 80)

def export_distributions(path, data, buckets):
    """Write per-category raw D_K lists + summary stats to a JSON file for plotting."""
    out = {
        "source": str(path),
        "model":  data["config"]["model"]["name"],
        "benchmark": data["config"]["benchmark"],
        "noise_scales": data["config"]["noise"]["scales"],
        "by_agreement": {},
        "overall": {},
    }

    for level in sorted(buckets.keys(), reverse=True):
        key = f"{level}/5"
        out["by_agreement"][key] = {
            "correct":            summarize(buckets[level]["correct"]),
            "incorrect":          summarize(buckets[level]["incorrect"]),
            "correct_pairwise":   summarize(buckets[level]["correct_pairwise"]),
            "incorrect_pairwise": summarize(buckets[level]["incorrect_pairwise"]),
        }

    all_correct             = [dk for b in buckets.values() for dk in b["correct"]]
    all_incorrect           = [dk for b in buckets.values() for dk in b["incorrect"]]
    all_correct_pairwise    = [dk for b in buckets.values() for dk in b["correct_pairwise"]]
    all_incorrect_pairwise  = [dk for b in buckets.values() for dk in b["incorrect_pairwise"]]
    out["overall"]["correct"]            = summarize(all_correct)
    out["overall"]["incorrect"]          = summarize(all_incorrect)
    out["overall"]["correct_pairwise"]   = summarize(all_correct_pairwise)
    out["overall"]["incorrect_pairwise"] = summarize(all_incorrect_pairwise)

    out_path = Path("results") / "dk_distributions.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDistributions written to: {out_path}")
    print("Run:  python plot_dk.py  to generate figures.")

def load_and_analyze(path, export=False):
    with open(path) as f:
        data = json.load(f)
    buckets = bucket_data(data)
    print_tables(path, data, buckets)
    if export:
        export_distributions(path, data, buckets)

if __name__ == "__main__":
    args = sys.argv[1:]
    export = "--export" in args
    paths = [a for a in args if not a.startswith("--")] or [
        "results/noisy_coconut_gsm8k_Qwen_Qwen2.5-7B-Instruct_20260313_220539.json"
    ]
    for path in paths:
        load_and_analyze(path, export=export)
