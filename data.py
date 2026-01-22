# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm import tqdm

def download_benchmarks(
    data_dir: str = "./data",
    benchmarks: Optional[List[str]] = None,
    force_redownload: bool = False
) -> dict:
    """
    Download GSM8K, GSM-Symbolic, and MMLU benchmarks to a data directory.
    
    Args:
        data_dir: Directory to save the downloaded datasets
        benchmarks: List of benchmarks to download. Options: "gsm8k", "gsm-symbolic", "mmlu"
                   If None, downloads all benchmarks.
        force_redownload: If True, redownload even if files exist
    
    Returns:
        Dictionary mapping benchmark names to their save paths
    """
    if benchmarks is None:
        benchmarks = ["gsm8k", "gsm-symbolic", "mmlu"]
    
    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Downloading {benchmark.upper()}")
        print(f"{'='*60}")
        
        benchmark_dir = data_path / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        if benchmark == "gsm8k":
            saved_paths[benchmark] = _download_gsm8k(benchmark_dir, force_redownload)
        elif benchmark == "gsm-symbolic":
            saved_paths[benchmark] = _download_gsm_symbolic(benchmark_dir, force_redownload)
        elif benchmark == "mmlu":
            saved_paths[benchmark] = _download_mmlu(benchmark_dir, force_redownload)
        else:
            print(f"Unknown benchmark: {benchmark}. Skipping.")
            continue
    
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for name, paths in saved_paths.items():
        print(f"\n{name.upper()}:")
        for split, path in paths.items():
            print(f"  {split}: {path}")
    
    return saved_paths


def _download_gsm8k(benchmark_dir: Path, force_redownload: bool) -> dict:
    """Download GSM8K dataset."""
    paths = {}
    
    for split in ["train", "test"]:
        output_file = benchmark_dir / f"{split}.json"
        
        if output_file.exists() and not force_redownload:
            print(f"  {split} already exists: {output_file}")
            paths[split] = str(output_file)
            continue
        
        print(f"  Downloading {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        # Convert to list of dicts with consistent format
        data = []
        for idx, item in enumerate(tqdm(dataset, desc=f"  Processing {split}")):
            # Parse the answer to extract steps and final answer
            answer_text = item["answer"]
            
            # GSM8K format: steps separated by newlines, final answer after ####
            parts = answer_text.split("####")
            if len(parts) == 2:
                steps_text = parts[0].strip()
                final_answer = parts[1].strip()
                steps = [s.strip() for s in steps_text.split("\n") if s.strip()]
            else:
                steps = [answer_text]
                final_answer = answer_text
            
            data.append({
                "question": item["question"],
                "steps": steps,
                "answer": final_answer,
                "full_answer": answer_text,
                "idx": idx
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved {len(data)} examples to {output_file}")
        paths[split] = str(output_file)
    
    return paths


def _download_gsm_symbolic(benchmark_dir: Path, force_redownload: bool) -> dict:
    """Download GSM-Symbolic dataset."""
    paths = {}
    
    output_file = benchmark_dir / "test.json"
    
    if output_file.exists() and not force_redownload:
        print(f"  test already exists: {output_file}")
        paths["test"] = str(output_file)
        return paths
    
    print("  Downloading test split...")
    dataset = load_dataset("apple/gsm-symbolic", split="test", trust_remote_code=True)
    
    data = []
    for idx, item in enumerate(tqdm(dataset, desc="  Processing test")):
        answer_text = item["answer"]
        
        # Parse similar to GSM8K
        parts = answer_text.split("####")
        if len(parts) == 2:
            steps_text = parts[0].strip()
            final_answer = parts[1].strip()
            steps = [s.strip() for s in steps_text.split("\n") if s.strip()]
        else:
            steps = [answer_text]
            final_answer = answer_text
        
        data.append({
            "question": item["question"],
            "steps": steps,
            "answer": final_answer,
            "full_answer": answer_text,
            "idx": idx
        })
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved {len(data)} examples to {output_file}")
    paths["test"] = str(output_file)
    
    return paths


def _download_mmlu(benchmark_dir: Path, force_redownload: bool) -> dict:
    """Download MMLU dataset."""
    paths = {}
    answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    for split in ["test", "validation", "dev"]:
        output_file = benchmark_dir / f"{split}.json"
        
        if output_file.exists() and not force_redownload:
            print(f"  {split} already exists: {output_file}")
            paths[split] = str(output_file)
            continue
        
        print(f"  Downloading {split} split...")
        try:
            dataset = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: Could not download {split} split: {e}")
            continue
        
        data = []
        for idx, item in enumerate(tqdm(dataset, desc=f"  Processing {split}")):
            # Format choices
            choices = item["choices"]
            choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
            formatted_question = f"{item['question']}\n\n{choices_text}"
            
            correct_answer = answer_mapping[item["answer"]]
            
            data.append({
                "question": formatted_question,
                "raw_question": item["question"],
                "choices": choices,
                "steps": [f"Analyzing the question about {item.get('subject', 'unknown')}"],
                "answer": correct_answer,
                "correct_choice_idx": item["answer"],
                "subject": item.get("subject", "unknown"),
                "idx": idx
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved {len(data)} examples to {output_file}")
        paths[split] = str(output_file)
    
    # Also save subjects list
    subjects_file = benchmark_dir / "subjects.json"
    if not subjects_file.exists() or force_redownload:
        try:
            dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
            subjects = list(set(item.get("subject", "unknown") for item in dataset))
            subjects.sort()
            with open(subjects_file, 'w') as f:
                json.dump(subjects, f, indent=2)
            print(f"  Saved {len(subjects)} subjects to {subjects_file}")
        except Exception:
            pass
    
    return paths


def get_benchmark_stats(data_dir: str = "./data") -> dict:
    """Get statistics about downloaded benchmarks."""
    data_path = Path(data_dir).expanduser()
    stats = {}
    
    for benchmark in ["gsm8k", "gsm-symbolic", "mmlu"]:
        benchmark_dir = data_path / benchmark
        if not benchmark_dir.exists():
            stats[benchmark] = {"status": "not downloaded"}
            continue
        
        benchmark_stats = {"status": "downloaded", "splits": {}}
        for json_file in benchmark_dir.glob("*.json"):
            if json_file.name == "subjects.json":
                continue
            split_name = json_file.stem
            with open(json_file, 'r') as f:
                data = json.load(f)
            benchmark_stats["splits"][split_name] = len(data)
        
        stats[benchmark] = benchmark_stats
    
    return stats


# ============================================================================
# Original Dataset Functions (unchanged)
# ============================================================================

def get_dataset(path, tokenizer, max_size=1000000000):

    def tokenize_sample(sample):

        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    # verify
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    assert (
        complete_tokenized
        == dataset[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
        + dataset[0]["answer_tokenized"]
    )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):

    def process_dataset(sample):

        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)

        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):

        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets for Coconut experiments"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded datasets (default: ./data)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["gsm8k", "gsm-symbolic", "mmlu", "all"],
        default=["all"],
        help="Benchmarks to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if files exist"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about downloaded datasets"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        print("\nBenchmark Statistics")
        print("=" * 60)
        stats = get_benchmark_stats(args.data_dir)
        for benchmark, info in stats.items():
            print(f"\n{benchmark.upper()}:")
            if info["status"] == "not downloaded":
                print("  Not downloaded")
            else:
                for split, count in info["splits"].items():
                    print(f"  {split}: {count} examples")
    else:
        benchmarks = None if "all" in args.benchmarks else args.benchmarks
        download_benchmarks(
            data_dir=args.data_dir,
            benchmarks=benchmarks,
            force_redownload=args.force
        )