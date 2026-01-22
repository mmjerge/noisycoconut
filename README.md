# NoisyCoconut

![NoisyCoconut Logo](./assets/noisy_coconut_diagram.png)

A memory-optimized implementation of the COCONUT approach for generating divergent reasoning paths in large language models with DeepSpeed acceleration.

## Overview

Noisy Coconut is a Python package for testing and evaluating the COCONUT (COntinuous latent mental COmputation with Noisy UpdateTs) approach with large language models. It implements true latent thinking entirely in hidden state space, without tokenizing during the thinking process.

The package is optimized for efficient inference with:
1. DeepSpeed for multi-GPU inference
2. Flash Attention for faster attention computation
3. Batch processing for parallel evaluation

## Features

- **True Latent Thinking**: Implements COCONUT with thinking entirely in hidden state space
- **DeepSpeed Optimization**: Multi-GPU inference with tensor parallelism
- **Flexible Benchmarking**: Ready-to-use benchmarks for GSM8k, GSM-Symbolic, and MMLU
- **Advanced Voting Systems**: Multiple voting schemes for aggregating results
- **Comprehensive Evaluation**: Detailed results and statistics
- **Configuration System**: OmegaConf-based configuration with YAML support

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU(s)

### Install from Source

```bash
git clone https://github.com/yourusername/NoisyCoconut.git
cd NoisyCoconut
pip install -e .
```

### Install with DeepSpeed

```bash
pip install "noisy_coconut[deepspeed]"
```

## Quick Start

### Basic Usage

```python
from noisy_coconut import DeepSpeedLatentPathEvaluator

# Initialize the evaluator
evaluator = DeepSpeedLatentPathEvaluator(
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    local_rank=0,
    world_size=1,
    max_new_tokens=1024,
    noise_scale=0.1,
    max_latent_steps=4,
    voting_scheme="simple"
)

# Generate divergent reasoning paths for a math problem
problem = "John has 5 pens and Mary has 3 times as many pens as John. How many pens do they have in total?"
paths = evaluator.generate_divergent_paths(problem, num_paths=3, benchmark_type="gsm8k")

# Print the results
for i, path_data in enumerate(paths):
    print(f"Path {i+1}:")
    print(f"Text: {path_data['text'][:200]}...")
    print(f"Metadata: {path_data['metadata']}")
    print()
```

## Configuration System

NoisyCoconut uses OmegaConf for flexible configuration management through YAML files and command-line overrides.

### Default Configuration

The package comes with a default configuration file that sets common parameters:

```yaml
# Model settings
model:
  name: "Qwen/Qwen2-7B-Instruct"
  dtype: "fp16" 
  use_flash_attention: true

# DeepSpeed settings
deepspeed:
  world_size: 4
  local_rank: -1
  enable_cuda_graph: true

# Generation settings
generation:
  max_new_tokens: 2000
  batch_size: 8

# COCONUT settings
coconut:
  noise_scale: 0.1
  max_latent_steps: 8
  latent_noise_steps: "1,2,3,4"
  voting_scheme: "simple"

# Benchmark settings
benchmark:
  type: "gsm-symbolic"
  num_samples: 1000
  num_paths: 5
  seed: 42

# Output settings
output:
  file: "noisy_coconut_results.json"
  checkpoint_dir: "checkpoints"
  verbose: false
```

### Using Custom Configurations

You can create your own configuration files:

```yaml
# my_experiment.yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  dtype: "bf16"

benchmark:
  type: "gsm8k"
  num_samples: 100
```

And use them via the command line:

```bash
noisy-coconut --config my_experiment.yaml
```

### Command-Line Usage

#### Using Default Settings

```bash
# Use with all default settings
noisy-coconut

# Run with DeepSpeed
deepspeed --num_gpus=4 -m noisy_coconut.cli.main
```

#### Overriding Specific Settings

```bash
# Override specific parameters with new dot notation
noisy-coconut --model.name "meta-llama/Llama-3.1-8B-Instruct" --benchmark.type gsm8k
```

#### Using Multiple GPUs

```bash
# Run with multiple GPUs
deepspeed --num_gpus=4 -m noisy_coconut.cli.main --model.name "meta-llama/Llama-3.1-70B-Instruct" --deepspeed.world_size 4
```

#### Legacy Command Format (Backward Compatible)

The package still supports the old-style command format:

```bash
deepspeed --num_gpus=4 -m noisy_coconut.cli.main \
  --model_name "Qwen/Qwen2-7B-Instruct" \
  --world_size 4 \
  --num_paths 5 \
  --num_samples 1000 \
  --max_tokens 2000 \
  --noise_scale 0.1 \
  --use_flash_attention \
  --latent_noise_steps '1,2,3,4' \
  --max_latent_steps 8 \
  --batch_size 8 \
  --benchmark gsm-symbolic \
  --enable_cuda_graph \
  --checkpoint_dir "checkpoints"
```

## Advanced Usage

### Using the Configuration System in Code

```python
import os
from noisy_coconut import DeepSpeedLatentPathEvaluator
from noisy_coconut.utils import get_config, config_to_evaluator_args

# Load configuration - automatically handles file loading and CLI overrides
config = get_config()

# Convert config to evaluator arguments
evaluator_args = config_to_evaluator_args(config)

# Create evaluator with configuration
evaluator = DeepSpeedLatentPathEvaluator(**evaluator_args)

# Use the evaluator
problem = "What is 2+2?"
paths = evaluator.generate_divergent_paths(problem, 
                                          num_paths=config.benchmark.num_paths, 
                                          benchmark_type=config.benchmark.type)
```

### Multi-GPU Inference with DeepSpeed

```python
import os
import torch
import deepspeed
from noisy_coconut import DeepSpeedLatentPathEvaluator
from noisy_coconut.utils import get_config, config_to_evaluator_args

# Load configuration
config = get_config()

# Adjust for multi-GPU setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
config.deepspeed.local_rank = local_rank
config.deepspeed.world_size = world_size

# Get evaluator arguments from config
evaluator_args = config_to_evaluator_args(config)

# Create evaluator
evaluator = DeepSpeedLatentPathEvaluator(**evaluator_args)

# Load benchmark data
from noisy_coconut.utils import load_gsm8k_samples
questions, references = load_gsm8k_samples(n_samples=10, seed=42)

# Run benchmark evaluation
results = evaluator.evaluate_benchmark(
    questions, references, num_paths=5, benchmark_type="gsm8k"
)
```

### Comparing Different Voting Schemes

```python
from noisy_coconut.utils import compare_voting_schemes, get_config, config_to_evaluator_args

# Get base config
config = get_config()

# Run evaluations with different voting schemes
all_results = {}
for scheme in ["simple", "weighted", "accuracy", "advanced"]:
    # Update voting scheme in config
    config.coconut.voting_scheme = scheme
    
    # Get evaluator args and create evaluator
    evaluator_args = config_to_evaluator_args(config)
    evaluator = DeepSpeedLatentPathEvaluator(**evaluator_args)
    
    # Run benchmark with this scheme
    results = run_benchmarks(evaluator, config.benchmark.type, 
                            config.benchmark.num_samples, 
                            config.benchmark.num_paths, 
                            config.benchmark.seed)
    all_results[scheme] = results

# Compare results across voting schemes
comparison = compare_voting_schemes(all_results, ["simple", "weighted", "accuracy", "advanced"])
print(f"Best voting scheme: {comparison['overall']['best_scheme']}")
```

## Benchmarks

NoisyCoconut supports the following benchmarks:

- **GSM8K**: Grade school math problems
- **GSM-Symbolic**: Symbolic math problems
- **MMLU**: Massive Multitask Language Understanding

To run a specific benchmark:

```bash
noisy-coconut --benchmark.type gsm8k --benchmark.num_samples 20
```

## Technical Details

### COCONUT Approach

The COCONUT approach, as described in "Training Large Language Models to Reason in a Continuous Latent Space" (Hao et al., 2024), involves:

1. Operating entirely in hidden state space without tokenizing during thinking
2. Applying noise to promote exploration of diverse reasoning paths
3. Iterative updates to the hidden state until convergence
4. Seamless transition from latent thinking to token generation

This implementation enhances the original approach with optimizations for memory efficiency and multi-GPU inference.

### Voting Schemes

NoisyCoconut supports multiple voting schemes for aggregating results:

- **Simple**: Basic majority voting
- **Weighted**: Weights votes by inverse of noise scale
- **Accuracy**: Weights votes by historical accuracy of each path
- **Advanced**: Tries multiple strategies and picks the best one

## Citation

If you use NoisyCoconut in your research, please cite:

```bibtex
@software{noisycoconut2025,
  author = {Michael Jerge},
  title = {NoisyCoconut: A Memory-Optimized Implementation of COCONUT for Large Language Models},
  year = {2025},
  url = {https://github.com/yourusername/NoisyCoconut}
}
```

## License

MIT
