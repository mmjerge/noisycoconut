# CLAUDE.md - AI Assistant Guide for Coconut

**Last Updated:** 2025-12-05
**Repository:** Coconut - Training Large Language Models to Reason in a Continuous Latent Space

## Project Overview

This repository implements **Coconut** (Continuous Chain-of-Thought), a novel approach to training LLMs that reasons in a continuous latent space rather than through explicit textual chain-of-thought. The project is based on the paper: [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

**Key Concept:** Instead of generating intermediate reasoning steps as text tokens, Coconut uses special latent tokens (`<|latent|>`, `<|start-latent|>`, `<|end-latent|>`) to represent continuous thought vectors in the hidden space.

## Repository Structure

```
coconut/
├── coconut.py              # Core Coconut model implementation (GPT2-based)
├── run.py                  # Main training script with distributed training
├── dataset.py              # Dataset loading and processing utilities
├── test.py                 # Noise robustness testing script
├── run_noisy_experiment.py # Comprehensive noise experiments on GSM8K
├── quick_test.py           # Quick model testing script
├── quick_branch_test.py    # Branch-specific testing
├── full_test.py            # Full evaluation suite
├── utils.py                # Utility functions (Config class, set_seed)
├── args/                   # YAML configuration files
│   ├── gsm_cot.yaml        # GSM8K Chain-of-Thought baseline
│   ├── gsm_coconut.yaml    # GSM8K Coconut training
│   ├── gsm_coconut_eval.yaml
│   ├── prontoqa_coconut.yaml
│   ├── prontoqa_coconut_eval.yaml
│   ├── prosqa_coconut.yaml
│   └── prosqa_coconut_eval.yaml
├── preprocessing/          # Data preprocessing scripts
│   ├── gsm_icot.bash       # GSM8K dataset preprocessing
│   ├── gsm_icot.py         # GSM8K data processing
│   └── prontoqa.py         # ProntoQA data processing
├── scripts/                # Execution scripts
│   ├── run_experiment.sh   # SLURM job submission script
│   ├── run_simple_experiment.sh
│   └── run_branch_experiment.sh
├── data/                   # Training/eval datasets (JSON format)
│   ├── prosqa_train.json
│   ├── prosqa_valid.json
│   └── prosqa_test.json
├── results/                # Experimental results
│   ├── noise_scale/
│   └── noisy_cconut/
├── ablation/               # Ablation study results
│   └── results/
├── logs/                   # Training and experiment logs
├── assets/                 # Project assets (images, etc.)
├── NOISE_TYPES_GUIDE.md    # Comprehensive guide to noise testing
└── requirements.txt        # Python dependencies
```

## Core Components

### 1. Coconut Model (`coconut.py`)

**Key Classes:**
- `Coconut`: Main model class extending GPT2LMHeadModel
- `Outputs`: Named tuple for model outputs (loss, inputs_embeds, logits)

**Special Tokens:**
- `<|latent|>`: Single latent thought token (used during training stages)
- `<|start-latent|>`: Marks beginning of continuous reasoning space
- `<|end-latent|>`: Marks end of continuous reasoning space

**Key Features:**
- Supports continuous thought insertion (controlled by `c_thought` parameter)
- Noise injection capabilities for robustness testing
- Multiple noise types: gaussian, gaussian_scaled, snr, uniform, orthogonal, targeted

**Constants:**
- `MAX_N_LATENT = 8`: Maximum number of latent tokens per reasoning step

### 2. Training Pipeline (`run.py`)

**Distributed Training Setup:**
- Uses PyTorch Distributed (NCCL backend)
- Supports both DDP and FSDP
- Multi-GPU training with gradient accumulation

**Training Stages:**
- Stage 0: Initial CoT training (baseline)
- Stages 1-N: Progressive latent space training (controlled by `max_latent_stage`)
- Each stage uses `epochs_per_stage` epochs
- Optimizer can be reset between stages (`reset_optimizer` flag)

**Key Functions:**
- `main()`: Entry point, handles distributed setup and training loop
- Checkpoint management with automatic resume from preemption
- WandB integration for logging

### 3. Dataset System (`dataset.py`)

**Data Format (JSON):**
```json
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  }
]
```

**Key Functions:**
- `get_dataset()`: Load and tokenize standard datasets
- `get_question_latent_dataset()`: Question + latent tokens
- `get_cot_latent_dataset()`: CoT with latent thoughts
- `MyCollator`: Custom data collator for batching

**Processing:**
- Tokenizes questions, steps, and answers separately
- Distributed processing (only rank 0 processes, then broadcasts)
- Verification step ensures correct tokenization

## Configuration System

All experiments are configured via YAML files in `args/` directory.

### Key Configuration Parameters

**General Settings:**
- `project`: WandB project name
- `save_path`: Checkpoint directory
- `name`: Experiment name
- `only_eval`: If True, only evaluate (no training)
- `debug`: If True, no WandB logging, uses subset of data

**Method Selection (set one to True):**
- `coconut`: Train Coconut model
- `cot`: Train CoT baseline
- `no_thoughts`: Train Coconut without thought tokens
- `no_cot`: Train without CoT

**Training Parameters:**
- `c_thought`: Number of continuous thoughts per reasoning step (typically 2)
- `epochs_per_stage`: Epochs per training stage (typically 3)
- `max_latent_stage`: Maximum training stages (typically 3, plus stage 0)
- `pad_latent_to_max`: Pad to max latent tokens if steps < stage
- `uniform_prob`: Probability to mix data from other stages (0.0 for standard, 0.3 for ablation)

**Model Settings:**
- `model_id`: HuggingFace model ID (e.g., "openai-community/gpt2")
- `load_model_path`: Path to checkpoint for initialization or evaluation
- `resume`: Epoch to resume from (skips initial stages)

**Optimization:**
- `batch_size_training`: Per-GPU batch size (typically 32)
- `gradient_accumulation_steps`: Gradient accumulation (typically 1)
- `lr`: Learning rate (typically 1e-4)
- `weight_decay`: Weight decay (typically 0.01)
- `bf16`: Use bfloat16 training

**Data:**
- `train_path`: Path to training JSON
- `val_path`: Path to validation/test JSON
- `seed`: Random seed (default: 0)

**Checkpointing:**
- `save_only_improve`: Save only when validation accuracy improves
- `reset_optimizer`: Reset optimizer when switching stages

## Development Workflows

### 1. Setting Up Environment

```bash
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
wandb login  # Required for logging
```

### 2. Data Preprocessing

**GSM8K:**
```bash
bash preprocessing/gsm_icot.bash
```

**ProntoQA:**
```bash
cd prontoqa  # Clone official repo first
python run_experiment.py --model-name json --model-size dummy \
  --ordering random --num-trials 10000 --few-shot-examples 0 \
  --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
# Copy 5hop_0shot_random.json to data/
python preprocessing/prontoqa.py
```

### 3. Training Workflow

**Standard Training (GSM8K example):**

1. **Stage 0 - CoT Training:**
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```
- Trains baseline CoT model
- Expected validation accuracy: ~40%
- Select checkpoint for Coconut initialization

2. **Coconut Training:**
```bash
# Update load_model_path in args/gsm_coconut.yaml with CoT checkpoint
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```
- Progressive latent space training
- Runs for `max_latent_stage` stages beyond initial stage
- Each stage: `epochs_per_stage` epochs

3. **Evaluation:**
```bash
# Update load_model_path in args/gsm_coconut_eval.yaml
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### 4. Testing and Experiments

**Quick Testing:**
```bash
python quick_test.py
```

**Noise Robustness Testing:**
```bash
python test.py  # Configure noise type/scales in main()
python run_noisy_experiment.py  # Full GSM8K noise analysis
```

**Supported Noise Types (see NOISE_TYPES_GUIDE.md):**
- `gaussian`: Standard Gaussian N(0, scale²)
- `gaussian_scaled`: Scaled relative to hidden state norm
- `snr`: Signal-to-Noise Ratio controlled
- `uniform`: Uniform random noise
- `orthogonal`: Perpendicular to hidden state
- `targeted`: Along/opposite to hidden state direction

## Key Conventions

### Code Style
- **Copyright headers**: All files have Meta Platforms copyright
- **License**: MIT License (see LICENSE file)
- **Imports**: Standard library → Third-party → Local imports
- **Type hints**: Used in newer test files (test.py, run_noisy_experiment.py)

### Naming Conventions
- **Files**: Snake_case (e.g., `coconut.py`, `run_noisy_experiment.py`)
- **Classes**: PascalCase (e.g., `Coconut`, `MyCollator`)
- **Functions**: Snake_case (e.g., `get_dataset`, `apply_noise_to_hidden_states`)
- **Config files**: Lowercase with underscores (e.g., `gsm_coconut.yaml`)

### Git Workflow
- **Main branch**: `main` (not master)
- **Recent activity**: Focused on results updates and ablation studies
- **Commit style**: Brief imperative messages (e.g., "Updated results", "added ablation results")

### Documentation
- **Docstrings**: Used for complex functions, especially in test files
- **Comments**: Used to explain non-obvious logic and experimental configurations
- **Guides**: Separate markdown files for complex topics (NOISE_TYPES_GUIDE.md)

## Important Notes for AI Assistants

### 1. Understanding Coconut Architecture
- Coconut is NOT a standard transformer - it modifies GPT2 to insert continuous thought tokens
- The model operates in stages, progressively adding more latent reasoning capacity
- Training requires careful checkpoint management between stages

### 2. Configuration Management
- NEVER hardcode paths - always use config YAML files
- Key paths to update: `save_path`, `load_model_path`, `train_path`, `val_path`
- Remember to update `resume` parameter when skipping initial training stages

### 3. Distributed Training
- Default setup assumes 4 x A100 80GB GPUs
- Adjust `batch_size_training`, `gradient_accumulation_steps`, and `nproc_per_node` for different setups
- Training uses `torchrun` NOT `python` directly

### 4. Checkpointing
- Checkpoints saved as `checkpoint_<epoch>/`
- Model automatically resumes from latest checkpoint if run is preempted
- Stage 0 checkpoint needed to initialize Coconut training
- Set `save_only_improve=False` for Coconut to ensure final stage checkpoints are saved

### 5. Datasets
- GSM8K: Mathematical reasoning (requires preprocessing)
- ProntoQA: Logical reasoning (requires external repo)
- ProsQA: Logical reasoning (included in data/)
- All datasets must follow the JSON format with question/answer/steps

### 6. Noise Experiments
- Noise is applied to **last hidden layer output of first latent pass**
- Multiple noise types available - see NOISE_TYPES_GUIDE.md
- Use `test.py` for quick tests, `run_noisy_experiment.py` for comprehensive analysis
- Results saved to JSON files with timestamps

### 7. WandB Integration
- MUST run `wandb login` before training
- Set `debug=True` in config to disable WandB
- Project name controlled by `project` parameter in config

### 8. Common Pitfalls
- Forgetting to update `load_model_path` between training stages
- Using wrong number of GPUs (affects effective batch size)
- Not preprocessing datasets before training
- Missing special tokens when using new base models
- Forgetting to set `only_eval=True` for evaluation runs

### 9. Hardware Requirements
- Default configs assume 4 x A100 80GB GPUs
- Minimum: 1 GPU with 16GB VRAM (reduce batch size)
- Training a full Coconut model can take 24-72 hours depending on dataset/hardware

### 10. File Modification Guidelines
- **Core model (coconut.py)**: Rarely modified - stable implementation
- **Training script (run.py)**: May need updates for new features/optimizations
- **Test scripts**: Frequently modified for new experiments
- **Config files**: Create new configs rather than modifying existing ones
- **Preprocessing**: Dataset-specific, modify carefully

## Common Tasks

### Adding a New Dataset
1. Create preprocessing script in `preprocessing/`
2. Ensure output format matches: `[{"question": ..., "answer": ..., "steps": [...]}]`
3. Save to `data/` directory
4. Create corresponding YAML configs in `args/`

### Creating a New Experiment
1. Copy relevant YAML config from `args/`
2. Update: `name`, `save_path`, `train_path`, `val_path`
3. Adjust hyperparameters as needed
4. Run with `torchrun`

### Debugging Training Issues
1. Enable debug mode: `debug: True` in config
2. Check dataset loading: verify JSON format and tokenization
3. Monitor WandB logs for NaN losses or gradient issues
4. Verify checkpoint paths are correct
5. Ensure adequate GPU memory (reduce batch size if needed)

### Running Ablation Studies
1. Set `uniform_prob: 0.3` for mixed stage training
2. Modify `c_thought` to test different latent capacities
3. Use `no_thoughts` or `no_cot` modes for baselines
4. Save results to `ablation/results/`

## Testing Checklist

Before committing changes:
- [ ] Run `quick_test.py` to verify basic functionality
- [ ] Check that config files have valid paths
- [ ] Verify dataset format matches expected structure
- [ ] Test with debug mode first
- [ ] Ensure special tokens are properly added to tokenizer
- [ ] Confirm checkpoints can be loaded correctly
- [ ] Validate distributed training works with your GPU setup

## Resources

- **Paper**: https://arxiv.org/abs/2412.06769
- **License**: MIT License (see LICENSE file)
- **Contributing**: See CONTRIBUTING.md
- **Code of Conduct**: See CODE_OF_CONDUCT.md
- **Noise Testing**: See NOISE_TYPES_GUIDE.md

## Dependencies

Core requirements (see requirements.txt):
- torch==2.5.1
- transformers==4.46.2
- wandb==0.18.7
- datasets==3.1.0
- numpy==2.1.3
- tqdm==4.67.0

## Recent Development Focus

Based on recent commits:
- Adding and analyzing experimental results
- Ablation studies on model components
- Noise robustness testing and analysis
- Support for GPT-based models (gpt-oss)

## Contact and Support

For bugs or questions:
1. Check existing issues in the repository
2. Review documentation in README.md and this file
3. Consult the paper for theoretical background
4. For security issues, use Meta's bug bounty program
