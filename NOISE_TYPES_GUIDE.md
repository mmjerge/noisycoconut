# COCONUT Noise Types Guide

This guide explains the different noise injection strategies implemented for testing COCONUT robustness.

## Overview

Noise is added to the **last hidden layer output of the first latent pass** in the continuous reasoning space. Different noise types test different hypotheses about what causes system breakdown.

## Noise Types

### 1. Gaussian (Standard)
**Type:** `"gaussian"`
**Direction:** `None`

Standard independent Gaussian noise: `N(0, scale^2)`

- **Scale interpretation:** Absolute noise magnitude
- **Recommended scales:** `[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]`
- **Use case:** Baseline comparison - tests raw magnitude sensitivity

```python
NOISE_TYPE = "gaussian"
NOISE_DIRECTION = None
NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
```

### 2. Gaussian Scaled
**Type:** `"gaussian_scaled"`
**Direction:** `None`

Gaussian noise scaled relative to hidden state norm.

- **Scale interpretation:** Fraction of hidden state norm (e.g., 0.5 = 50% of norm)
- **Recommended scales:** `[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]`
- **Use case:** Test if breakdown depends on relative perturbation vs absolute magnitude

**Formula:** `noise_norm = scale * ||hidden_state||`

```python
NOISE_TYPE = "gaussian_scaled"
NOISE_DIRECTION = None
NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
```

### 3. Signal-to-Noise Ratio (SNR)
**Type:** `"snr"`
**Direction:** `None`

Gaussian noise calibrated by Signal-to-Noise Ratio - provides **predictable, controlled degradation**.

- **Scale interpretation:** SNR value (Signal_Power / Noise_Power)
  - `SNR = 10.0`: Signal is 10x stronger than noise (minimal degradation)
  - `SNR = 1.0`: Signal equals noise (50/50 mix)
  - `SNR = 0.1`: Noise is 10x stronger than signal (severe degradation)
  - `SNR = 0.01`: Noise is 100x stronger (complete destruction)
- **Recommended scales:** `[100.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]` (high to low quality)
- **Use case:** Most interpretable - directly controls signal quality degradation

**Key advantage:** SNR is normalized and comparable across different models/layers/positions

**Formula:** `noise_norm = signal_norm / SNR`

```python
NOISE_TYPE = "snr"
NOISE_DIRECTION = None
# From high quality (SNR=100) to completely destroyed (SNR=0.01)
NOISE_SCALES = [100.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
OUTPUT_FILE = "results_snr.json"
```

**Expected behavior:**
- SNR ≥ 10: Should maintain good performance (signal dominates)
- SNR ~ 1-5: Gradual degradation (signal and noise comparable)
- SNR ≤ 0.1: Severe breakdown (noise dominates)

### 4. Uniform
**Type:** `"uniform"`
**Direction:** `None`

Uniform random noise: `U(-scale, scale)`

- **Scale interpretation:** Half-width of uniform distribution
- **Recommended scales:** `[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]`
- **Use case:** Test if distribution shape matters (vs Gaussian)

```python
NOISE_TYPE = "uniform"
NOISE_DIRECTION = None
NOISE_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
```

### 5. Orthogonal
**Type:** `"orthogonal"`
**Direction:** `None`

Noise perpendicular to the hidden state direction (uses Gram-Schmidt projection).

- **Scale interpretation:** Fraction of hidden state norm in orthogonal direction
- **Recommended scales:** `[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]`
- **Use case:** Test if direction matters - does perpendicular noise cause less damage?

**Key property:** Cosine similarity between noise and hidden state ≈ 0

```python
NOISE_TYPE = "orthogonal"
NOISE_DIRECTION = None
NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
```

### 6. Targeted (Amplify)
**Type:** `"targeted"`
**Direction:** `"same"`

Noise in the same direction as hidden states - amplifies the representation.

- **Scale interpretation:** Amplification factor (1.0 = double the magnitude)
- **Recommended scales:** `[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]`
- **Use case:** Does amplification help or hurt reasoning?

**Formula:** `new_state = hidden_state * (1 + scale)`

```python
NOISE_TYPE = "targeted"
NOISE_DIRECTION = "same"
NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
```

### 7. Targeted (Dampen)
**Type:** `"targeted"`
**Direction:** `"opposite"`

Noise opposite to hidden states - dampens/reduces the representation.

- **Scale interpretation:** Dampening factor (1.0 = zero out completely)
- **Recommended scales:** `[0.0, 0.1, 0.2, 0.5, 0.9, 1.0]`
- **Use case:** Test if reducing signal strength causes breakdown

**Formula:** `new_state = hidden_state * (1 - scale)`

```python
NOISE_TYPE = "targeted"
NOISE_DIRECTION = "opposite"
NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 0.9, 1.0]
```

## Experimental Hypotheses

### Hypothesis 1: Magnitude matters most
**Test:** Compare `gaussian` vs `gaussian_scaled` vs `orthogonal`
- If `gaussian_scaled` and `orthogonal` are more robust → absolute magnitude matters
- If similar breakdown → relative perturbation matters more

### Hypothesis 2: Direction matters
**Test:** Compare `gaussian` vs `orthogonal` vs `targeted`
- If `orthogonal` is more robust → direction parallel to signal is critical
- If `orthogonal` breaks similarly → any perturbation is harmful

### Hypothesis 3: Amplification vs reduction
**Test:** Compare `targeted (same)` vs `targeted (opposite)`
- Which causes breakdown faster?
- Does the system need precise magnitude or just direction?

## Typical Hidden State Norms

Based on common LLMs (Llama, GPT):
- Hidden state norms: ~10-50 (varies by layer and model)
- For `gaussian` noise to compete with signal: scale ≥ 1.0
- For `gaussian_scaled` with scale=1.0: noise norm = hidden state norm

## Usage in test.py

Simply edit the `main()` function in `test.py`:

```python
# Uncomment the configuration you want to test
# NOISE_TYPE = "gaussian_scaled"
# NOISE_DIRECTION = None
# NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# OUTPUT_FILE = "results_gaussian_scaled.json"
```

Then run:
```bash
python test.py
```

## Interpreting Results

The script will output:
1. Accuracy by noise scale
2. Status indicators (Good / Degraded / Broken)
3. Sample outputs showing degradation
4. JSON file with full results

**Look for:**
- **Breakdown threshold:** At what scale does accuracy drop below 50%?
- **Degradation pattern:** Sudden cliff or gradual decay?
- **Output quality:** Nonsensical text, wrong reasoning, or just wrong answer?

## Quick Start Example

To find the breakdown threshold with scaled Gaussian noise:

1. Edit `test.py`, uncomment:
```python
NOISE_TYPE = "gaussian_scaled"
NOISE_DIRECTION = None
NOISE_SCALES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
OUTPUT_FILE = "results_gaussian_scaled.json"
```

2. Run: `python test.py`

3. Check where accuracy drops significantly in the output

4. Refine scale range around breakdown point for finer granularity
