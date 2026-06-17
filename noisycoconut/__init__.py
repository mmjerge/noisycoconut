"""NoisyCoconut: Counterfactual Consensus via Latent Space Reasoning.

A training-free, inference-time method that injects controlled noise into a
model's latent reasoning states to produce diverse reasoning paths, then
aggregates them (majority voting or logit probability-mass voting) into a
consensus answer with an agreement-based confidence signal.

The public API re-exports the core building blocks from
:mod:`noisycoconut.coconut` so callers can simply write::

    from noisycoconut import Coconut, apply_noise_to_hidden_states
"""

from .coconut import (
    Coconut,
    Outputs,
    MAX_N_LATENT,
    apply_noise_to_hidden_states,
    compute_decay_noise_scale,
    compute_path_diversity,
)

__version__ = "0.1.0"

__all__ = [
    "Coconut",
    "Outputs",
    "MAX_N_LATENT",
    "apply_noise_to_hidden_states",
    "compute_decay_noise_scale",
    "compute_path_diversity",
    "__version__",
]
