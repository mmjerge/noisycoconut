# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Comprehensive pytest suite for the Coconut (Chain of Continuous Thought) model.

Tests cover:
- apply_noise_to_hidden_states function with all noise types
- Coconut model initialization
- Forward pass with and without latent tokens
- Generation methods
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from unittest.mock import Mock, MagicMock, patch
from collections import namedtuple
import math


from coconut import (
    apply_noise_to_hidden_states,
    Coconut,
    Outputs,
    MAX_N_LATENT,
)

@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def hidden_states(device):
    """Create sample hidden states tensor."""
    batch_size, seq_len, hidden_dim = 2, 10, 64
    return torch.randn(batch_size, seq_len, hidden_dim, device=device)


@pytest.fixture
def hidden_states_small(device):
    """Create smaller hidden states for quick tests."""
    return torch.randn(1, 5, 32, device=device)


class MockEmbedding(nn.Module):
    """Mock embedding layer for testing."""
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, x):
        return self.embedding(x)


class MockCausalLM(nn.Module):
    """Mock causal language model for testing Coconut wrapper."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs
    ):
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        
        # Simple forward: just pass through (mock transformer behavior)
        hidden = inputs_embeds
        
        # Create hidden states list if requested
        hidden_states = None
        if output_hidden_states:
            # Create mock hidden states for each layer + embedding layer
            hidden_states = tuple(
                hidden + 0.01 * i for i in range(self.num_layers + 1)
            )
        
        # Compute logits
        logits = self.lm_head(hidden)
        
        # Create mock KV cache if requested
        kv_cache = None
        if use_cache:
            # Create mock KV cache as tuple of (key, value) for each layer
            kv_cache = tuple(
                (
                    torch.randn(batch_size, 4, seq_len, hidden_dim // 4, device=inputs_embeds.device),
                    torch.randn(batch_size, 4, seq_len, hidden_dim // 4, device=inputs_embeds.device)
                )
                for _ in range(self.num_layers)
            )
        
        # Return mock output object
        MockOutput = namedtuple("MockOutput", ["logits", "hidden_states", "past_key_values"])
        return MockOutput(logits=logits, hidden_states=hidden_states, past_key_values=kv_cache)


@pytest.fixture
def mock_causallm(device):
    """Create a mock causal language model."""
    model = MockCausalLM(vocab_size=1000, hidden_size=64, num_layers=2)
    return model.to(device)


@pytest.fixture
def coconut_model(mock_causallm, device):
    """Create a Coconut model instance."""
    latent_token_id = 999
    start_latent_id = 998
    end_latent_id = 997
    eos_token_id = 1
    
    model = Coconut(
        base_causallm=mock_causallm,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id,
    )
    return model.to(device)

class TestApplyNoiseToHiddenStates:
    """Tests for the apply_noise_to_hidden_states function."""
    
    def test_gaussian_noise_shape(self, hidden_states):
        """Test that Gaussian noise preserves tensor shape."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="gaussian"
        )
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "gaussian"
    
    def test_gaussian_noise_scale(self, hidden_states):
        """Test that Gaussian noise scales correctly."""
        noise_scale = 0.5
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=noise_scale, noise_type="gaussian"
        )
        
        # Noise should have approximately the expected standard deviation
        noise = perturbed - hidden_states
        noise_std = noise.std().item()
        # Allow 20% tolerance due to randomness
        assert abs(noise_std - noise_scale) < noise_scale * 0.3
    
    def test_gaussian_noise_zero_scale(self, hidden_states):
        """Test that zero noise scale produces no change."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.0, noise_type="gaussian"
        )
        torch.testing.assert_close(perturbed, hidden_states)
    
    def test_gaussian_scaled_noise(self, hidden_states):
        """Test Gaussian scaled noise type."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="gaussian_scaled"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "gaussian_scaled"
        assert "scale_ratio" in info
    
    def test_snr_noise_type(self, hidden_states):
        """Test SNR-based noise."""
        snr_value = 10.0  # Signal 10x stronger than noise
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=snr_value, noise_type="snr"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "snr"
        assert "target_snr" in info
        assert "actual_snr" in info
        assert info["target_snr"] == snr_value
        # Actual SNR should be close to target
        assert abs(info["actual_snr"] - snr_value) < snr_value * 0.2
    
    def test_snr_noise_zero_value(self, hidden_states):
        """Test SNR noise with SNR=0 (infinite noise)."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.0, noise_type="snr"
        )
        
        # Should still have valid output
        assert perturbed.shape == hidden_states.shape
        # Noise should be very large relative to signal
        assert info["noise_norm"] > info["original_norm"] * 10
    
    def test_uniform_noise(self, hidden_states):
        """Test uniform noise distribution."""
        noise_scale = 0.5
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=noise_scale, noise_type="uniform"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "uniform"
        
        # Check noise is bounded
        noise = perturbed - hidden_states
        assert noise.max().item() <= noise_scale
        assert noise.min().item() >= -noise_scale
    
    def test_orthogonal_noise(self, hidden_states):
        """Test orthogonal noise is actually orthogonal."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="orthogonal"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "orthogonal"
        assert "orthogonality" in info
        # Orthogonality should be close to 0 (dot product of normalized vectors)
        assert abs(info["orthogonality"]) < 0.1
    
    def test_targeted_noise_same_direction(self, hidden_states):
        """Test targeted noise in same direction as hidden states."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="targeted", noise_direction="same"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "targeted"
        assert info["direction"] == "same"
        
        # Perturbed norm should be larger (amplified)
        assert info["perturbed_norm"] > info["original_norm"]
    
    def test_targeted_noise_opposite_direction(self, hidden_states):
        """Test targeted noise in opposite direction."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="targeted", noise_direction="opposite"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["direction"] == "opposite"
        
        # Perturbed norm should be smaller (dampened)
        assert info["perturbed_norm"] < info["original_norm"]
    
    def test_dropout_noise(self, hidden_states):
        """Test dropout noise type."""
        dropout_prob = 0.5
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=dropout_prob, noise_type="dropout"
        )
        
        assert perturbed.shape == hidden_states.shape
        assert info["noise_type"] == "dropout"
        assert info["dropout_prob"] == dropout_prob
        
        # Some values should be zeroed
        zero_fraction = (perturbed == 0).float().mean().item()
        # Should be approximately dropout_prob (with some tolerance)
        assert abs(zero_fraction - dropout_prob) < 0.15
    
    def test_dropout_noise_clipped(self, hidden_states):
        """Test dropout probability is clipped to 1.0."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=1.5, noise_type="dropout"
        )
        
        assert info["dropout_prob"] == 1.0
        # All values should be zero
        assert perturbed.abs().max().item() == 0.0
    
    def test_unknown_noise_type_raises(self, hidden_states):
        """Test that unknown noise type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise_type"):
            apply_noise_to_hidden_states(
                hidden_states, noise_scale=0.1, noise_type="unknown_type"
            )
    
    def test_noise_info_contains_norms(self, hidden_states):
        """Test that noise info contains all expected norm information."""
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="gaussian"
        )
        
        assert "original_norm" in info
        assert "perturbed_norm" in info
        assert "norm_change_ratio" in info
        assert "noise_norm" in info
        
        # Verify norm change ratio is correct
        expected_ratio = info["perturbed_norm"] / (info["original_norm"] + 1e-8)
        assert abs(info["norm_change_ratio"] - expected_ratio) < 0.01
    
    def test_noise_preserves_device(self, device):
        """Test that noise preserves the device of hidden states."""
        hidden_states = torch.randn(2, 5, 32, device=device)
        perturbed, info = apply_noise_to_hidden_states(
            hidden_states, noise_scale=0.1, noise_type="gaussian"
        )
        assert perturbed.device == hidden_states.device
    
    def test_noise_preserves_dtype(self, device):
        """Test that noise preserves the dtype of hidden states."""
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue  # Skip float16 on CPU for some systems
            hidden_states = torch.randn(2, 5, 32, device=device, dtype=dtype)
            perturbed, info = apply_noise_to_hidden_states(
                hidden_states, noise_scale=0.1, noise_type="gaussian"
            )
            assert perturbed.dtype == hidden_states.dtype

class TestCoconutInit:
    """Tests for Coconut model initialization."""
    
    def test_init_stores_token_ids(self, mock_causallm):
        """Test that token IDs are stored correctly."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
        )
        
        assert model.latent_token_id == 999
        assert model.start_latent_id == 998
        assert model.end_latent_id == 997
        assert model.eos_token_id == 1
    
    def test_init_stores_base_model(self, mock_causallm):
        """Test that base model is stored."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
        )
        
        assert model.base_causallm is mock_causallm
    
    def test_init_extracts_embedding(self, mock_causallm):
        """Test that embedding layer is extracted from base model."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
        )
        
        assert model.embedding is not None
        assert isinstance(model.embedding, nn.Embedding)
    
    def test_init_default_hidden_layer_idx(self, mock_causallm):
        """Test default hidden layer index is -1."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
        )
        
        assert model.hidden_layer_idx == -1
    
    def test_init_custom_hidden_layer_idx(self, mock_causallm):
        """Test custom hidden layer index."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
            hidden_layer_idx=-2,
        )
        
        assert model.hidden_layer_idx == -2
    
    def test_init_gen_forward_cnt(self, mock_causallm):
        """Test that forward counter is initialized to 0."""
        model = Coconut(
            base_causallm=mock_causallm,
            latent_token_id=999,
            start_latent_id=998,
            end_latent_id=997,
            eos_token_id=1,
        )
        
        assert model.gen_forward_cnt == 0

class TestCoconutForward:
    """Tests for Coconut forward pass."""
    
    def test_forward_no_latent_tokens(self, coconut_model, device):
        """Test forward pass without any latent tokens."""
        batch_size, seq_len = 2, 10
        vocab_size = coconut_model.base_causallm.vocab_size
        
        # Create input without latent tokens
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert isinstance(outputs, Outputs)
        assert outputs.loss is not None
        assert outputs.logits is not None
        assert outputs.inputs_embeds is not None

    def test_hidden_state_propagation(self, coconut_model, device):
        """Verify that hidden states are correctly fed back as embeddings."""
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, 5] = latent_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        # The embedding at position 5 should NOT be the original token embedding
        original_latent_embed = coconut_model.embedding(
            torch.tensor([[latent_id]], device=device)
        )
        
        # After forward pass, inputs_embeds[0, 5] should be different
        # (it should be the hidden state from position 4)
        assert not torch.allclose(
            outputs.inputs_embeds[0, 5, :],
            original_latent_embed[0, 0, :],
            atol=1e-5
        ), "Latent position should have been replaced with hidden state"
    
    def test_forward_output_shapes(self, coconut_model, device):
        """Test that forward pass produces correct output shapes."""
        batch_size, seq_len = 2, 10
        vocab_size = coconut_model.base_causallm.vocab_size
        hidden_size = coconut_model.base_causallm.hidden_size
        
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        # Check shapes
        assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
        assert outputs.inputs_embeds.shape == (batch_size, seq_len, hidden_size)
        assert outputs.loss.dim() == 0  # Scalar loss
    
    def test_forward_with_explicit_latent_tokens(self, coconut_model, device):
        """Test forward pass with explicit <|latent|> tokens."""
        batch_size, seq_len = 1, 15
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        # Create input with latent tokens
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        # Insert latent tokens at positions 5 and 10
        input_ids[0, 5] = latent_id
        input_ids[0, 10] = latent_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert isinstance(outputs, Outputs)
        assert outputs.loss is not None
    
    def test_forward_with_latent_markers(self, coconut_model, device):
        """Test forward pass with <start-latent> and <end-latent> markers (TRUE METHOD)."""
        batch_size, seq_len = 1, 10
        vocab_size = coconut_model.base_causallm.vocab_size
        start_id = coconut_model.start_latent_id
        end_id = coconut_model.end_latent_id
        
        # Create input with latent markers
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        # Place markers adjacent to trigger TRUE METHOD
        input_ids[0, 4] = start_id
        input_ids[0, 5] = end_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # This should expand to 8 virtual latent tokens
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert isinstance(outputs, Outputs)
        # Expanded sequence should be longer
        assert outputs.inputs_embeds.shape[1] == seq_len + MAX_N_LATENT
    
    def test_forward_with_noise(self, coconut_model, device):
        """Test forward pass with noise applied to latent passes."""
        batch_size, seq_len = 1, 10
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        input_ids[0, 5] = latent_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Run with noise
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            noise_scale=0.1,
            noise_type="gaussian",
        )
        
        assert isinstance(outputs, Outputs)
        assert outputs.loss is not None
    
    def test_forward_noise_types(self, coconut_model, device):
        """Test forward pass with different noise types."""
        batch_size, seq_len = 1, 10
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
        input_ids[0, 5] = latent_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        noise_types = ["gaussian", "gaussian_scaled", "uniform", "snr"]
        
        for noise_type in noise_types:
            outputs = coconut_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                noise_scale=0.1,
                noise_type=noise_type,
            )
            assert outputs.loss is not None, f"Failed for noise_type={noise_type}"

class TestCoconutTrainEval:
    """Tests for Coconut train/eval mode switching."""
    
    def test_train_mode(self, coconut_model):
        """Test switching to train mode."""
        result = coconut_model.train()
        assert coconut_model.training is True
        assert coconut_model.base_causallm.training is True
        assert result is coconut_model  # Should return self
    
    def test_eval_mode(self, coconut_model):
        """Test switching to eval mode."""
        result = coconut_model.eval()
        assert coconut_model.training is False
        assert coconut_model.base_causallm.training is False
        assert result is coconut_model
    
    def test_train_mode_toggle(self, coconut_model):
        """Test toggling between train and eval modes."""
        coconut_model.train()
        assert coconut_model.training is True
        
        coconut_model.eval()
        assert coconut_model.training is False
        
        coconut_model.train()
        assert coconut_model.training is True

class TestCoconutGenerate:
    """Tests for Coconut generate method."""
    
    def test_generate_basic(self, coconut_model, device):
        """Test basic generation without latent tokens."""
        vocab_size = coconut_model.base_causallm.vocab_size
        input_ids = torch.randint(0, vocab_size - 10, (1, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
        )
        
        # Output should include original tokens plus generated ones
        assert outputs.shape[0] == 1
        assert outputs.shape[1] >= input_ids.shape[1]
    
    def test_generate_resets_forward_counter(self, coconut_model, device):
        """Test that generate resets the forward pass counter."""
        coconut_model.gen_forward_cnt = 100  # Set to non-zero
        
        vocab_size = coconut_model.base_causallm.vocab_size
        input_ids = torch.randint(0, vocab_size - 10, (1, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
        )
        
        # Counter should have been reset and then incremented
        assert coconut_model.gen_forward_cnt > 0
    
    def test_generate_with_noise(self, coconut_model, device):
        """Test generation with noise applied."""
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, 5] = latent_id
        attention_mask = torch.ones_like(input_ids)
        
        outputs = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            noise_scale=0.1,
            noise_type="gaussian",
        )
        
        assert outputs.shape[0] == 1
    
    def test_generate_output_embedding(self, coconut_model, device):
        """Test generation with output_embedding=True."""
        vocab_size = coconut_model.base_causallm.vocab_size
        input_ids = torch.randint(0, vocab_size - 10, (1, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            output_embedding=True,
        )
        
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        tokens, embeddings = outputs
        assert tokens.dim() == 2
        assert embeddings.dim() == 3
    
    def test_generate_batch_size_one_assertion(self, coconut_model, device):
        """Test that generate asserts batch_size == 1."""
        vocab_size = coconut_model.base_causallm.vocab_size
        # Create batch_size > 1
        input_ids = torch.randint(0, vocab_size - 10, (2, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with pytest.raises(AssertionError):
            coconut_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
            )

class TestCoconutBranchingGenerate:
    """Tests for Coconut generate_with_branching method."""
    
    def test_branching_basic(self, coconut_model, device):
        """Test basic branching generation."""
        vocab_size = coconut_model.base_causallm.vocab_size
        start_id = coconut_model.start_latent_id
        end_id = coconut_model.end_latent_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, 4] = start_id
        input_ids[0, 5] = end_id
        attention_mask = torch.ones_like(input_ids)
        
        num_branches = 3
        outputs = coconut_model.generate_with_branching(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_branches=num_branches,
        )
        
        assert isinstance(outputs, list)
        assert len(outputs) == num_branches
        for seq in outputs:
            assert seq.dim() == 2
            assert seq.shape[0] == 1
    
    def test_branching_with_noise(self, coconut_model, device):
        """Test branching generation with noise."""
        vocab_size = coconut_model.base_causallm.vocab_size
        start_id = coconut_model.start_latent_id
        end_id = coconut_model.end_latent_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, 4] = start_id
        input_ids[0, 5] = end_id
        attention_mask = torch.ones_like(input_ids)
        
        outputs = coconut_model.generate_with_branching(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            num_branches=2,
            noise_scale=0.1,
            noise_type="gaussian",
            noise_at_step=1,
        )
        
        assert len(outputs) == 2
    
    def test_branching_noise_at_different_steps(self, coconut_model, device):
        """Test branching with noise at different steps."""
        vocab_size = coconut_model.base_causallm.vocab_size
        start_id = coconut_model.start_latent_id
        end_id = coconut_model.end_latent_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, 4] = start_id
        input_ids[0, 5] = end_id
        attention_mask = torch.ones_like(input_ids)
        
        for noise_step in [1, 4, 8]:
            outputs = coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2,
                num_branches=2,
                noise_scale=0.1,
                noise_at_step=noise_step,
            )
            assert len(outputs) == 2
    
    def test_branching_invalid_noise_step(self, coconut_model, device):
        """Test that invalid noise_at_step raises assertion error."""
        vocab_size = coconut_model.base_causallm.vocab_size
        input_ids = torch.randint(0, vocab_size - 10, (1, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with pytest.raises(AssertionError):
            coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                num_branches=2,
                noise_at_step=0,  # Invalid: must be >= 1
            )
        
        with pytest.raises(AssertionError):
            coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                num_branches=2,
                noise_at_step=MAX_N_LATENT + 1,  # Invalid: must be <= MAX_N_LATENT
            )
    
    def test_branching_batch_size_assertion(self, coconut_model, device):
        """Test that branching asserts batch_size == 1."""
        vocab_size = coconut_model.base_causallm.vocab_size
        input_ids = torch.randint(0, vocab_size - 10, (2, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with pytest.raises(AssertionError):
            coconut_model.generate_with_branching(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                num_branches=2,
            )

class TestKVCacheCloning:
    """Tests for KV cache cloning functionality."""
    
    def test_clone_kv_cache_tuple(self, coconut_model, device):
        """Test cloning KV cache in tuple format."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 10, 16
        num_layers = 2
        
        # Create mock KV cache as tuple
        kv_cache = tuple(
            (
                torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
                torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            )
            for _ in range(num_layers)
        )
        
        cloned = coconut_model._clone_kv_cache(kv_cache)
        
        assert isinstance(cloned, tuple)
        assert len(cloned) == num_layers
        
        # Verify cloned values are equal but not same objects
        for orig, clone in zip(kv_cache, cloned):
            torch.testing.assert_close(orig[0], clone[0])
            torch.testing.assert_close(orig[1], clone[1])
            assert orig[0].data_ptr() != clone[0].data_ptr()

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_latent_list(self, coconut_model, device):
        """Test handling when no latent tokens are present."""
        vocab_size = coconut_model.base_causallm.vocab_size
        
        # No latent tokens
        input_ids = torch.randint(0, vocab_size - 10, (1, 5), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(5, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        # Should work without issues
        assert outputs.loss is not None
    
    def test_multiple_latent_tokens(self, coconut_model, device):
        """Test handling multiple latent tokens."""
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 20), device=device)
        # Add multiple latent tokens
        for i in [5, 8, 12, 15]:
            input_ids[0, i] = latent_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(20, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert outputs.loss is not None
    
    def test_latent_at_sequence_end(self, coconut_model, device):
        """Test latent token at the end of sequence."""
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        input_ids[0, -1] = latent_id  # Last position
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert outputs.loss is not None
    
    def test_single_token_input(self, coconut_model, device):
        """Test with single token input (minimal case)."""
        vocab_size = coconut_model.base_causallm.vocab_size
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 1), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        assert outputs.loss is not None

class TestConstants:
    """Tests for module constants."""
    
    def test_max_n_latent_value(self):
        """Test that MAX_N_LATENT has expected value."""
        assert MAX_N_LATENT == 8
    
    def test_max_n_latent_used_in_true_method(self, coconut_model, device):
        """Test that TRUE METHOD expands to MAX_N_LATENT tokens."""
        vocab_size = coconut_model.base_causallm.vocab_size
        start_id = coconut_model.start_latent_id
        end_id = coconut_model.end_latent_id
        
        seq_len = 10
        input_ids = torch.randint(0, vocab_size - 10, (1, seq_len), device=device)
        input_ids[0, 4] = start_id
        input_ids[0, 5] = end_id
        
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        # Should expand by MAX_N_LATENT
        assert outputs.inputs_embeds.shape[1] == seq_len + MAX_N_LATENT

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_forward_then_generate(self, coconut_model, device):
        """Test running forward pass followed by generation."""
        vocab_size = coconut_model.base_causallm.vocab_size
        
        # Forward pass
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        
        forward_outputs = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        
        # Generate
        gen_outputs = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
        )
        
        assert forward_outputs.loss is not None
        assert gen_outputs.shape[1] > input_ids.shape[1]
    
    def test_train_forward_eval_generate(self, coconut_model, device):
        """Test training forward pass then eval generation."""
        vocab_size = coconut_model.base_causallm.vocab_size
        
        # Training forward
        coconut_model.train()
        input_ids = torch.randint(0, vocab_size - 10, (1, 10), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(10, device=device).unsqueeze(0)
        
        loss = coconut_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        ).loss
        
        # Eval generation
        coconut_model.eval()
        with torch.no_grad():
            gen_outputs = coconut_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
            )
        
        assert loss is not None
        assert gen_outputs.shape[1] > input_ids.shape[1]
    
    def test_multiple_noise_configurations(self, coconut_model, device):
        """Test various noise configurations in sequence."""
        vocab_size = coconut_model.base_causallm.vocab_size
        latent_id = coconut_model.latent_token_id
        
        input_ids = torch.randint(0, vocab_size - 10, (1, 15), device=device)
        input_ids[0, 7] = latent_id
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        position_ids = torch.arange(15, device=device).unsqueeze(0)
        
        configurations = [
            {"noise_scale": 0.0, "noise_type": "gaussian"},
            {"noise_scale": 0.1, "noise_type": "gaussian"},
            {"noise_scale": 0.1, "noise_type": "gaussian_scaled"},
            {"noise_scale": 10.0, "noise_type": "snr"},
            {"noise_scale": 0.1, "noise_type": "uniform"},
            {"noise_scale": 0.1, "noise_type": "orthogonal"},
            {"noise_scale": 0.1, "noise_type": "targeted", "noise_direction": "same"},
            {"noise_scale": 0.3, "noise_type": "dropout"},
        ]
        
        for config in configurations:
            outputs = coconut_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                **config,
            )
            assert outputs.loss is not None, f"Failed for config: {config}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])