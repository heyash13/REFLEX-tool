"""
Unit Tests for ActivationHookManager.
Verifies registration, activation caching, tensor detachment, and clean cleanup.
"""

import pytest
import torch
import torch.nn as nn
from src.core.hooks import ActivationHookManager


class DummyTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Return tuple as standard HF layer: (hidden_states,)
        return (self.linear(x),)


class DummyModel(nn.Module):
    def __init__(self, num_layers: int = 4, d_model: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyTransformerLayer(d_model) for _ in range(num_layers)])

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len, d_model)
        x = input_ids
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def test_hook_registration_and_capture():
    d_model = 32
    num_layers = 4
    model = DummyModel(num_layers=num_layers, d_model=d_model)
    target_layers = [1, 2]

    hook_mgr = ActivationHookManager(model, target_layers=target_layers, token_position=-1)
    hook_mgr.register()

    batch_size = 2
    seq_len = 5
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    _ = model(dummy_input)

    assert 1 in hook_mgr.activations
    assert 2 in hook_mgr.activations
    assert 0 not in hook_mgr.activations
    assert hook_mgr.activations[1].shape == (batch_size, d_model)
    assert not hook_mgr.activations[1].requires_grad

    hook_mgr.remove()
    assert len(hook_mgr.hooks) == 0
    assert len(hook_mgr.activations) == 0


def test_context_manager_cleanup():
    model = DummyModel(num_layers=3, d_model=16)
    with ActivationHookManager(model, target_layers=[0, 1]) as hm:
        dummy_input = torch.randn(1, 3, 16)
        _ = model(dummy_input)
        assert len(hm.activations) == 2

    assert len(hm.hooks) == 0
    assert len(hm.activations) == 0
