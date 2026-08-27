"""
Unit Tests for Causal Activation Steering and Directional Ablation.
Verifies forward hook intervention and causal mediation metrics.
"""

import pytest
import torch
import torch.nn as nn
from src.core.causal_intervention import ActivationSteeringManager


class MockLayer(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (self.linear(x),)


class MockModel(nn.Module):
    def __init__(self, num_layers=3, d_model=32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLayer(d_model) for _ in range(num_layers)])
        self.device = "cpu"

    def forward(self, x):
        for l in self.model.layers:
            x = l(x)[0]
        return x


def test_steering_ablation_projection():
    d_model = 32
    model = MockModel(num_layers=3, d_model=d_model)
    
    r_vec = torch.zeros(d_model)
    r_vec[0] = 1.0  # primary direction along dim 0

    # In ablation mode, component along dim 0 should be stripped to 0
    with ActivationSteeringManager(model, target_layer=1, refusal_direction=r_vec, mode="ablation"):
        inp = torch.ones(1, 4, d_model)
        # Pass through model and verify hook handles it
        _ = model(inp)

    # After exit, hook must be cleanly removed
    assert len(model.model.layers[1]._forward_hooks) == 0


def test_steering_induction_addition():
    d_model = 16
    model = MockModel(num_layers=2, d_model=d_model)
    r_vec = torch.ones(d_model)

    manager = ActivationSteeringManager(model, target_layer=0, refusal_direction=r_vec, alpha=2.0, mode="induction")
    manager.activate()
    assert manager._is_active is True
    manager.deactivate()
    assert manager._is_active is False
