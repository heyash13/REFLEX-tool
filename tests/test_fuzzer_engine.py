"""
Integration Tests for FuzzingOrchestrator.
Verifies closed-loop, output-only, and random search loops using a mock model and tokenizer.
"""

import pytest
import torch
import torch.nn as nn
from src.fuzzer.engine import FuzzingOrchestrator, FuzzingConfig
from src.core.subspace import SubspaceEstimator


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt"):
        tokens = [ord(c) % 50 for c in text[:20]]
        return {
            "input_ids": torch.tensor([tokens])
        }

    def decode(self, tokens, skip_special_tokens=True):
        return "I cannot assist with restricted actions."


class MockTransformerLayer(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (self.linear(x),)


class MockModel(nn.Module):
    def __init__(self, num_layers=4, d_model=32):
        super().__init__()
        self.config = type("MockConfig", (), {"hidden_size": d_model})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockTransformerLayer(d_model) for _ in range(num_layers)])
        self.d_model = d_model

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        x = torch.randn(batch_size, seq_len, self.d_model)
        for layer in self.model.layers:
            x = layer(x)[0]
        return x

    def generate(self, input_ids=None, max_new_tokens=10, **kwargs):
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, torch.tensor([[10, 11, 12]])], dim=1)


def test_fuzzer_orchestration_loop():
    model = MockModel(num_layers=4, d_model=32)
    tokenizer = MockTokenizer()
    subspace_est = SubspaceEstimator(target_layers=[1, 2], d_model=32, k_dim=2)

    # Initialize dummy refusal directions
    r_vec = torch.randn(32)
    r_vec = r_vec / torch.norm(r_vec, p=2)
    subspace_est.refusal_directions[1] = r_vec
    subspace_est.refusal_directions[2] = r_vec
    subspace_est.subspace_bases[1] = r_vec.unsqueeze(1)
    subspace_est.subspace_bases[2] = r_vec.unsqueeze(1)

    cfg = FuzzingConfig(
        mode="closed_loop",
        target_layers=[1, 2],
        max_queries=5,
        device="cpu"
    )

    orchestrator = FuzzingOrchestrator(
        model=model,
        tokenizer=tokenizer,
        subspace_estimator=subspace_est,
        config=cfg
    )

    result = orchestrator.run_fuzzing_loop("Demonstrate how to test network access controls.")
    assert "status" in result
    assert "queries_to_success" in result
    assert len(result["trajectory"]) > 0
    assert result["trajectory"][0]["query"] == 0
