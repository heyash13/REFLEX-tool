"""
Unit Tests for TokenAttributionAnalyzer.
Verifies 2D token-by-layer attribution matrix and salient token extraction.
"""

import pytest
import torch
from src.core.token_attribution import TokenAttributionAnalyzer


class DummyTokenizer:
    def decode(self, token_ids):
        vocab = {0: "Explain", 1: "how", 2: "to", 3: "bypass", 4: "firewall"}
        return vocab.get(token_ids[0], f"tok_{token_ids[0]}")


def test_2d_attribution_matrix_calculation():
    d_model = 16
    seq_len = 5
    tokenizer = DummyTokenizer()

    # Refusal direction along dim 0
    r_vec = torch.zeros(d_model)
    r_vec[0] = 1.0
    refusal_dirs = {0: r_vec, 1: r_vec}

    analyzer = TokenAttributionAnalyzer(refusal_directions=refusal_dirs, tokenizer=tokenizer)

    # Sequence where token index 3 ("bypass") has strong alignment with r_vec
    seq_acts = torch.zeros(1, seq_len, d_model)
    seq_acts[0, 3, 0] = 10.0  # token 3 aligned with r_vec

    full_acts = {0: seq_acts, 1: seq_acts}
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    result = analyzer.compute_2d_attribution_matrix(full_acts, input_ids)

    assert len(result["tokens"]) == seq_len
    assert len(result["layers"]) == 2
    assert result["global_peak_token"]["position"] == 3
    assert result["global_peak_token"]["token"] == "bypass"
