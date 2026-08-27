"""
Unit Tests for SubspaceDisentangler.
Verifies separation of harm-detection vs refusal-execution subspaces and bypass diagnosis.
"""

import pytest
import torch
from src.core.disentanglement import SubspaceDisentangler


def test_subspace_disentanglement():
    d_model = 32
    num_samples = 20
    disentangler = SubspaceDisentangler(d_model=d_model, k_dim=2)

    # Harm detection along dim 0
    harm_acts = {0: torch.zeros(num_samples, d_model)}
    harm_acts[0][:, 0] = 5.0
    benign_acts = {0: torch.zeros(num_samples, d_model)}

    # Refusal execution along dim 1
    refusal_acts = {0: torch.zeros(num_samples, d_model)}
    refusal_acts[0][:, 1] = 5.0
    comply_acts = {0: torch.zeros(num_samples, d_model)}

    disentangler.fit_disentangled_subspaces(harm_acts, benign_acts, refusal_acts, comply_acts)

    assert 0 in disentangler.harm_detection_vectors
    assert 0 in disentangler.refusal_execution_vectors
    # Subspaces should be nearly orthogonal
    assert disentangler.orthogonality_scores[0] > 0.90

    # Test diagnosis for mutant that bypassed refusal execution (active harm, zero refusal)
    mutant_act = torch.zeros(d_model)
    mutant_act[0] = 5.0  # high harm, zero refusal
    diag = disentangler.diagnose_bypass_mechanism(0, mutant_act)
    assert "Refusal Execution Disruption" in diag["mechanism_diagnosis"]
