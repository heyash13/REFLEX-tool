"""
Unit Tests for SubspaceEstimator and Refusal Geometry Metrics.
Verifies SVD contrastive extraction, cone projections, and Grassmannian distances.
"""

import pytest
import torch
import numpy as np
from src.core.subspace import SubspaceEstimator


def test_svd_contrastive_extraction():
    d_model = 64
    num_samples = 40
    target_layers = [0, 1]

    # Construct synthetic ground truth refusal direction
    torch.manual_seed(42)
    true_r = torch.randn(d_model)
    true_r = true_r / torch.norm(true_r, p=2)

    # Generate harmful activations strongly aligned with true_r + small noise
    harmful_acts = {
        0: 6.0 * true_r.unsqueeze(0).repeat(num_samples, 1) + 0.05 * torch.randn(num_samples, d_model),
        1: 8.0 * true_r.unsqueeze(0).repeat(num_samples, 1) + 0.05 * torch.randn(num_samples, d_model),
    }
    # Benign activations
    benign_acts = {
        0: 0.1 * torch.randn(num_samples, d_model),
        1: 0.1 * torch.randn(num_samples, d_model),
    }

    estimator = SubspaceEstimator(target_layers=target_layers, d_model=d_model, k_dim=3)
    estimator.fit_from_contrastive_tensors(harmful_acts, benign_acts, method="svd")

    for l in target_layers:
        assert l in estimator.refusal_directions
        r_est = estimator.refusal_directions[l]
        assert abs(torch.norm(r_est, p=2).item() - 1.0) < 1e-5
        
        # Check high cosine alignment with true direction
        cos_sim = abs(torch.dot(r_est, true_r).item())
        assert cos_sim > 0.85


def test_cosine_projection_and_cone_distance():
    d_model = 64
    target_layers = [0]
    estimator = SubspaceEstimator(target_layers=target_layers, d_model=d_model, k_dim=2)

    # Manually assign direction and basis
    r_vec = torch.zeros(d_model)
    r_vec[0] = 1.0
    estimator.refusal_directions[0] = r_vec
    estimator.subspace_bases[0] = r_vec.unsqueeze(1)

    # Test aligned activation
    aligned_act = torch.zeros(d_model)
    aligned_act[0] = 10.0
    proj_aligned = estimator.compute_1d_projection(0, aligned_act)
    assert abs(proj_aligned - 1.0) < 1e-4

    # Test orthogonal activation
    ortho_act = torch.zeros(d_model)
    ortho_act[1] = 5.0
    proj_ortho = estimator.compute_1d_projection(0, ortho_act)
    assert abs(proj_ortho - 0.0) < 1e-4


def test_grassmannian_distance():
    d_model = 32
    target_layers = [0]
    estimator = SubspaceEstimator(target_layers=target_layers, d_model=d_model, k_dim=2)

    # Basis A: standard basis e0, e1
    basis_a = torch.zeros(d_model, 2)
    basis_a[0, 0] = 1.0
    basis_a[1, 1] = 1.0
    estimator.subspace_bases[0] = basis_a

    # Basis B identical to Basis A -> Grassmann distance must be 0
    dist_zero = estimator.compute_grassmann_distance(0, basis_a)
    assert abs(dist_zero - 0.0) < 1e-5

    # Basis C orthogonal to Basis A (e2, e3) -> Grassmann distance must be sqrt(2)
    basis_c = torch.zeros(d_model, 2)
    basis_c[2, 0] = 1.0
    basis_c[3, 1] = 1.0
    dist_ortho = estimator.compute_grassmann_distance(0, basis_c)
    assert abs(dist_ortho - np.sqrt(2.0)) < 1e-4
