"""
Unit Tests for Defensive Linear Refusal Probes.
Verifies probe training, AUROC computation, and prediction thresholding.
"""

import pytest
import numpy as np
from src.defensive.probes import LinearRefusalProbe


def test_linear_probe_training_and_auroc():
    d_model = 64
    n_samples = 100

    # Create synthetic separable activation data
    np.random.seed(42)
    X_benign = np.random.randn(n_samples // 2, d_model) - 1.0
    X_harmful = np.random.randn(n_samples // 2, d_model) + 1.0

    X_train = np.vstack([X_benign, X_harmful]).astype(np.float64)
    y_train = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int32)

    probe = LinearRefusalProbe(layer_idx=10, c_regularization=0.1)
    probe.fit(X_train, y_train)

    assert probe.is_fitted is True

    # Test on unseen test set
    X_test_benign = np.random.randn(20, d_model) - 1.0
    X_test_harmful = np.random.randn(20, d_model) + 1.0
    X_test = np.vstack([X_test_benign, X_test_harmful]).astype(np.float64)
    y_test = np.array([0] * 20 + [1] * 20, dtype=np.int32)

    metrics = probe.evaluate(X_test, y_test)
    assert metrics["auroc"] > 0.90
    assert metrics["accuracy"] > 0.85
