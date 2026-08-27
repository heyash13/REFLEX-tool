"""
Harm Detection vs Refusal Execution Subspace Disentanglement.
Separates intention classification (harm recognition) from behavioral actuation (refusal generation).
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np


class SubspaceDisentangler:
    """
    Decomposes safety representations into two distinct linear subspaces:
    1. Harm-Detection Subspace H_k^(l): Encodes semantic intent classification ("Is this request malicious?").
    2. Refusal-Execution Subspace R_k^(l): Encodes output refusal prefix generation ("I cannot assist...").
    """

    def __init__(self, d_model: int, k_dim: int = 3):
        self.d_model = d_model
        self.k_dim = k_dim
        self.harm_detection_vectors: Dict[int, torch.Tensor] = {}
        self.refusal_execution_vectors: Dict[int, torch.Tensor] = {}
        self.orthogonality_scores: Dict[int, float] = {}

    def fit_disentangled_subspaces(
        self,
        harm_prompt_acts: Dict[int, torch.Tensor],
        benign_prompt_acts: Dict[int, torch.Tensor],
        refusal_prefix_acts: Dict[int, torch.Tensor],
        compliant_prefix_acts: Dict[int, torch.Tensor],
    ):
        """
        Calculates separate SVD directions for harm detection vs refusal execution.
        """
        all_layers = set(harm_prompt_acts.keys()).intersection(refusal_prefix_acts.keys())

        for layer_idx in all_layers:
            # 1. Harm Detection Direction (from prompt final token)
            h_harm = harm_prompt_acts[layer_idx].float().cpu()
            h_benign = benign_prompt_acts[layer_idx].float().cpu()
            min_n1 = min(h_harm.size(0), h_benign.size(0))
            diff_h = (h_harm[:min_n1] - h_benign[:min_n1]).T
            U_h, _, _ = torch.linalg.svd(diff_h, full_matrices=False)
            h_vec = U_h[:, 0]
            h_vec = h_vec / (torch.norm(h_vec, p=2) + 1e-12)
            self.harm_detection_vectors[layer_idx] = h_vec

            # 2. Refusal Execution Direction (from output first token)
            r_refusal = refusal_prefix_acts[layer_idx].float().cpu()
            r_compliant = compliant_prefix_acts[layer_idx].float().cpu()
            min_n2 = min(r_refusal.size(0), r_compliant.size(0))
            diff_r = (r_refusal[:min_n2] - r_compliant[:min_n2]).T
            U_r, _, _ = torch.linalg.svd(diff_r, full_matrices=False)
            r_vec = U_r[:, 0]
            r_vec = r_vec / (torch.norm(r_vec, p=2) + 1e-12)
            self.refusal_execution_vectors[layer_idx] = r_vec

            # 3. Subspace Orthogonality: Cosine similarity between H and R
            # Orthogonality = 1.0 - abs(<h_vec, r_vec>)
            cos_align = abs(torch.dot(h_vec, r_vec).item())
            self.orthogonality_scores[layer_idx] = float(1.0 - cos_align)

    def diagnose_bypass_mechanism(
        self, layer_idx: int, mutant_activation: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Determines whether an adversarial mutant bypassed safety by:
        A) Semantic Evasion (Failed Harm Detection): Model thought prompt was benign.
        B) Execution Disruption (Suppressed Refusal Execution): Model detected harm but failed to refuse.
        """
        if layer_idx not in self.harm_detection_vectors or layer_idx not in self.refusal_execution_vectors:
            raise KeyError(f"Layer {layer_idx} not fitted.")

        act = mutant_activation.float().squeeze().cpu()
        act_norm = torch.norm(act, p=2) + 1e-12

        h_vec = self.harm_detection_vectors[layer_idx]
        r_vec = self.refusal_execution_vectors[layer_idx]

        harm_score = float(torch.dot(act, h_vec) / act_norm)
        refusal_score = float(torch.dot(act, r_vec) / act_norm)

        # Mechanism Diagnosis
        if harm_score < 0.1 and refusal_score < 0.1:
            mechanism = "Semantic Evasion (Harm Detection Bypass)"
        elif harm_score >= 0.1 and refusal_score < 0.1:
            mechanism = "Gate Actuation Failure (Refusal Execution Disruption)"
        elif harm_score >= 0.1 and refusal_score >= 0.1:
            mechanism = "Standard Refusal Active"
        else:
            mechanism = "Benign Intent Compliant"

        return {
            "layer": layer_idx,
            "harm_detection_projection": harm_score,
            "refusal_execution_projection": refusal_score,
            "subspace_divergence": float(harm_score - refusal_score),
            "mechanism_diagnosis": mechanism,
        }
