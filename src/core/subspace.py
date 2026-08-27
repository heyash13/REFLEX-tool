"""
Subspace and Refusal Direction Geometry Analysis.
Calculates 1D refusal vectors, k-dimensional concept cones, and Grassmannian distances.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from scipy.linalg import subspace_angles


class SubspaceEstimator:
    """
    Computes and analyzes refusal geometries across transformer layers.
    Includes 1D vector extraction, k-D concept cone projection, and Grassmannian distance metrics.
    """

    def __init__(self, target_layers: List[int], d_model: int, k_dim: int = 5):
        self.target_layers = target_layers
        self.d_model = d_model
        self.k_dim = k_dim
        
        # Primary 1D unit refusal directions per layer: (d_model,)
        self.refusal_directions: Dict[int, torch.Tensor] = {}
        # Top-k basis matrices per layer: (d_model, k)
        self.subspace_bases: Dict[int, torch.Tensor] = {}
        # Singular values per layer: (min(d_model, N),)
        self.singular_values: Dict[int, torch.Tensor] = {}

    def fit_from_contrastive_tensors(
        self,
        harmful_acts: Dict[int, torch.Tensor],
        benign_acts: Dict[int, torch.Tensor],
        method: str = "svd"
    ):
        """
        Estimates refusal directions and subspaces from paired harmful and benign activations.
        harmful_acts[l]: Tensor of shape (N, d_model)
        benign_acts[l]: Tensor of shape (N, d_model)
        """
        for layer_idx in self.target_layers:
            if layer_idx not in harmful_acts or layer_idx not in benign_acts:
                continue
                
            h_harm = harmful_acts[layer_idx].float().cpu()
            h_benign = benign_acts[layer_idx].float().cpu()
            
            # Ensure equal sample count
            min_n = min(h_harm.size(0), h_benign.size(0))
            h_harm = h_harm[:min_n]
            h_benign = h_benign[:min_n]
            
            # Difference matrix Delta_H: (d_model, N)
            diff_matrix = (h_harm - h_benign).T
            
            if method == "mean_diff":
                # Direct difference of means: mean(harm) - mean(benign)
                mean_diff = (h_harm.mean(dim=0) - h_benign.mean(dim=0))
                r_vec = mean_diff / (torch.norm(mean_diff, p=2) + 1e-12)
                self.refusal_directions[layer_idx] = r_vec
                self.subspace_bases[layer_idx] = r_vec.unsqueeze(1)
                self.singular_values[layer_idx] = torch.tensor([1.0])
            else:
                # SVD on centered difference matrix
                U, S, Vh = torch.linalg.svd(diff_matrix, full_matrices=False)
                
                # Primary 1D direction is top left-singular vector
                r_vec = U[:, 0]
                r_vec = r_vec / (torch.norm(r_vec, p=2) + 1e-12)
                self.refusal_directions[layer_idx] = r_vec
                
                # Top-k basis vectors
                k = min(self.k_dim, U.size(1))
                self.subspace_bases[layer_idx] = U[:, :k]
                self.singular_values[layer_idx] = S

    def compute_1d_projection(self, layer_idx: int, activation: torch.Tensor) -> float:
        """
        Computes cosine similarity between token activation and 1D refusal direction.
        Value in [-1.0, 1.0]. High positive values indicate strong refusal alignment.
        """
        if layer_idx not in self.refusal_directions:
            return 0.0
            
        r_vec = self.refusal_directions[layer_idx].to(activation.device)
        act = activation.float().squeeze()
        act_norm = torch.norm(act, p=2)
        
        if act_norm < 1e-12:
            return 0.0
            
        projection = torch.dot(act, r_vec) / act_norm
        return projection.item()

    def compute_cone_distance(self, layer_idx: int, activation: torch.Tensor) -> float:
        """
        Computes non-negative projection distance onto the concept cone spanned by top-k bases.
        Approximated via Non-Negative Least Squares (NNLS).
        """
        if layer_idx not in self.subspace_bases:
            return 0.0
            
        basis = self.subspace_bases[layer_idx].float().cpu().numpy()  # (d_model, k)
        act = activation.float().squeeze().cpu().numpy()              # (d_model,)
        
        # Coordinate projection
        coeffs = np.linalg.lstsq(basis, act, rcond=None)[0]
        # Rectify to enforce non-negative cone membership
        coeffs_nonneg = np.maximum(coeffs, 0.0)
        proj_point = basis @ coeffs_nonneg
        
        # Normalized distance
        dist = np.linalg.norm(act - proj_point) / (np.linalg.norm(act) + 1e-12)
        return float(dist)

    def compute_grassmann_distance(self, layer_idx: int, comparison_basis: torch.Tensor) -> float:
        """
        Calculates Grassmannian distance between base refusal subspace and comparison subspace.
        Grassmann distance: sqrt(sum(sin^2(theta_i))) over principal subspace angles theta_i.
        """
        if layer_idx not in self.subspace_bases:
            return 0.0
            
        basis_a = self.subspace_bases[layer_idx].float().cpu().numpy()
        basis_b = comparison_basis.float().cpu().numpy()
        
        # Compute principal angles via scipy SVD of basis_a.T @ basis_b
        angles = subspace_angles(basis_a, basis_b)
        grassmann_dist = np.sqrt(np.sum(np.sin(angles) ** 2))
        return float(grassmann_dist)

    def locate_peak_refusal_layer(self, activations: Dict[int, torch.Tensor]) -> Tuple[int, float]:
        """Identifies the layer index with the highest cosine alignment to refusal direction."""
        best_layer = -1
        max_proj = -float("inf")
        
        for layer_idx, act in activations.items():
            proj = self.compute_1d_projection(layer_idx, act)
            if proj > max_proj:
                max_proj = proj
                best_layer = layer_idx
                
        return best_layer, max_proj
