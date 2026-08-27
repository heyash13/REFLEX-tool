"""
Token-Level Mechanistic Attribution and 2D Layer-by-Token Heatmap Generator.
Identifies the precise token positions where the refusal representation crystallizes.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from transformers import PreTrainedTokenizer


class TokenAttributionAnalyzer:
    """
    Computes 2D mechanistic refusal attribution across (Token Sequence Length x Transformer Layers).
    S(t, l) = <h_t^(l), r^(l)> / (||h_t^(l)||_2 * ||r^(l)||_2)
    """

    def __init__(self, refusal_directions: Dict[int, torch.Tensor], tokenizer: PreTrainedTokenizer):
        self.refusal_directions = refusal_directions
        self.tokenizer = tokenizer

    def compute_2d_attribution_matrix(
        self,
        full_sequence_activations: Dict[int, torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Calculates the 2D attribution matrix across all tokens and layers.
        full_sequence_activations[l]: shape (1, seq_len, d_model)
        input_ids: shape (1, seq_len)
        """
        seq_len = input_ids.shape[1]
        raw_ids = input_ids[0].tolist()
        tokens = [self.tokenizer.decode([int(tid)]) for tid in raw_ids]
        layers = sorted(list(full_sequence_activations.keys()))
        
        matrix = np.zeros((len(layers), seq_len), dtype=np.float32)
        
        for row_idx, l in enumerate(layers):
            if l not in self.refusal_directions:
                continue
            r_vec = self.refusal_directions[l].float().cpu()
            r_norm = torch.norm(r_vec, p=2).item() + 1e-12
            
            act_seq = full_sequence_activations[l][0].float().cpu()  # (seq_len, d_model)
            
            for t in range(seq_len):
                h_t = act_seq[t]
                h_norm = torch.norm(h_t, p=2).item() + 1e-12
                proj = torch.dot(h_t, r_vec).item() / (h_norm * r_norm)
                matrix[row_idx, t] = proj

        # Identify top salient tokens
        token_saliency = np.mean(matrix, axis=0)  # average across layers
        top_token_indices = np.argsort(token_saliency)[::-1][:5]
        
        salient_tokens = [
            {
                "position": int(idx),
                "token": tokens[idx],
                "mean_attribution": float(token_saliency[idx]),
                "peak_layer": int(layers[np.argmax(matrix[:, idx])]),
                "peak_attribution": float(np.max(matrix[:, idx])),
            }
            for idx in top_token_indices
        ]

        return {
            "tokens": tokens,
            "layers": layers,
            "attribution_matrix": matrix.tolist(),
            "salient_tokens": salient_tokens,
            "global_peak_token": salient_tokens[0] if salient_tokens else None,
        }
