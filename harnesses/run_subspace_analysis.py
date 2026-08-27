"""
Harness for Layer-Wise Subspace and Refusal Direction Analysis.
Computes refusal directions across layers, generates layer profiles, and computes concept cone metrics.
"""

import argparse
import json
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from src.core.hooks import ActivationHookManager
from src.core.subspace import SubspaceEstimator
from src.data.loader import BenchmarkDataLoader


def analyze_model_subspaces(
    model_name: str,
    output_dir: str = "artifacts/subspace_analysis",
    k_dim: int = 5,
    device: str = "cpu"
):
    console = Console()
    console.print(f"[bold green]Starting Subspace Analysis on model:[/bold green] {model_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    ).eval()
    
    # 2. Setup hooks on all layers
    hook_mgr = ActivationHookManager(model, token_position=-1, offload_to_cpu=True)
    target_layers = hook_mgr.target_layers
    d_model = model.config.hidden_size
    
    estimator = SubspaceEstimator(target_layers=target_layers, d_model=d_model, k_dim=k_dim)
    
    # 3. Load dataset
    pairs = BenchmarkDataLoader.get_builtin_dataset()
    console.print(f"Loaded [bold]{len(pairs)}[/bold] contrastive prompt pairs.")
    
    harmful_acts = {l: [] for l in target_layers}
    benign_acts = {l: [] for l in target_layers}
    
    # 4. Extract activations
    hook_mgr.register()
    try:
        with torch.no_grad():
            for p in pairs:
                # Forward harmful
                inputs_h = tokenizer(p.harmful_prompt, return_tensors="pt").to(device)
                hook_mgr.clear()
                _ = model(**inputs_h)
                for l in target_layers:
                    if l in hook_mgr.activations:
                        harmful_acts[l].append(hook_mgr.activations[l])
                
                # Forward benign
                inputs_b = tokenizer(p.benign_prompt, return_tensors="pt").to(device)
                hook_mgr.clear()
                _ = model(**inputs_b)
                for l in target_layers:
                    if l in hook_mgr.activations:
                        benign_acts[l].append(hook_mgr.activations[l])
    finally:
        hook_mgr.remove()
        
    # Stack tensors
    h_tensors = {l: torch.cat(harmful_acts[l], dim=0) for l in target_layers if len(harmful_acts[l]) > 0}
    b_tensors = {l: torch.cat(benign_acts[l], dim=0) for l in target_layers if len(benign_acts[l]) > 0}
    
    # 5. Fit SVD Subspace
    estimator.fit_from_contrastive_tensors(h_tensors, b_tensors, method="svd")
    
    # 6. Evaluate layer profile
    table = Table(title=f"Layer-wise Refusal Geometry ({model_name})")
    table.add_column("Layer", justify="right", style="cyan")
    table.add_column("Relative Depth", justify="right", style="magenta")
    table.add_column("Top Singular Val", justify="right", style="green")
    table.add_column("Harm vs Benign Cosine Sep", justify="right", style="yellow")
    
    layer_metrics = []
    for l in target_layers:
        if l in estimator.refusal_directions:
            s_val = float(estimator.singular_values[l][0].item()) if l in estimator.singular_values else 0.0
            
            # Compute mean projection for harmful vs benign
            h_proj = [estimator.compute_1d_projection(l, act) for act in h_tensors[l]]
            b_proj = [estimator.compute_1d_projection(l, act) for act in b_tensors[l]]
            sep = float(np.mean(h_proj) - np.mean(b_proj))
            rel_depth = float(l) / float(len(target_layers))
            
            table.add_row(str(l), f"{rel_depth:.2f}", f"{s_val:.4f}", f"{sep:.4f}")
            layer_metrics.append({
                "layer": l,
                "relative_depth": rel_depth,
                "top_singular_value": s_val,
                "cosine_separation": sep
            })
            
    console.print(table)
    
    out_path = os.path.join(output_dir, "subspace_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(layer_metrics, f, indent=2)
    console.print(f"[bold green]Subspace metrics saved to:[/bold green] {out_path}")
    return layer_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    analyze_model_subspaces(args.model, device=args.device)
