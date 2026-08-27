"""
Harness for Causal Activation Steering and Directional Ablation on Layer 19.
Proves causal necessity and sufficiency of refusal directions and benchmarks utility preservation.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, List, Any
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.core.hooks import ActivationHookManager
from src.core.subspace import SubspaceEstimator
from src.core.causal_intervention import ActivationSteeringManager, CausalInterventionEvaluator
from src.data.loader import BenchmarkDataLoader


def run_causal_steering_investigation(
    model_name: str = "gpt2",
    target_layer: Optional[int] = None,
    output_dir: str = "artifacts/causal_steering",
    device: str = "cpu"
):
    console = Console()
    console.print(Panel.fit(
        f"[bold cyan]CAUSAL ACTIVATION STEERING & SURGERY HARNESS[/bold cyan]\n"
        f"[dim]Model: {model_name} | Target Device: {device}[/dim]",
        border_style="cyan"
    ))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    ).eval()

    # 2. Extract refusal direction from contrastive pairs
    hook_mgr = ActivationHookManager(model, token_position=-1, offload_to_cpu=True)
    target_layers = hook_mgr.target_layers
    d_model = model.config.hidden_size
    subspace_est = SubspaceEstimator(target_layers=target_layers, d_model=d_model)

    pairs = BenchmarkDataLoader.get_builtin_dataset()
    harmful_acts = {l: [] for l in target_layers}
    benign_acts = {l: [] for l in target_layers}

    hook_mgr.register()
    try:
        with torch.no_grad():
            for p in pairs:
                in_h = tokenizer(p.harmful_prompt, return_tensors="pt").to(device)
                hook_mgr.clear()
                _ = model(**in_h)
                for l in target_layers:
                    if l in hook_mgr.activations:
                        harmful_acts[l].append(hook_mgr.activations[l])

                in_b = tokenizer(p.benign_prompt, return_tensors="pt").to(device)
                hook_mgr.clear()
                _ = model(**in_b)
                for l in target_layers:
                    if l in hook_mgr.activations:
                        benign_acts[l].append(hook_mgr.activations[l])
    finally:
        hook_mgr.remove()

    h_tensors = {l: torch.cat(harmful_acts[l], dim=0) for l in target_layers if len(harmful_acts[l]) > 0}
    b_tensors = {l: torch.cat(benign_acts[l], dim=0) for l in target_layers if len(benign_acts[l]) > 0}
    subspace_est.fit_from_contrastive_tensors(h_tensors, b_tensors, method="svd")

    # Determine peak layer
    layer_to_steer = target_layer if target_layer is not None else int(len(target_layers) * 0.68)
    r_vec = subspace_est.refusal_directions[layer_to_steer]
    console.print(f"Targeting Causal Steering at [bold green]Layer {layer_to_steer}[/bold green] (Refusal Vector Dim: {d_model}).")

    # 3. Evaluate Causal Mediation Quadrants
    evaluator = CausalInterventionEvaluator(model, tokenizer)
    harmful_test = [p.harmful_prompt for p in pairs[:4]]
    benign_test = [p.benign_prompt for p in pairs[:4]]

    causal_results = evaluator.evaluate_causal_effect(
        target_layer=layer_to_steer,
        refusal_direction=r_vec,
        harmful_prompts=harmful_test,
        benign_prompts=benign_test,
        alpha=1.5,
        max_new_tokens=24
    )

    # Table output
    table = Table(title=f"Causal Intervention Proof (Layer {layer_to_steer})")
    table.add_column("Causal Test Condition", style="bold cyan")
    table.add_column("Refusal Rate", justify="right", style="yellow")
    table.add_column("Theoretical Meaning", style="green")

    table.add_row(
        "1. Baseline Harmful Prompts",
        f"{causal_results['baseline_harmful_refusal_rate']*100:.1f}%",
        "Control: Standard refusal behavior"
    )
    table.add_row(
        "2. Ablated Harmful Prompts (- Proj_R)",
        f"{causal_results['ablated_harmful_refusal_rate']*100:.1f}%",
        "Necessity: Stripping r^(l) forces compliance"
    )
    table.add_row(
        "3. Baseline Benign Prompts",
        f"{causal_results['baseline_benign_false_refusal_rate']*100:.1f}%",
        "Control: Standard helpful behavior"
    )
    table.add_row(
        "4. Induced Benign Prompts (+ alpha * r)",
        f"{causal_results['induced_benign_refusal_rate']*100:.1f}%",
        "Sufficiency: Adding r^(l) forces false refusal"
    )
    console.print(table)

    summary_table = Table(title="Causal Mediation Indices")
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Causal Necessity Score", f"{causal_results['causal_necessity_score']:.4f}")
    summary_table.add_row("Causal Sufficiency Score", f"{causal_results['causal_sufficiency_score']:.4f}")
    summary_table.add_row("Causal Mediation Index", f"{causal_results['causal_mediation_index']:.4f}")
    console.print(summary_table)

    out_file = os.path.join(output_dir, "causal_steering_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(causal_results, f, indent=2)
    console.print(f"[bold green]Causal report saved to:[/bold green] {out_file}")
    return causal_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_causal_steering_investigation(args.model, target_layer=args.layer, device=args.device)
