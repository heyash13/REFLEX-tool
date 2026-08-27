"""
Harness for Defensive Linear Probes and In-Flight Interception Benchmark.
Trains probes on layer representations and measures AUROC, F1, and latency speedup vs autoregressive generation.
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
from src.data.loader import BenchmarkDataLoader
from src.defensive.probes import LinearRefusalProbe, EarlyExitDetector


def run_probe_benchmarking(
    model_name: str,
    output_dir: str = "artifacts/probe_benchmark",
    device: str = "cpu"
):
    console = Console()
    console.print(f"[bold green]Starting Defensive Probe Benchmark on:[/bold green] {model_name}")
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

    # 2. Extract layer activations for training
    hook_mgr = ActivationHookManager(model, token_position=-1, offload_to_cpu=True)
    target_layers = hook_mgr.target_layers
    d_model = model.config.hidden_size

    pairs = BenchmarkDataLoader.get_builtin_dataset()
    console.print(f"Extracting activation vectors across {len(pairs)} contrastive pairs...")

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
                        harmful_acts[l].append(hook_mgr.activations[l].numpy())

                in_b = tokenizer(p.benign_prompt, return_tensors="pt").to(device)
                hook_mgr.clear()
                _ = model(**in_b)
                for l in target_layers:
                    if l in hook_mgr.activations:
                        benign_acts[l].append(hook_mgr.activations[l].numpy())
    finally:
        hook_mgr.remove()

    # 3. Train and evaluate probe per layer
    probe_results = []
    table = Table(title="Layer-Wise Defensive Probe Performance")
    table.add_column("Layer", justify="right", style="cyan")
    table.add_column("AUROC", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="yellow")
    table.add_column("F1-Score", justify="right", style="magenta")

    best_layer = 0
    best_auroc = -1.0
    best_probe = None

    for l in target_layers:
        if len(harmful_acts[l]) == 0 or len(benign_acts[l]) == 0:
            continue
            
        X_harm = np.concatenate(harmful_acts[l], axis=0)
        X_benign = np.concatenate(benign_acts[l], axis=0)
        
        # Split train / test (75% train, 25% test)
        n_harm = len(X_harm)
        n_benign = len(X_benign)
        split_h = max(1, int(0.75 * n_harm))
        split_b = max(1, int(0.75 * n_benign))
        
        X_train = np.vstack([X_benign[:split_b], X_harm[:split_h]])
        y_train = np.array([0] * split_b + [1] * split_h)
        
        X_test = np.vstack([X_benign[split_b:], X_harm[split_h:]])
        y_test = np.array([0] * (n_benign - split_b) + [1] * (n_harm - split_h))
        
        probe = LinearRefusalProbe(layer_idx=l, c_regularization=1.0)
        probe.fit(X_train, y_train)
        
        metrics = probe.evaluate(X_test, y_test)
        table.add_row(
            str(l),
            f"{metrics['auroc']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['f1']:.4f}"
        )
        probe_results.append({"layer": l, **metrics})
        
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_layer = l
            best_probe = probe

    console.print(table)

    # 4. Benchmark latency vs full autoregressive generation
    if best_probe is not None:
        console.print(f"[bold cyan]Benchmarking In-Flight Interception Latency at Layer {best_layer}...[/bold cyan]")
        detector = EarlyExitDetector(best_probe)
        test_prompts = [p.harmful_prompt for p in pairs[:4]]
        lat_metrics = detector.benchmark_latency(model, tokenizer, test_prompts, device=device, max_gen_tokens=32)
        
        lat_table = Table(title="In-Flight Probe vs Generation Latency")
        lat_table.add_column("Metric", style="bold cyan")
        lat_table.add_column("Value", style="green")
        lat_table.add_row("Early Probe Latency (ms)", f"{lat_metrics['avg_probe_latency_ms']:.2f} ms")
        lat_table.add_row("Full Autoregressive Latency (ms)", f"{lat_metrics['avg_full_generation_latency_ms']:.2f} ms")
        lat_table.add_row("Inference Speedup", f"{lat_metrics['speedup_factor']:.2f}x")
        lat_table.add_row("Latency Reduction (%)", f"{lat_metrics['latency_reduction_pct']:.1f}%")
        console.print(lat_table)
    else:
        lat_metrics = {}

    out_data = {
        "probe_performance_by_layer": probe_results,
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "latency_metrics": lat_metrics
    }
    out_file = os.path.join(output_dir, "probe_benchmark_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    console.print(f"[bold green]Probe report written to:[/bold green] {out_file}")
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_probe_benchmarking(args.model, device=args.device)
