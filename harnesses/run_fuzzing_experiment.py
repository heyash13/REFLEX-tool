"""
Harness for Comparative Fuzzing Experiments.
Executes Closed-Loop vs Output-Only vs Random-Search campaigns, logging trajectories and ASR metrics.
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
from src.fuzzer.engine import FuzzingOrchestrator, FuzzingConfig


def run_comparative_fuzzing(
    model_name: str,
    max_queries: int = 15,
    output_dir: str = "artifacts/fuzzing_results",
    device: str = "cpu"
):
    console = Console()
    console.print(f"[bold cyan]Initiating Fuzzing Experiment on:[/bold cyan] {model_name}")
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

    # 2. Extract initial subspace estimates from builtin contrastive pairs
    hook_mgr = ActivationHookManager(model, token_position=-1, offload_to_cpu=True)
    target_layers = hook_mgr.target_layers
    d_model = model.config.hidden_size
    subspace_est = SubspaceEstimator(target_layers=target_layers, d_model=d_model)

    pairs = BenchmarkDataLoader.get_builtin_dataset()
    seed_prompts = BenchmarkDataLoader.get_fuzzing_seeds(pairs)[:4]

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

    # 3. Run comparative campaigns across modes
    modes = ["closed_loop", "output_only", "random_search"]
    experiment_results = {}

    summary_table = Table(title="Fuzzing Experiment Results Summary")
    summary_table.add_column("Search Mode", style="cyan bold")
    summary_table.add_column("ASR (%)", justify="right", style="green")
    summary_table.add_column("Mean QFS", justify="right", style="yellow")
    summary_table.add_column("Successes / Total", justify="right", style="magenta")

    for mode in modes:
        console.print(f"Executing campaign mode: [bold]{mode}[/bold]...")
        cfg = FuzzingConfig(
            mode=mode,
            target_layers=target_layers,
            max_queries=max_queries,
            bandit_type="random" if mode == "random_search" else "ucb1",
            device=device
        )
        orchestrator = FuzzingOrchestrator(
            model=model,
            tokenizer=tokenizer,
            subspace_estimator=subspace_est,
            config=cfg
        )

        mode_runs = []
        qfs_list = []
        successes = 0

        for seed in seed_prompts:
            run_res = orchestrator.run_fuzzing_loop(seed)
            mode_runs.append(run_res)
            if run_res["status"] in ("SUCCESS", "SUCCESS_INITIAL"):
                successes += 1
                qfs_list.append(run_res["queries_to_success"])
            else:
                qfs_list.append(max_queries)

        asr = (successes / len(seed_prompts)) * 100.0
        mean_qfs = float(np.mean(qfs_list))
        
        summary_table.add_row(
            mode,
            f"{asr:.1f}%",
            f"{mean_qfs:.2f}",
            f"{successes}/{len(seed_prompts)}"
        )

        experiment_results[mode] = {
            "asr_pct": asr,
            "mean_qfs": mean_qfs,
            "total_prompts": len(seed_prompts),
            "successes": successes,
            "runs": mode_runs
        }

    console.print(summary_table)

    out_file = os.path.join(output_dir, "comparative_fuzzing_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    console.print(f"[bold green]Full report written to:[/bold green] {out_file}")
    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_comparative_fuzzing(args.model, max_queries=args.queries, device=args.device)
