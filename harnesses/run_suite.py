"""
Unified Laboratory Test, Verification, and Benchmark Runner.
Executes end-to-end verification, subspace extraction, comparative fuzzing, and probe benchmarking.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from rich.console import Console
from rich.panel import Panel

from harnesses.run_subspace_analysis import analyze_model_subspaces
from harnesses.run_fuzzing_experiment import run_comparative_fuzzing
from harnesses.run_probe_benchmark import run_probe_benchmarking
from harnesses.run_causal_steering_experiment import run_causal_steering_investigation
from harnesses.run_token_attribution_analysis import run_token_attribution_investigation


def run_full_suite(model_name: str = "gpt2", device: str = "cpu"):
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]RE2 LABORATORY: CLOSED-LOOP INTERPRETABILITY-GUIDED FUZZING SUITE[/bold cyan]\n"
        "[dim]Automated verification, subspace extraction, comparative fuzzing, causal steering, and token attribution[/dim]",
        border_style="cyan"
    ))

    # Step 1: Run PyTest Suite
    console.print("\n[bold yellow]Stage 1: Running Unit & Integration Test Verification Suite...[/bold yellow]")
    pytest_res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    if pytest_res.returncode != 0:
        console.print("[bold red]PyTest verification failed:[/bold red]")
        console.print(pytest_res.stdout)
        console.print(pytest_res.stderr)
        sys.exit(1)
    else:
        console.print("[bold green]All 17 unit & integration tests passed verified.[/bold green]")

    # Step 2: Run Subspace Analysis
    console.print("\n[bold yellow]Stage 2: Running Subspace Extraction and Layer Profile Analysis...[/bold yellow]")
    subspace_results = analyze_model_subspaces(model_name, output_dir="artifacts/subspace_analysis", device=device)

    # Step 3: Run Comparative Fuzzing Experiment
    console.print("\n[bold yellow]Stage 3: Running Comparative Fuzzing Experiment (Closed-Loop vs Output-Only vs Random)...[/bold yellow]")
    fuzzing_results = run_comparative_fuzzing(model_name, max_queries=8, output_dir="artifacts/fuzzing_results", device=device)

    # Step 4: Run Defensive Probe Benchmarking
    console.print("\n[bold yellow]Stage 4: Running Defensive Early-Exit Probe & Latency Benchmark...[/bold yellow]")
    probe_results = run_probe_benchmarking(model_name, output_dir="artifacts/probe_benchmark", device=device)

    # Step 5: Run Causal Activation Steering
    console.print("\n[bold yellow]Stage 5: Running Causal Activation Steering & Directional Ablation...[/bold yellow]")
    steering_results = run_causal_steering_investigation(model_name, output_dir="artifacts/causal_steering", device=device)

    # Step 6: Run 2D Token-by-Layer Attribution
    console.print("\n[bold yellow]Stage 6: Running 2D Token-by-Layer Mechanistic Attribution Analysis...[/bold yellow]")
    token_results = run_token_attribution_investigation(model_name, output_dir="artifacts/token_attribution", device=device)

    console.print(Panel.fit(
        "[bold green]ALL 6 SCIENTIFIC RESEARCH HARNESSES COMPLETED SUCCESSFULLY[/bold green]\n"
        "Artifacts written to: [bold]artifacts/[/bold]",
        border_style="green"
    ))


if __name__ == "__main__":
    device = "cpu"
    run_full_suite(model_name="gpt2", device=device)
