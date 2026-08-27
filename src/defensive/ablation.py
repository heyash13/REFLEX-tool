"""
Ablation Study Suite for Closed-Loop Interpretability-Guided Fuzzing.
Runs systematic ablations across feedback signals, layer depths, and subspace dimensionalities.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ..fuzzer.engine import FuzzingOrchestrator, FuzzingConfig


class AblationSuite:
    """
    Executes controlled ablation experiments to isolate the causal impact of:
    1. Feedback signal: Closed-loop (Activation+Output) vs Output-Only vs Activation-Only vs Random
    2. Target layer depth: Early vs Middle vs Late transformer blocks
    3. Subspace dimensionality: 1D vector vs k-D concept cone
    """

    def __init__(self, model: Any, tokenizer: Any, subspace_estimator: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.subspace_est = subspace_estimator

    def run_signal_ablation(
        self,
        seed_prompts: List[str],
        max_queries: int = 30,
        target_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compares search trajectories across 4 signal regimes holding seed corpus and operators constant.
        """
        modes = ["closed_loop", "output_only", "activation_only", "random_search"]
        results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
        summary: Dict[str, Dict[str, float]] = {}

        layers = target_layers or self.subspace_est.target_layers

        for mode in modes:
            cfg = FuzzingConfig(
                mode=mode,
                target_layers=layers,
                max_queries=max_queries,
                bandit_type="random" if mode == "random_search" else "ucb1",
            )
            orchestrator = FuzzingOrchestrator(
                model=self.model,
                tokenizer=self.tokenizer,
                subspace_estimator=self.subspace_est,
                config=cfg,
            )

            qfs_list = []
            success_count = 0

            for seed in seed_prompts:
                res = orchestrator.run_fuzzing_loop(seed)
                results[mode].append(res)
                if res["status"] in ("SUCCESS", "SUCCESS_INITIAL"):
                    success_count += 1
                    qfs_list.append(res["queries_to_success"])
                else:
                    qfs_list.append(max_queries)

            asr = (success_count / len(seed_prompts)) * 100.0 if seed_prompts else 0.0
            mean_qfs = float(np.mean(qfs_list)) if qfs_list else 0.0
            median_qfs = float(np.median(qfs_list)) if qfs_list else 0.0

            summary[mode] = {
                "asr_pct": asr,
                "mean_queries_to_success": mean_qfs,
                "median_queries_to_success": median_qfs,
                "total_prompts": len(seed_prompts),
                "successful_prompts": success_count,
            }

        return {
            "summary": summary,
            "raw_results": results,
        }
