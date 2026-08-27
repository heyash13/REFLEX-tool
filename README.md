# REFLEX: REFusal Localization and EXploration Engine

## Closed-Loop, Interpretability-Guided Fuzzing and In-Flight Defense for LLM Refusal Mechanisms

REFLEX is a research laboratory and experimentation testbed designed to couple transformer mechanistic interpretability (layer-wise residual stream activations, SVD contrastive directions, concept cones) with discrete adversarial fuzzing (multi-armed bandit mutation scheduling).

---

## 1. Quickstart & Installation

```bash
# Initialize virtual environment with Python 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

---

## 2. Running Verification & Experiments

### Execute the Unified Test & Benchmark Suite
```bash
.venv/bin/python harnesses/run_suite.py
```

### Run Unit & Integration Tests
```bash
.venv/bin/pytest tests/ -v
```

### Extract Subspace Profiles Across Layers
```bash
.venv/bin/python harnesses/run_subspace_analysis.py --model gpt2
```

### Run Comparative Fuzzing (Closed-Loop vs Output-Only vs Random)
```bash
.venv/bin/python harnesses/run_fuzzing_experiment.py --model gpt2 --queries 15
```

### Benchmark Defensive In-Flight Probes & Latency
```bash
.venv/bin/python harnesses/run_probe_benchmark.py --model gpt2
```

---

## 3. Architecture Overview

- `src/core/hooks.py`: Zero-leak forward hook manager with CPU detachment.
- `src/core/subspace.py`: Contrastive difference-in-means, SVD refusal directions, concept cones, and Grassmannian distances.
- `src/core/evaluator.py`: Standard prefix refusal classification and multi-objective fitness aggregation.
- `src/fuzzer/mutators.py`: 8 discrete mutation operators (lexical, syntactic, hypothetical, cipher, and composition).
- `src/fuzzer/bandit.py`: UCB-1, Epsilon-Greedy, and Population Manager for mutation scheduling.
- `src/fuzzer/engine.py`: Closed-loop, output-only, and random search orchestrator.
- `src/defensive/probes.py`: In-flight early-warning linear probes and inference latency profiler.
- `src/defensive/ablation.py`: Ablation study runner across signal regimes and layer depths.
- `src/data/loader.py`: Benchmark dataset loaders (JailbreakBench, HarmBench schemas).

---

## 4. Key Experimental Invariants

- **Multi-Objective Fitness**:
  `Fitness(x') = lambda_1 * S_bypass(x') + lambda_2 * S_intent(x, x') - lambda_3 * S_ppl(x') + lambda_4 * I_comply(x')`
- **UCB-1 Selection Score**:
  `Score_t(m) = Q_bar_t(m) + c * sqrt((2 * ln(t)) / N_t(m))`
- **Grassmannian Distance**:
  `delta_Grassmann(R_a, R_b) = sqrt(sum_{i=1}^k sin^2(theta_i))`
