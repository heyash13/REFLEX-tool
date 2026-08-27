"""
Unit Tests for Multi-Armed Bandit Schedulers and Population Managers.
Verifies UCB-1 convergence towards high-reward arms and population ranking.
"""

import pytest
from src.fuzzer.bandit import UCB1Selector, EpsilonGreedySelector, PopulationManager


def test_ucb1_exploitation_behavior():
    operators = ["op_weak", "op_strong"]
    bandit = UCB1Selector(operators, c=1.0)

    # Initial exploration phase
    op1 = bandit.select()
    bandit.update(op1, 0.1)
    op2 = bandit.select()
    bandit.update(op2, 0.9)

    # Over multiple rounds with op_strong yielding high reward, op_strong should dominate selections
    for _ in range(50):
        chosen = bandit.select()
        reward = 1.0 if chosen == "op_strong" else 0.0
        bandit.update(chosen, reward)

    assert bandit.counts["op_strong"] > bandit.counts["op_weak"]


def test_population_manager_capacity():
    pop = PopulationManager(capacity=3)
    pop.add("prompt_low", fitness=0.1)
    pop.add("prompt_med", fitness=0.5)
    pop.add("prompt_high", fitness=0.9)
    pop.add("prompt_highest", fitness=1.5)

    assert len(pop.population) == 3
    best = pop.get_best()
    assert best is not None
    assert best["prompt"] == "prompt_highest"
    assert best["fitness"] == 1.5
