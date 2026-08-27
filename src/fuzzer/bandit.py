"""
Multi-Armed Bandit and Population Search Schedulers.
Implements UCB-1, Epsilon-Greedy, Thompson Sampling, and Genetic Population Managers.
"""

from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np


class BanditSelector:
    """Base class for mutation operator selection algorithms."""

    def select(self) -> str:
        raise NotImplementedError

    def update(self, operator: str, reward: float):
        raise NotImplementedError


class UCB1Selector(BanditSelector):
    """
    Upper Confidence Bound (UCB-1) operator selector.
    Balances exploitation (mean reward Q_bar) with exploration (c * sqrt(2 * ln(t) / N_t)).
    """

    def __init__(self, operator_names: List[str], c: float = 1.414):
        self.operator_names = list(operator_names)
        self.c = c
        self.counts: Dict[str, int] = {op: 0 for op in self.operator_names}
        self.total_rewards: Dict[str, float] = {op: 0.0 for op in self.operator_names}
        self.total_pulls: int = 0

    def select(self) -> str:
        # Initial exploration: pull each arm at least once
        for op in self.operator_names:
            if self.counts[op] == 0:
                return op

        best_score = -float("inf")
        best_op = self.operator_names[0]

        for op in self.operator_names:
            n_t = self.counts[op]
            mean_q = self.total_rewards[op] / n_t
            confidence_bound = self.c * math.sqrt((2.0 * math.log(self.total_pulls)) / n_t)
            score = mean_q + confidence_bound
            
            if score > best_score:
                best_score = score
                best_op = op

        return best_op

    def update(self, operator: str, reward: float):
        if operator in self.counts:
            self.counts[operator] += 1
            self.total_rewards[operator] += reward
            self.total_pulls += 1


class EpsilonGreedySelector(BanditSelector):
    """Epsilon-Greedy selection with optional epsilon decay."""

    def __init__(self, operator_names: List[str], epsilon: float = 0.2, decay: float = 0.99):
        self.operator_names = list(operator_names)
        self.epsilon = epsilon
        self.decay = decay
        self.counts: Dict[str, int] = {op: 0 for op in self.operator_names}
        self.total_rewards: Dict[str, float] = {op: 0.0 for op in self.operator_names}

    def select(self) -> str:
        if random.random() < self.epsilon or all(self.counts[op] == 0 for op in self.operator_names):
            return random.choice(self.operator_names)
        
        # Greedy choice based on empirical mean
        best_mean = -float("inf")
        best_op = self.operator_names[0]
        for op in self.operator_names:
            if self.counts[op] > 0:
                mean_q = self.total_rewards[op] / self.counts[op]
                if mean_q > best_mean:
                    best_mean = mean_q
                    best_op = op
        return best_op

    def update(self, operator: str, reward: float):
        if operator in self.counts:
            self.counts[operator] += 1
            self.total_rewards[operator] += reward
            self.epsilon = max(0.01, self.epsilon * self.decay)


class RandomSelector(BanditSelector):
    """Uniform random baseline selector."""

    def __init__(self, operator_names: List[str]):
        self.operator_names = list(operator_names)

    def select(self) -> str:
        return random.choice(self.operator_names)

    def update(self, operator: str, reward: float):
        pass


class PopulationManager:
    """Maintains a population of elite adversarial candidates for genetic crossover/mutation."""

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.population: List[Dict] = []

    def add(self, prompt: str, fitness: float, metadata: Optional[Dict] = None):
        entry = {
            "prompt": prompt,
            "fitness": fitness,
            "metadata": metadata or {}
        }
        self.population.append(entry)
        # Keep top performers sorted by fitness descending
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        if len(self.population) > self.capacity:
            self.population = self.population[:self.capacity]

    def sample_parent(self) -> str:
        if not self.population:
            raise ValueError("Population is empty.")
        # Fitness-proportional or tournament selection
        weights = [max(0.01, item["fitness"]) for item in self.population]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen = np.random.choice(len(self.population), p=probs)
        return self.population[chosen]["prompt"]

    def get_best(self) -> Optional[Dict]:
        return self.population[0] if self.population else None
