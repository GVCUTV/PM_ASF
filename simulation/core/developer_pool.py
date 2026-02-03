import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import Stage


@dataclass
class Developer:
    state: str
    next_transition_time: float


class DeveloperPool:
    def __init__(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        stint_pmf: Dict[str, List[Tuple[float, float]]],
        developer_count: int,
        rng: random.Random,
    ) -> None:
        self.transition_matrix = transition_matrix
        self.stint_pmf = stint_pmf
        self.developer_count = developer_count
        self.rng = rng
        self.developers: List[Developer] = []
        self._initialize_developers()

    def _initialize_developers(self) -> None:
        stationary = self._stationary_distribution()
        states, weights = zip(*stationary.items())
        for _ in range(self.developer_count):
            state = self._weighted_choice(states, weights)
            stint = self._sample_stint(state)
            self.developers.append(Developer(state=state, next_transition_time=stint))

    def _stationary_distribution(self) -> Dict[str, float]:
        states = list(self.transition_matrix.keys())
        probs = {state: 1.0 / len(states) for state in states}
        for _ in range(1000):
            new_probs = {state: 0.0 for state in states}
            for from_state, row in self.transition_matrix.items():
                for to_state, prob in row.items():
                    new_probs[to_state] += probs[from_state] * prob
            diff = sum(abs(new_probs[state] - probs[state]) for state in states)
            probs = new_probs
            if diff < 1e-8:
                break
        total = sum(probs.values())
        return {state: prob / total for state, prob in probs.items()}

    def _weighted_choice(self, states: Tuple[str, ...], weights: Tuple[float, ...]) -> str:
        r = self.rng.random()
        cumulative = 0.0
        for state, weight in zip(states, weights):
            cumulative += weight
            if r <= cumulative:
                return state
        return states[-1]

    def _sample_stint(self, state: str) -> float:
        pmf = self.stint_pmf.get(state)
        if not pmf:
            raise ValueError(f"Missing stint PMF for state {state}")
        lengths, weights = zip(*pmf)
        r = self.rng.random()
        cumulative = 0.0
        for length, weight in zip(lengths, weights):
            cumulative += weight
            if r <= cumulative:
                return float(length)
        return float(lengths[-1])

    def update(self, current_time: float) -> None:
        for developer in self.developers:
            while developer.next_transition_time <= current_time:
                developer.state = self._next_state(developer.state)
                stint = self._sample_stint(developer.state)
                developer.next_transition_time += stint

    def _next_state(self, state: str) -> str:
        row = self.transition_matrix[state]
        states = tuple(row.keys())
        weights = tuple(row.values())
        return self._weighted_choice(states, weights)

    def next_transition_time(self) -> float:
        return min(dev.next_transition_time for dev in self.developers)

    def capacity_counts(self) -> Dict[Stage, int]:
        counts = {Stage.DEV: 0, Stage.REVIEW: 0, Stage.TESTING: 0}
        for developer in self.developers:
            if developer.state == "DEV":
                counts[Stage.DEV] += 1
            elif developer.state == "REV":
                counts[Stage.REVIEW] += 1
            elif developer.state == "TEST":
                counts[Stage.TESTING] += 1
        return counts
