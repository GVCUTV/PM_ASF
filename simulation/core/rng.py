import math
import random
from dataclasses import dataclass


@dataclass
class RNGStreams:
    base_seed: int

    def __post_init__(self) -> None:
        self.arrivals = random.Random(self.base_seed + 1)
        self.services = random.Random(self.base_seed + 2)
        self.routing = random.Random(self.base_seed + 3)
        self.developer = random.Random(self.base_seed + 4)


def exponential(rng: random.Random, rate: float) -> float:
    if rate <= 0:
        raise ValueError("Arrival rate must be positive")
    u = rng.random()
    return -math.log(1.0 - u) / rate


def lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    return rng.lognormvariate(mu, sigma)


def weibull(rng: random.Random, shape: float, scale: float) -> float:
    return rng.weibullvariate(shape, scale)
