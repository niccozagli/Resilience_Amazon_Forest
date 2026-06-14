from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class ChaoticMap1D:
    """
    1D chaotic map:

        x_{n+1} = [ alpha * x_n - gamma * sin(6 x_n) + Delta * cos(3 x_n) ] mod 2π
    """

    alpha: float = 3.0
    gamma: float = 0.4
    Delta: float = 0.08

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.mod(
            self.alpha * x - self.gamma * np.sin(6 * x) + self.Delta * np.cos(3 * x),
            TWO_PI,
        )

    def iterate(self, x0: float, n_steps: int) -> np.ndarray:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        xs = np.empty(n_steps + 1, dtype=float)
        xs[0] = x0
        x = x0
        for i in range(1, n_steps + 1):
            x = float(self.step(x))
            xs[i] = x
        return xs
