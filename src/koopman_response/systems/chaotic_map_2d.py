from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoisyChaoticMap2D:
    """
    2D chaotic map with optional additive noise:

        x_{n+1}, y_{n+1} = [ A (x_n, y_n) + (1/pi) * zeta(x_n + y_n) * (1, 1)
                            + sigma * eta_n ] mod 1

    where zeta(s) = arctan(|mu| sin(2π s - alpha) / (1 - |mu| cos(2π s - alpha)))
    and eta_n is standard Gaussian noise in R^2.
    """

    sigma: float = 0.0
    mu_abs: float = 0.88
    alpha: float = -2.4
    zeta_scale: float = 1.0 / np.pi
    A: np.ndarray = field(
        default_factory=lambda: np.array([[2.0, 1.0], [1.0, 1.0]], dtype=float)
    )

    def _zeta(self, s: np.ndarray) -> np.ndarray:
        angle = 2.0 * np.pi * s - self.alpha
        numerator = self.mu_abs * np.sin(angle)
        denominator = 1.0 - self.mu_abs * np.cos(angle)
        return np.arctan(numerator / denominator)

    def _step(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = state[0] + state[1]
        zeta_val = self._zeta(s)
        coupling = self.zeta_scale * zeta_val * np.array([1.0, 1.0])
        noise = self.sigma * rng.standard_normal(2)
        next_state = self.A @ state + coupling + noise
        return np.mod(next_state, 1.0)

    def iterate(
        self, x0: float, y0: float, n_steps: int, seed: int | None = None
    ) -> np.ndarray:
        """
        Iterate the map for n_steps, returning an array of shape (n_steps + 1, 2).
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        rng = np.random.default_rng(seed)
        traj = np.zeros((n_steps + 1, 2), dtype=float)
        traj[0] = np.array([x0, y0], dtype=float)
        for i in range(n_steps):
            traj[i + 1] = self._step(traj[i], rng)
        return traj
