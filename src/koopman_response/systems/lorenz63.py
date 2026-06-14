"""Lorenz 63 system with additive noise."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore
        def _wrap(func):
            return func

        return _wrap


@njit(cache=True)
def _integrate_em_lorenz(
    rho: float,
    sigma: float,
    beta: float,
    noise: float,
    t0: float,
    tf: float,
    dt: float,
    tau: int,
    transient: float,
    y0: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    n_steps = int((tf - t0) / dt)
    n_saves = n_steps // tau
    tsave = t0 + np.arange(n_saves) * (tau * dt)
    ysave = np.zeros((n_saves, 3))

    yold = y0.copy()
    index = 0
    for i in range(n_steps):
        x = yold[0]
        yv = yold[1]
        z = yold[2]

        f0 = sigma * (yv - x)
        f1 = x * (rho - z) - yv
        f2 = x * yv - beta * z

        dW0 = np.random.normal(0.0, np.sqrt(dt))
        dW1 = np.random.normal(0.0, np.sqrt(dt))
        dW2 = np.random.normal(0.0, np.sqrt(dt))

        ynew0 = x + f0 * dt + noise * dW0
        ynew1 = yv + f1 * dt + noise * dW1
        ynew2 = z + f2 * dt + noise * dW2

        if i % tau == 0:
            ysave[index, 0] = ynew0
            ysave[index, 1] = ynew1
            ysave[index, 2] = ynew2
            index += 1

        yold[0] = ynew0
        yold[1] = ynew1
        yold[2] = ynew2

    if transient <= t0:
        return tsave, ysave

    start = 0
    while start < tsave.shape[0] and tsave[start] < transient:
        start += 1

    return tsave[start:], ysave[start:, :]


class NoisyLorenz63:
    def __init__(
        self,
        rho: float = 28.0,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        noise: float = 2.0,
    ):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.noise = noise

        self.default_y0 = np.array([1.0, 0.5, 2.0], dtype=float)

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        sigma = self.sigma
        rho = self.rho
        beta = self.beta
        x, yv, z = y
        return np.array([sigma * (yv - x), x * (rho - z) - yv, x * yv - beta * z])

    def diffusion(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        _ = y
        return self.noise * np.eye(3)

    def integrate_em_jit(
        self,
        t_span: tuple[float, float] = (0.0, 1_000_000.0),
        dt: float = 0.001,
        tau: int = 100,
        transient: float = 500.0,
        y0: np.ndarray | None = None,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast Euler-Maruyama integrator for Lorenz63 using numba JIT.
        """
        if not _NUMBA_AVAILABLE:
            raise ImportError("numba is required for integrate_em_jit")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if tau <= 0:
            raise ValueError("tau must be a positive integer")
        if t_span[1] <= t_span[0]:
            raise ValueError("t_span must satisfy t_span[1] > t_span[0]")
        if y0 is None:
            y0 = self.default_y0
        y0 = np.asarray(y0, dtype=float)
        return _integrate_em_lorenz(
            rho=float(self.rho),
            sigma=float(self.sigma),
            beta=float(self.beta),
            noise=float(self.noise),
            t0=float(t_span[0]),
            tf=float(t_span[1]),
            dt=float(dt),
            tau=int(tau),
            transient=float(transient),
            y0=y0,
            seed=int(seed),
        )
