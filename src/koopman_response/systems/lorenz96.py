"""Lorenz 96 system with additive noise."""

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
def _integrate_em_lorenz96(
    forcing: float,
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
    n_state = y0.shape[0]
    ysave = np.zeros((n_saves, n_state))

    yold = y0.copy()
    ynew = np.empty_like(yold)
    index = 0
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        for j in range(n_state):
            jp1 = j + 1
            if jp1 == n_state:
                jp1 = 0
            jm1 = j - 1
            if jm1 < 0:
                jm1 = n_state - 1
            jm2 = j - 2
            if jm2 < 0:
                jm2 += n_state

            drift = (yold[jp1] - yold[jm2]) * yold[jm1] - yold[j] + forcing
            dW = np.random.normal(0.0, sqrt_dt)
            ynew[j] = yold[j] + drift * dt + noise * dW

        if i % tau == 0:
            ysave[index, :] = ynew
            index += 1

        yold, ynew = ynew, yold

    if transient <= t0:
        return tsave, ysave

    start = 0
    while start < tsave.shape[0] and tsave[start] < transient:
        start += 1

    return tsave[start:], ysave[start:, :]


class NoisyLorenz96:
    def __init__(
        self,
        n_state: int = 40,
        forcing: float = 8.0,
        noise: float = 2.5,
    ):
        if n_state < 4:
            raise ValueError("n_state must be at least 4")
        self.n_state = int(n_state)
        self.forcing = float(forcing)
        self.noise = float(noise)

        default_y0 = self.forcing * np.ones(self.n_state, dtype=float)
        default_y0[0] += 0.01
        self.default_y0 = default_y0

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        n_state = self.n_state
        forcing = self.forcing
        out = np.empty(n_state, dtype=float)
        for j in range(n_state):
            jp1 = (j + 1) % n_state
            jm1 = (j - 1) % n_state
            jm2 = (j - 2) % n_state
            out[j] = (y[jp1] - y[jm2]) * y[jm1] - y[j] + forcing
        return out

    def diffusion(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        _ = y
        return self.noise * np.eye(self.n_state)

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
        Fast Euler-Maruyama integrator for Lorenz96 using numba JIT.
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
        if y0.shape[0] != self.n_state:
            raise ValueError("y0 must have length n_state")
        return _integrate_em_lorenz96(
            forcing=float(self.forcing),
            noise=float(self.noise),
            t0=float(t_span[0]),
            tf=float(t_span[1]),
            dt=float(dt),
            tau=int(tau),
            transient=float(transient),
            y0=y0,
            seed=int(seed),
        )
