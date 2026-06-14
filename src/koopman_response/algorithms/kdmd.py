from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.kernels import Kernel


def _kernel_matrix(
    kernel: Kernel,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int | None,
    show_progress: bool,
) -> np.ndarray:
    n_rows = X.shape[0]
    n_cols = Y.shape[0]

    sample = kernel(X[:1], Y[:1])
    dtype = sample.dtype
    K = np.empty((n_rows, n_cols), dtype=dtype)

    if batch_size is None or batch_size >= n_rows:
        K[:, :] = kernel(X, Y)
        return K

    iterator = range(0, n_rows, batch_size)
    for start in tqdm(iterator, disable=not show_progress):
        end = min(start + batch_size, n_rows)
        K[start:end, :] = kernel(X[start:end], Y)
    return K


def _iter_trajectory_segments(trajectories: np.ndarray | Sequence[np.ndarray]):
    if isinstance(trajectories, np.ndarray):
        if trajectories.ndim == 2:
            yield trajectories
            return
        if trajectories.ndim == 3:
            for i in range(trajectories.shape[0]):
                yield trajectories[i]
            return
        raise ValueError("trajectories must be a 2D or 3D array")
    for segment in trajectories:
        yield segment


class KernelDMD:
    """
    Kernel Dynamic Mode Decomposition (KDMD).

    This class mirrors EDMD but uses a kernel to avoid explicit feature maps.
    """

    def __init__(
        self,
        kernel: Kernel,
        dt_eff: float | None = None,
        reg: float = 0.0,
        use_pinv: bool = False,
    ):
        if reg < 0:
            raise ValueError("reg must be non-negative")
        self.kernel = kernel
        self.dt_eff = dt_eff
        self.reg = float(reg)
        self.use_pinv = bool(use_pinv)

        self.G: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.reference_data: Optional[np.ndarray] = None

    def fit_snapshots(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int = 5_000,
        show_progress: bool = True,
        fit_kernel: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit KDMD on snapshot pairs (X, Y) and compute Gram matrices.

        Uses the convention:
            G_hat[i, j] = k(X[i], X[j])
            A_hat[i, j] = k(Y[i], X[j])
        """
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if fit_kernel:
            self.kernel.fit(X)

        Kxx = _kernel_matrix(self.kernel, X, X, batch_size, show_progress)
        Kyx = _kernel_matrix(self.kernel, Y, X, batch_size, show_progress)

        G = Kxx #/ Kxx.shape[0]
        A = Kyx #/ Kyx.shape[0]

        self.G = G
        self.A = A
        self.K = None
        self.reference_data = X
        return G, A

    def solve_koopman(
        self,
        reg: float | None = None,
        use_pinv: bool | None = None,
    ) -> np.ndarray:
        """
        Solve for the Koopman matrix K given stored G and A.
        """
        if self.G is None or self.A is None:
            raise ValueError("G and A are not set. Run fit_snapshots() first.")

        G = self.G
        A = self.A
        reg_val = self.reg if reg is None else float(reg)
        use_pinv_val = self.use_pinv if use_pinv is None else bool(use_pinv)

        if reg_val < 0:
            raise ValueError("reg must be non-negative")
        if reg_val > 0.0:
            G = G + reg_val * np.eye(G.shape[0], dtype=G.dtype)

        if use_pinv_val:
            K = np.linalg.pinv(G) @ A
        else:
            K = np.linalg.solve(G, A)

        self.K = K
        return K

    def gram(self) -> np.ndarray:
        if self.G is None:
            raise ValueError("G is not set. Run fit_snapshots() first.")
        return self.G

    def kernel_inner_product_inv_measure(
        self,
        trajectories: np.ndarray | Sequence[np.ndarray],
        reference_data: np.ndarray | None = None,
        batch_size: int = 5_000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Estimate the invariant-measure inner product of kernel functions:

            (G_kernel)_{ij} = (1/M) sum_k k(x_k, x_i)^* k(x_k, x_j)

        where x_k are samples from the invariant measure and x_i are reference points.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        ref = self.reference_data if reference_data is None else reference_data
        if ref is None:
            raise ValueError("reference_data must be provided or set by fit_snapshots()")
        ref_arr = np.asarray(ref, dtype=float)
        if ref_arr.ndim != 2:
            raise ValueError("reference_data must be a 2D array")

        G = None
        n_total = 0
        segments = _iter_trajectory_segments(trajectories)

        for segment in segments:
            seg_arr = np.asarray(segment, dtype=float)
            if seg_arr.ndim != 2:
                raise ValueError("each trajectory segment must be a 2D array")
            n_samples = seg_arr.shape[0]
            iterator = range(0, n_samples, batch_size)
            for start in tqdm(iterator, disable=not show_progress):
                end = min(start + batch_size, n_samples)
                K_eval = self.kernel(seg_arr[start:end], ref_arr)
                if G is None:
                    n_ref = K_eval.shape[1]
                    G = np.zeros((n_ref, n_ref), dtype=K_eval.dtype)
                G += K_eval.conj().T @ K_eval
                n_total += K_eval.shape[0]

        if G is None or n_total == 0:
            raise ValueError("no samples provided in trajectories")
        G /= n_total
        return G
