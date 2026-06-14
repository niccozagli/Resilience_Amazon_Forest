from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from tqdm import tqdm

from koopman_response.algorithms.dictionaries import Dictionary
from koopman_response.algorithms.kernels import Kernel
from koopman_response.utils.koopman import get_spectral_properties


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


def _normalize_active_dims(
    X_values: np.ndarray
    | dict[int | str, np.ndarray | float | int | None]
    | Sequence[np.ndarray | float | int | None],
    n_samples: int,
    dim: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Normalize X_values into a list of (dim_index, values) pairs, skipping zeros.

    Accepted formats:
    - ndarray shape (n_samples, dim)
    - dict {dim_index: values} (indices or 'x','y','z' for dim>=3)
    - sequence length dim with entries as arrays, scalars, or None
    """
    active: list[tuple[int, np.ndarray]] = []

    if isinstance(X_values, dict):
        items = []
        for k, v in X_values.items():
            if isinstance(k, str):
                if k.isdigit():
                    d = int(k)
                elif k.lower() in {"x", "y", "z"} and dim >= 3:
                    d = {"x": 0, "y": 1, "z": 2}[k.lower()]
                else:
                    raise ValueError(
                        "X_values dict keys must be int indices or one of 'x','y','z' for dim>=3"
                    )
            else:
                d = int(k)
            items.append((d, v))
    elif isinstance(X_values, (list, tuple)):
        if len(X_values) != dim:
            raise ValueError("X_values sequence must have length dim")
        items = list(enumerate(X_values))
    else:
        X_arr = np.asarray(X_values)
        if X_arr.ndim != 2 or X_arr.shape != (n_samples, dim):
            raise ValueError("X_values must have shape (n_samples, dim)")
        for d in range(dim):
            col = X_arr[:, d]
            if np.any(col != 0):
                active.append((d, col))
        return active

    for d, v in items:
        if d < 0 or d >= dim:
            raise ValueError(f"dimension index {d} is out of bounds for dim={dim}")
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 0:
            if float(arr) == 0.0:
                continue
            arr = np.full(n_samples, float(arr))
        elif arr.shape == (n_samples, 1):
            arr = arr[:, 0]
        elif arr.shape != (n_samples,):
            raise ValueError(
                "Each X_values entry must be a scalar or have shape (n_samples,) or (n_samples, 1)"
            )
        if np.any(arr != 0):
            active.append((d, arr))

    return active


@dataclass
class KoopmanSpectrumEDMD:
    """
    Container for Koopman spectral objects and eigenfunction evaluation.
    """

    K: np.ndarray
    dictionary: Dictionary
    eigenvalues: np.ndarray
    right_eigvecs: np.ndarray
    left_eigvecs: np.ndarray

    @classmethod
    def from_koopman_matrix(
        cls, K: np.ndarray, dictionary: Dictionary
    ) -> "KoopmanSpectrumEDMD":
        eigenvalues, right_eigvecs, left_eigvecs = get_spectral_properties(K)
        return cls(
            K=K,
            dictionary=dictionary,
            eigenvalues=eigenvalues,
            right_eigvecs=right_eigvecs,
            left_eigvecs=left_eigvecs,
        )

    def evaluate_eigenfunctions(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate all Koopman eigenfunctions at a batch of points.

        Returns shape (n_samples, n_eig).
        """
        phi = self.dictionary.evaluate_batch(data)
        return phi @ self.right_eigvecs

    def continuous_time_eigenvalues(self, dt_eff: float) -> np.ndarray:
        """
        Convert discrete-time eigenvalues to continuous-time using dt_eff.
        """
        return np.log(self.eigenvalues) / dt_eff

    def eigenfunction(self, i: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable eigenfunction phi_i(x).
        """
        if i < 0 or i >= self.right_eigvecs.shape[1]:
            raise IndexError("eigenfunction index out of bounds")
        v = self.right_eigvecs[:, i]

        def _phi(x: np.ndarray) -> np.ndarray:
            return self.dictionary.evaluate(x) @ v

        return _phi

    def eigenfunction_inner_product(self, G: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix of Koopman eigenfunctions:

            Xi^* G Xi

        where Xi is the matrix of right eigenvectors (columns), and G is the
        EDMD Gram matrix.
        """
        return self.right_eigvecs.conj().T @ G @ self.right_eigvecs

    def psi_inner(self, data: np.ndarray, f_values: np.ndarray) -> np.ndarray:
        """
        Compute <psi, f>_0 from trajectory samples:

            <psi, f>_0 = (1/N) * Phi^* f

        where Phi is the dictionary evaluation on data.
        """
        phi = self.dictionary.evaluate_batch(data)
        f_vals = np.asarray(f_values)
        if f_vals.ndim == 1:
            f_vals = f_vals[:, None]
        if f_vals.shape[0] != phi.shape[0]:
            raise ValueError("f_values and data must have matching first dimension")
        inner = (phi.conj().T @ f_vals) / phi.shape[0]
        if inner.ndim == 2 and inner.shape[1] == 1:
            return inner[:, 0]
        return inner

    def phi_inner(self, G: np.ndarray, psi_inner: np.ndarray) -> np.ndarray:
        """
        Compute <phi, f>_0 from <psi, f>_0:

            <phi, f>_0 = Xi^* <psi, f>_0
        """
        return self.right_eigvecs.conj().T @ psi_inner

    def best_coefficients(self, G: np.ndarray, psi_inner: np.ndarray) -> np.ndarray:
        """
        Compute coefficients for the best decomposition in the Koopman basis:

            f_hat = G_phi^+ <phi, f>_0
                  = (Xi^* G Xi)^+ (Xi^* <psi, f>_0)
        """
        G_phi = self.eigenfunction_inner_product(G)
        phi_inner = self.phi_inner(G, psi_inner)
        coeffs = np.linalg.pinv(G_phi) @ phi_inner
        if coeffs.ndim == 2 and coeffs.shape[1] == 1:
            return coeffs[:, 0]
        return coeffs

    def correlation_function_discrete(
        self,
        G_phi: np.ndarray,
        coeff_f: np.ndarray,
        coeff_g: np.ndarray,
        eigenvalues: np.ndarray | None = None,
    ):
        """
        Return a callable C_fg(k) using Koopman eigenfunctions (discrete time):

            C_fg(k) = coeff_g^* @ G_phi @ (coeff_f * lambda^k)

        where G_phi is the Gram matrix of Koopman eigenfunctions and lambda are
        the eigenvalues used in the power. If eigenvalues is None, self.eigenvalues
        are used.
        """
        eigs = self.eigenvalues if eigenvalues is None else eigenvalues

        coeff_f = np.asarray(coeff_f).reshape(-1)
        coeff_g = np.asarray(coeff_g).reshape(-1)
        eigs = np.asarray(eigs).reshape(-1)

        # drop the static mode
        coeff_f = coeff_f[1:]
        coeff_g = coeff_g[1:]
        eigs = eigs[1:]
        G_phi = G_phi[1:, 1:]
        row = coeff_g.conj() @ G_phi

        def _corr(k):
            if np.isscalar(k):
                return row @ (coeff_f * (eigs ** k))
            k_arr = np.asarray(k)
            pow_k = np.power(eigs, k_arr[:, None])
            return (pow_k * coeff_f[None, :]) @ row

        return _corr

    def correlation_function_continuous(
        self,
        G_phi: np.ndarray,
        coeff_f: np.ndarray,
        coeff_g: np.ndarray,
        eigenvalues: np.ndarray | None = None,
    ):
        """
        Return a callable C_fg(t) using Koopman eigenfunctions (continuous time):

            C_fg(t) = coeff_g^* @ G_phi @ (coeff_f * exp(t * lambda))

        where G_phi is the Gram matrix of Koopman eigenfunctions and lambda are
        continuous-time eigenvalues used in the exponential. If eigenvalues is None,
        self.eigenvalues are used.
        """
        eigs = self.eigenvalues if eigenvalues is None else eigenvalues

        coeff_f = np.asarray(coeff_f).reshape(-1)
        coeff_g = np.asarray(coeff_g).reshape(-1)
        eigs = np.asarray(eigs).reshape(-1)

        # drop the static mode
        coeff_f = coeff_f[1:]
        coeff_g = coeff_g[1:]
        eigs = eigs[1:]
        G_phi = G_phi[1:, 1:]
        row = coeff_g.conj() @ G_phi

        def _corr(t):
            if np.isscalar(t):
                return row @ (coeff_f * np.exp(t * eigs))
            t_arr = np.asarray(t)
            exp_t = np.exp(np.outer(t_arr, eigs))
            return (exp_t * coeff_f[None, :]) @ row

        return _corr


@dataclass
class KoopmanSpectrumKDMD:
    """
    Container for Koopman spectral objects in KDMD (no explicit dictionary).
    """

    K: np.ndarray
    eigenvalues: np.ndarray
    right_eigvecs: np.ndarray
    left_eigvecs: np.ndarray
    kernel: Kernel | None = None
    reference_data: np.ndarray | None = None
    U_r: np.ndarray | None = None
    S_r: np.ndarray | None = None

    @classmethod
    def from_koopman_matrix(
        cls,
        K: np.ndarray,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> "KoopmanSpectrumKDMD":
        eigenvalues, right_eigvecs, left_eigvecs = get_spectral_properties(K)
        return cls(
            K=K,
            eigenvalues=eigenvalues,
            right_eigvecs=right_eigvecs,
            left_eigvecs=left_eigvecs,
            kernel=kernel,
            reference_data=reference_data,
            U_r=U_r,
            S_r=S_r,
        )

    def continuous_time_eigenvalues(self, dt_eff: float) -> np.ndarray:
        """
        Convert discrete-time eigenvalues to continuous-time using dt_eff.
        """
        return np.log(self.eigenvalues) / dt_eff

    def _resolve_kdmd_params(
        self,
        kernel: Kernel | None,
        reference_data: np.ndarray | None,
        U_r: np.ndarray | None,
        S_r: np.ndarray | None,
    ) -> tuple[Kernel, np.ndarray, np.ndarray, np.ndarray]:
        kernel_val = self.kernel if kernel is None else kernel
        ref_val = self.reference_data if reference_data is None else reference_data
        U_r_val = self.U_r if U_r is None else U_r
        S_r_val = self.S_r if S_r is None else S_r
        if kernel_val is None:
            raise ValueError("kernel must be provided or stored on KoopmanSpectrumKDMD")
        if ref_val is None:
            raise ValueError("reference_data must be provided or stored on KoopmanSpectrumKDMD")
        if U_r_val is None or S_r_val is None:
            raise ValueError("U_r and S_r must be provided or stored on KoopmanSpectrumKDMD")
        return kernel_val, np.asarray(ref_val), U_r_val, S_r_val

    def _resolve_reduction_params(
        self,
        U_r: np.ndarray | None,
        S_r: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        U_r_val = self.U_r if U_r is None else U_r
        S_r_val = self.S_r if S_r is None else S_r
        if U_r_val is None or S_r_val is None:
            raise ValueError("U_r and S_r must be provided or stored on KoopmanSpectrumKDMD")
        return U_r_val, S_r_val

    def koopman_modes(
        self,
        observable: np.ndarray,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute Koopman mode coefficients for an observable in KDMD.

        Formula:
            a = W^* S_r^{-1/2} U_r^* f
        where W are left eigenvectors of K_r, and f is the observable sampled
        at the reference (training) points.
        """
        U_r_val, S_r_val = self._resolve_reduction_params(U_r, S_r)
        obs = np.asarray(observable).reshape(-1)
        if obs.shape[0] != U_r_val.shape[0]:
            raise ValueError("observable length must match number of reference points")
        proj = (U_r_val.conj().T @ obs) / np.sqrt(S_r_val)
        return self.left_eigvecs.conj().T @ proj

    def eigenfunction_coefficients(
        self,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return coefficients C for Koopman eigenfunctions in kernel expansion:

            C = U_r S_r^{-1/2} V
        """
        U_r_val, S_r_val = self._resolve_reduction_params(U_r, S_r)
        transform = U_r_val * (1.0 / np.sqrt(S_r_val))[None, :]
        return transform @ self.right_eigvecs

    def eigenfunction_gram(
        self,
        S_r: np.ndarray | None = None,
        n_samples: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute the Gram matrix of Koopman eigenfunctions in terms of S_r:

            G_phi = V^* diag(S_r) V

        If normalize=True, divide by n_samples (empirical average).
        """
        S_r_val = self.S_r if S_r is None else S_r
        if S_r_val is None:
            raise ValueError("S_r must be provided or stored on KoopmanSpectrumKDMD")
        G_phi = self.right_eigvecs.conj().T @ np.diag(S_r_val) @ self.right_eigvecs
        if normalize:
            if n_samples is None:
                raise ValueError("n_samples must be provided when normalize=True")
            G_phi = G_phi / float(n_samples)
        return G_phi

    def delta_from_trajectory(
        self,
        trajectories: np.ndarray | Sequence[np.ndarray],
        X_values: np.ndarray
        | dict[int | str, np.ndarray | float | int | None]
        | Sequence[np.ndarray | float | int | None]
        | Callable[[np.ndarray], np.ndarray],
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
        batch_size: int = 5_000,
        show_progress: bool = True,
        return_b: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Estimate Delta_k = sum_j c_k^*(j) <X · ∇_x k(x, x_j)>_0 from trajectories.

        This computes the empirical average of the perturbation field dotted
        with the kernel gradient, then projects onto Koopman eigenfunctions.

        X_values can be:
            - array shape (n_samples, dim)
            - dict {dim_index: values} (indices or 'x','y','z' for dim>=3)
            - sequence length dim with entries as arrays, scalars, or None
            - callable X_fn(data) -> array shape (n_samples, dim)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )
        if not hasattr(kernel_val, "grad_x"):
            raise ValueError("kernel does not support grad_x")

        ref_arr = np.asarray(ref, dtype=float)
        if ref_arr.ndim != 2:
            raise ValueError("reference_data must be a 2D array")
        dim = ref_arr.shape[1]

        # Precompute Koopman eigenfunction coefficients C = U_r S_r^{-1/2} V.
        C = self.eigenfunction_coefficients(U_r_val, S_r_val)

        # If X_values is callable, we evaluate per segment to avoid storing
        # large arrays in memory.
        X_fn = X_values if callable(X_values) else None

        G = None
        n_total = 0
        segments = _iter_trajectory_segments(trajectories)

        for segment in segments:
            seg_arr = np.asarray(segment, dtype=float)
            if seg_arr.ndim != 2:
                raise ValueError("each trajectory segment must be a 2D array")
            n_samples = seg_arr.shape[0]

            # Normalize X_values for this segment. If a callable is provided,
            # evaluate it on the segment to get per-sample values.
            if X_fn is not None:
                X_segment = X_fn(seg_arr)
                active = _normalize_active_dims(X_segment, n_samples, dim)
            else:
                active = _normalize_active_dims(X_values, n_samples, dim)

            # If no active dimensions, we can skip this segment entirely.
            if not active:
                continue

            iterator = range(0, n_samples, batch_size)
            for start in tqdm(iterator, disable=not show_progress):
                end = min(start + batch_size, n_samples)
                x_batch = seg_arr[start:end]

                # Compute kernel gradients for the batch.
                # Shape: (batch, n_centers, dim)
                grad_k = kernel_val.grad_x(x_batch, ref_arr)

                # Accumulate X · ∇_x k(x, x_j) only over active dimensions.
                # This avoids unnecessary work for zero components.
                inner = np.zeros((grad_k.shape[0], grad_k.shape[1]), dtype=grad_k.dtype)
                for d, values in active:
                    v_batch = values[start:end]
                    inner += grad_k[:, :, d] * v_batch[:, None]

                # Sum over samples to build the empirical average.
                if G is None:
                    G = np.zeros((ref_arr.shape[0],), dtype=inner.dtype)
                G += inner.sum(axis=0)
                n_total += inner.shape[0]

        if G is None or n_total == 0:
            raise ValueError("no samples provided to estimate Delta_k")

        # b_j = <X · ∇_x k(x, x_j)>_0
        b = G / float(n_total)

        # Delta_k = sum_j c_k^*(j) b_j
        Delta = C.conj().T @ b
        return (Delta, b) if return_b else Delta

    def eigenfunction_inner_product(self, G: np.ndarray) -> np.ndarray:
        """
        Compute the Gram matrix of KDMD eigenfunctions:

            Xi^* G Xi

        where Xi is the matrix of right eigenvectors (columns), and G is the
        kernel Gram matrix in feature space.
        """
        return self.right_eigvecs.conj().T @ G @ self.right_eigvecs

    def inner_product_inv_measure(
        self,
        trajectories: np.ndarray | Sequence[np.ndarray],
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
        batch_size: int = 5_000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Estimate <phi_k, phi_l>_0 from trajectory segments using batching.

        trajectories can be:
            - array of shape (n_samples, dim)
            - array of shape (n_segments, n_samples, dim)
            - sequence of arrays with shape (n_samples, dim)

        This computes C^* G0 C without explicitly forming G0, where
        C = U_r S_r^{-1/2} V are the kernel expansion coefficients and
        (G0)_{lr} = (1/M) sum_k k(x_k, x_l)^* k(x_k, x_r).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )
        inv_sqrt = 1.0 / np.sqrt(S_r_val)
        transform = U_r_val * inv_sqrt[None, :]
        if self.right_eigvecs.shape[0] != transform.shape[1]:
            raise ValueError("right_eigvecs must match reduced dimension")
        coeffs = transform @ self.right_eigvecs

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
                phi = kernel_val(seg_arr[start:end], ref) @ coeffs
                if G is None:
                    n_features = phi.shape[1]
                    G = np.zeros((n_features, n_features), dtype=phi.dtype)
                G += phi.conj().T @ phi
                n_total += phi.shape[0]

        if G is None or n_total == 0:
            raise ValueError("no samples provided in trajectories")
        G /= n_total
        return G

    def evaluate_eigenfunctions(
        self,
        data: np.ndarray,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Evaluate KDMD eigenfunctions at data points using kernel features.

        Uses:
            phi(x) = k(x, X_ref) @ U_r @ diag(1/sqrt(S_r)) @ v_k
        """
        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )
        data_arr = np.asarray(data, dtype=float)
        squeeze = False
        if data_arr.ndim == 1:
            data_arr = data_arr[None, :]
            squeeze = True
        if data_arr.ndim != 2:
            raise ValueError("data must be a 1D or 2D array")
        if ref.ndim != 2:
            raise ValueError("reference_data must be a 2D array")
        if U_r_val.shape[0] != ref.shape[0]:
            raise ValueError("U_r rows must match reference_data length")
        if U_r_val.shape[1] != S_r_val.shape[0]:
            raise ValueError("U_r columns must match S_r length")
        if self.right_eigvecs.shape[0] != U_r_val.shape[1]:
            raise ValueError("right_eigvecs must match reduced dimension")

        inv_sqrt = 1.0 / np.sqrt(S_r_val)
        transform = U_r_val * inv_sqrt[None, :]
        basis = transform @ self.right_eigvecs

        n_samples = data_arr.shape[0]
        if batch_size is None or batch_size >= n_samples:
            phi = kernel_val(data_arr, ref) @ basis
        else:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            phi = np.empty((n_samples, basis.shape[1]), dtype=basis.dtype)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                phi[start:end] = kernel_val(data_arr[start:end], ref) @ basis

        return phi[0] if squeeze else phi

    def eigenfunction(
        self,
        k: int,
        kernel: Kernel | None = None,
        reference_data: np.ndarray | None = None,
        U_r: np.ndarray | None = None,
        S_r: np.ndarray | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable KDMD eigenfunction phi_k(x).
        """
        if k < 0 or k >= self.right_eigvecs.shape[1]:
            raise IndexError("eigenfunction index out of bounds")
        v_k = self.right_eigvecs[:, k]
        kernel_val, ref, U_r_val, S_r_val = self._resolve_kdmd_params(
            kernel, reference_data, U_r, S_r
        )

        def _phi(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim != 1:
                raise ValueError("x must be a 1D array")
            inv_sqrt = 1.0 / np.sqrt(S_r_val)
            transform = U_r_val * inv_sqrt[None, :]
            return kernel_val(x_arr[None, :], ref) @ (transform @ v_k)

        return _phi

    def correlation_function_continuous(
        self,
        G_phi: np.ndarray,
        coeff_f: np.ndarray,
        coeff_g: np.ndarray,
        eigenvalues: np.ndarray | None = None,
    ):
        """
        Return a callable C_fg(t) using Koopman eigenfunctions (continuous time):

            C_fg(t) = coeff_g^* @ G_phi @ (coeff_f * exp(t * lambda))

        where G_phi is the Gram matrix of Koopman eigenfunctions and lambda are
        continuous-time eigenvalues used in the exponential. If eigenvalues is None,
        self.eigenvalues are used.
        """
        eigs = self.eigenvalues if eigenvalues is None else eigenvalues

        coeff_f = np.asarray(coeff_f).reshape(-1)
        coeff_g = np.asarray(coeff_g).reshape(-1)
        eigs = np.asarray(eigs).reshape(-1)

        # drop the static mode
        coeff_f = coeff_f[1:]
        coeff_g = coeff_g[1:]
        eigs = eigs[1:]
        G_phi = G_phi[1:, 1:]
        row = coeff_g.conj() @ G_phi

        def _corr(t):
            if np.isscalar(t):
                return row @ (coeff_f * np.exp(t * eigs))
            t_arr = np.asarray(t)
            exp_t = np.exp(np.outer(t_arr, eigs))
            return (exp_t * coeff_f[None, :]) @ row

        return _corr
