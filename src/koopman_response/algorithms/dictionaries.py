from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import Callable, DefaultDict, Dict, List, Sequence, Tuple, cast

import numpy as np
from scipy.special import eval_chebyt

from koopman_response.utils.signal import find_index


def _iter_active_dims(
    X_values: np.ndarray | Dict[int, np.ndarray] | Sequence[np.ndarray | float | int | None],
    n_samples: int,
    dim: int,
) -> List[Tuple[int, np.ndarray]]:
    """
    Normalize X_values into a list of (dim_index, values) pairs, skipping zeros.

    Accepted formats:
    - ndarray shape (n_samples, dim)
    - dict {dim_index: values}
    - sequence length dim with entries as arrays, scalars, or None
    """
    active: List[Tuple[int, np.ndarray]] = []

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


def chebyshev_indices(degree: int, dim: int) -> List[Tuple[int, ...]]:
    indices = [
        cast(Tuple[int, ...], i)
        for i in product(range(degree + 1), repeat=dim)
        if sum(i) <= degree
    ]
    return indices


def fourier_indices(order: int, dim: int) -> List[Tuple[int, ...]]:
    indices: List[Tuple[int, ...]] = []
    for k in product(range(-order, order + 1), repeat=dim):
        if all(ki == 0 for ki in k):
            continue
        if sum(abs(ki) for ki in k) <= order:
            indices.append(cast(Tuple[int, ...], k))
    return indices


class Dictionary(ABC):
    """Abstract dictionary interface for EDMD-style algorithms."""

    def fit(self, data: np.ndarray) -> "Dictionary":
        """Optional data-dependent fitting. Default is a no-op."""
        _ = data
        return self

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate dictionary at a single point x (shape: (dim,))."""

    @abstractmethod
    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        """Evaluate dictionary for a batch (shape: (n_samples, dim))."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of dictionary features."""

    def build_derivative_matrix(self, direction: int) -> np.ndarray:
        """
        Optional: return a matrix D such that D @ c gives coefficients of
        d/dx_direction applied to a dictionary expansion.
        """
        raise NotImplementedError(
            "build_derivative_matrix not implemented for this dictionary"
        )

    def delta_from_trajectory(
        self, data: np.ndarray, X_values: np.ndarray
    ) -> np.ndarray:
        """
        Optional: compute Delta_i = <X · ∇psi_i^*, 1>_0 from trajectory samples.
        """
        raise NotImplementedError(
            "delta_from_trajectory not implemented for this dictionary"
        )

    def delta_from_callable(
        self, data: np.ndarray, X_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Optional: compute Delta_i from trajectory samples by evaluating X_fn on data.
        """
        X_values = X_fn(data)
        return self.delta_from_trajectory(data, X_values)

    def response_coefficients(self, G: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Compute response coefficients:

            gamma = G^+ Delta
        """
        return np.linalg.pinv(G) @ delta

    def delta_from_constant(
        self, data: np.ndarray, X_const: np.ndarray
    ) -> np.ndarray:
        """
        Optional: compute Delta_i for constant forcing X_const.
        """
        X_const = np.asarray(X_const)
        if X_const.ndim != 1:
            raise ValueError("X_const must be a 1D array of shape (dim,)")
        X_values = np.tile(X_const, (data.shape[0], 1))
        return self.delta_from_trajectory(data, X_values)


class ChebyshevDictionary(Dictionary):
    """Tensorized Chebyshev dictionary on [-1, 1]^dim."""

    def __init__(self, degree: int, dim: int = 3):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.degree = degree
        self.dim = dim
        self.indices = chebyshev_indices(degree, dim)
        self._derivative_matrices: Dict[int, np.ndarray] = {}

    @property
    def n_features(self) -> int:
        return len(self.indices)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected x with dim={self.dim}, got {x.shape[0]}")
        return np.array(
            [
                np.prod([eval_chebyt(k, x[d]) for d, k in enumerate(idx)])
                for idx in self.indices
            ],
            dtype=np.float64,
        )

    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != self.dim:
            raise ValueError(
                f"Expected data with dim={self.dim}, got {data.shape[1]}"
            )
        n_samples = data.shape[0]
        n_features = len(self.indices)
        psi = np.empty((n_samples, n_features), dtype=np.float64)
        for n, idx in enumerate(self.indices):
            col = np.ones(n_samples, dtype=np.float64)
            for d, k in enumerate(idx):
                col *= eval_chebyt(k, data[:, d])
            psi[:, n] = col
        return psi

    def chebyshev_U_to_T_matrix(self, n: int) -> np.ndarray:
        """
        Create a matrix M such that:
        M[n, m] gives the coefficient of T_m in U_n(x)
        Returns an (n x n) matrix.
        """
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(0, i + 1, 2):
                m[i, j] = 2
        m[0, 0] = 1  # U_0(x) = T_0(x)
        return m

    def spectral_derivative_tensor_chebyshev_explicit(
        self, c_flat: np.ndarray, direction: int
    ) -> np.ndarray:
        """
        Compute the spectral derivative of a tensorized Chebyshev T_n basis expansion
        along the specified direction (0..dim-1), using the exact formula.
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim - 1}]")

        index_map = {idx: n for n, idx in enumerate(self.indices)}
        dc_dict: DefaultDict[Tuple[int, ...], float] = defaultdict(float)

        for n, idx in enumerate(self.indices):
            coeff = c_flat[n]
            deg = idx[direction]

            if deg == 0:
                continue  # T_0 -> 0

            scale = deg  # from derivative rule
            u_index = deg - 1

            if u_index % 2 == 0:  # even
                for m in range(0, u_index + 1, 2):
                    new_idx = list(idx)
                    new_idx[direction] = m
                    new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                    if new_idx_tuple in index_map:
                        dc_dict[new_idx_tuple] += coeff * scale * 2
                # subtract the constant 1 term
                new_idx = list(idx)
                new_idx[direction] = 0
                new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                if new_idx_tuple in index_map:
                    dc_dict[new_idx_tuple] -= coeff * scale
            else:  # odd
                for m in range(1, u_index + 1, 2):
                    new_idx = list(idx)
                    new_idx[direction] = m
                    new_idx_tuple = cast(Tuple[int, ...], tuple(new_idx))
                    if new_idx_tuple in index_map:
                        dc_dict[new_idx_tuple] += coeff * scale * 2

        dc_flat = np.zeros_like(c_flat)
        for idx, val in dc_dict.items():
            dc_flat[index_map[idx]] = val

        return dc_flat

    def build_derivative_matrix(self, direction: int) -> np.ndarray:
        """
        Constructs the matrix A^{(direction)} such that:
        A @ c = coefficients of d/dx_i f(x), when f(x) = sum c_n psi_n(x)
        """
        if direction in self._derivative_matrices:
            return self._derivative_matrices[direction]

        n = len(self.indices)
        a = np.zeros((n, n))
        eye = np.eye(n)
        for i in range(n):
            a[:, i] = self.spectral_derivative_tensor_chebyshev_explicit(
                eye[:, i], direction
            )
        self._derivative_matrices[direction] = a
        return a

    def delta_from_trajectory(
        self, data: np.ndarray, X_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute Delta_i = (1/N) sum_t X(x_t) · ∇psi_i^*(x_t)
        for Chebyshev dictionary basis functions.

        Skips directions where X_values is identically zero.

        X_values can be:
        - ndarray (n_samples, dim)
        - dict {dim_index: values}
        - sequence length dim with arrays/scalars/None
        """
        data = np.asarray(data)

        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError("data must have shape (n_samples, dim)")

        active = _iter_active_dims(X_values, data.shape[0], self.dim)
        if not active:
            return np.zeros(self.n_features, dtype=np.float64)

        psi = self.evaluate_batch(data)
        delta = np.zeros(self.n_features, dtype=np.float64)
        for d, x_d in active:
            dpsi = psi @ self.build_derivative_matrix(d)
            delta += np.mean(dpsi.conj() * x_d[:, None], axis=0)
        return delta

    def delta_from_constant(
        self, data: np.ndarray, X_const: np.ndarray
    ) -> np.ndarray:
        """
        Compute Delta_i = (1/N) sum_t X_const · ∇psi_i^*(x_t)
        for constant forcing X_const.
        """
        data = np.asarray(data)
        X_const = np.asarray(X_const)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError("data must have shape (n_samples, dim)")
        if X_const.ndim != 1 or X_const.shape[0] != self.dim:
            raise ValueError("X_const must have shape (dim,)")

        active_dims = np.where(X_const != 0)[0]
        if active_dims.size == 0:
            return np.zeros(self.n_features, dtype=np.float64)

        psi = self.evaluate_batch(data)
        delta = np.zeros(self.n_features, dtype=np.float64)
        for d in active_dims:
            dpsi = psi @ self.build_derivative_matrix(d)
            delta += X_const[d] * np.mean(dpsi.conj(), axis=0)
        return delta

    def get_decomposition_observables(self) -> Dict[str, np.ndarray]:
        """
        Return Chebyshev coefficient vectors for common observables.
        """
        if self.dim != 3:
            raise ValueError("Observable decomposition is defined for dim=3.")

        dictionary_decomposition: Dict[str, np.ndarray] = {}

        def coeff_for_degree(degree: Tuple[int, int, int], weight: float = 1.0):
            index = find_index(self.indices, degree)
            coeffs = np.zeros(len(self.indices))
            coeffs[index] = weight
            return coeffs

        dictionary_decomposition["z"] = coeff_for_degree((0, 0, 1))

        # x^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((2, 0, 0), 1 / 2)
        dictionary_decomposition["x^2"] = coeffs

        # y^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((0, 2, 0), 1 / 2)
        dictionary_decomposition["y^2"] = coeffs

        # z^2
        coeffs = np.zeros(len(self.indices))
        coeffs += coeff_for_degree((0, 0, 0), 1 / 2)
        coeffs += coeff_for_degree((0, 0, 2), 1 / 2)
        dictionary_decomposition["z^2"] = coeffs

        # xy
        dictionary_decomposition["xy"] = coeff_for_degree((1, 1, 0))

        return dictionary_decomposition


class FourierDictionary(Dictionary):
    """
    Complex Fourier dictionary on a torus with possibly different periods per axis.

    Features: exp(i * 2π * sum_d k_d * x_d / L_d) for integer k with |k|_1 <= order.
    Inputs are assumed to be in the same units as L (period L per dimension).
    If normalize=True, all features are scaled by 1/sqrt(prod(L)).
    """

    def __init__(
        self,
        order: int,
        dim: int = 1,
        L: float | Sequence[float] = 2 * np.pi,
        include_constant: bool = True,
        normalize: bool = True,
    ):
        if order < 0:
            raise ValueError("order must be >= 0")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.order = order
        self.dim = dim
        self.include_constant = include_constant
        self.k_vectors = np.array(fourier_indices(order, dim), dtype=int)

        if isinstance(L, (list, tuple, np.ndarray)):
            if len(L) != dim:
                raise ValueError("L must have length dim when passing a sequence")
            self.L = np.array(L, dtype=float)
        else:
            self.L = np.full(dim, float(L))

        if np.any(self.L <= 0):
            raise ValueError("L must be positive in every dimension")

        self.normalize = normalize
        self._norm_factor = 1.0 / np.sqrt(np.prod(self.L)) if normalize else 1.0

    @property
    def n_features(self) -> int:
        base = 1 if self.include_constant else 0
        return base + self.k_vectors.shape[0]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected x with dim={self.dim}, got {x.shape[0]}")
        if self.k_vectors.size == 0:
            base = np.array([1.0], dtype=np.complex128) if self.include_constant else np.array([], dtype=np.complex128)
            return base * self._norm_factor

        phases = (self.k_vectors / self.L) @ x
        values = np.exp(1j * 2 * np.pi * phases)

        if self.include_constant:
            return np.concatenate(([1.0], values)).astype(np.complex128) * self._norm_factor
        return values.astype(np.complex128) * self._norm_factor

    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != self.dim:
            raise ValueError(
                f"Expected data with dim={self.dim}, got {data.shape[1]}"
            )

        n_samples = data.shape[0]
        n_k = self.k_vectors.shape[0]

        if n_k == 0:
            if self.include_constant:
                return np.ones((n_samples, 1), dtype=np.complex128) * self._norm_factor
            return np.empty((n_samples, 0), dtype=np.complex128)

        phases = (data / self.L) @ self.k_vectors.T
        values = np.exp(1j * 2 * np.pi * phases)

        if self.include_constant:
            ones = np.ones((n_samples, 1), dtype=np.complex128)
            return np.concatenate((ones, values), axis=1) * self._norm_factor
        return values.astype(np.complex128) * self._norm_factor

    def derivative_factors(self) -> np.ndarray:
        """
        Return factors F such that:
            d/dx_d psi_j = F[j, d] * psi_j
        for each basis function psi_j.
        """
        n_k = self.k_vectors.shape[0]
        factors = np.zeros((self.n_features, self.dim), dtype=np.complex128)
        if n_k == 0:
            return factors

        base = 1 if self.include_constant else 0
        coeffs = (1j * 2 * np.pi) * (self.k_vectors / self.L)
        factors[base : base + n_k, :] = coeffs
        return factors

    def build_derivative_matrix(self, direction: int) -> np.ndarray:
        """
        Diagonal matrix for d/dx_direction in the Fourier basis.
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim - 1}]")
        factors = self.derivative_factors()[:, direction]
        return np.diag(factors)

    def delta_from_trajectory(
        self, data: np.ndarray, X_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute Delta_i = (1/N) sum_t X(x_t) · ∇psi_i^*(x_t)
        for Fourier dictionary basis functions.

        X_values can be:
        - ndarray (n_samples, dim)
        - dict {dim_index: values}
        - sequence length dim with arrays/scalars/None
        """
        data = np.asarray(data)

        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError("data must have shape (n_samples, dim)")

        active = _iter_active_dims(X_values, data.shape[0], self.dim)
        if not active:
            return np.zeros(self.n_features, dtype=np.complex128)

        phi = self.evaluate_batch(data)
        factors = self.derivative_factors()

        phi_conj = phi.conj()
        factors_conj = factors.conj()

        delta = np.zeros(self.n_features, dtype=np.complex128)
        for d, x_d in active:
            delta += np.mean(phi_conj * factors_conj[:, d] * x_d[:, None], axis=0)

        return delta

    def delta_from_constant(
        self, data: np.ndarray, X_const: np.ndarray
    ) -> np.ndarray:
        """
        Compute Delta_i = (1/N) sum_t X_const · ∇psi_i^*(x_t)
        for constant forcing X_const.
        """
        data = np.asarray(data)
        X_const = np.asarray(X_const)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError("data must have shape (n_samples, dim)")
        if X_const.ndim != 1 or X_const.shape[0] != self.dim:
            raise ValueError("X_const must have shape (dim,)")

        phi = self.evaluate_batch(data)
        factors = self.derivative_factors()

        phi_conj = phi.conj()
        weights = factors.conj() @ X_const  # shape (n_features,)
        return np.mean(phi_conj * weights[None, :], axis=0)
