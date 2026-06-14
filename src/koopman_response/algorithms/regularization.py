from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh


@dataclass
class TSVDRegularizer:
    """
    Truncated SVD regularizer for EDMD.

    Given G and A, computes an orthonormalized reduced Koopman operator:
        K_r = S_r^{-1/2} U_r^* A U_r S_r^{-1/2}
    where G = U S U^* and r is chosen by rank or relative threshold.
    """

    rel_threshold: float = 1e-6
    rank: Optional[int] = None

    Ur: Optional[np.ndarray] = None
    Sr: Optional[np.ndarray] = None
    Kr: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    factorization_is_full: bool = False

    def factorize(
        self,
        G: np.ndarray,
        method: str = "eigh",
        symmetrize: bool = True,
        rel_threshold: Optional[float] = None,
        rank: Optional[int] = None,
        initial_rank: int = 64,
        max_rank: Optional[int] = None,
        growth_factor: float = 2.0,
        tol: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorize the Gram matrix once and cache the result.

        Parameters:
            G: Gram matrix (square).
            method: "eigh" (symmetric eigendecomposition), "svd", or "eigsh"
                (partial symmetric eigendecomposition).
            symmetrize: if True, use (G + G^*) / 2 before eigh.
            rel_threshold: relative threshold for partial "eigsh" factorization.
            rank: fixed rank for partial "eigsh" factorization.
            initial_rank: starting rank for adaptive partial factorization.
            max_rank: maximum rank to try in adaptive partial factorization.
            growth_factor: factor used to increase rank adaptively.
            tol: eigsh solver tolerance.
        """
        if G.shape[0] != G.shape[1]:
            raise ValueError("G must be square")

        if rel_threshold is not None:
            self.rel_threshold = float(rel_threshold)
            self.rank = None
        if rank is not None:
            self.rank = int(rank)
            self.rel_threshold = None

        if method == "eigh":
            G_use = 0.5 * (G + G.conj().T) if symmetrize else G
            U, S = self._factorize_full_eigh(G_use)
            factorization_is_full = True
        elif method == "svd":
            U, S, _ = np.linalg.svd(G, full_matrices=False)
            factorization_is_full = True
        elif method == "eigsh":
            G_use = 0.5 * (G + G.conj().T) if symmetrize else G
            U, S, factorization_is_full = self._factorize_eigsh(
                G_use,
                initial_rank=initial_rank,
                max_rank=max_rank,
                growth_factor=growth_factor,
                tol=tol,
            )
        else:
            raise ValueError("method must be 'eigh', 'svd', or 'eigsh'")

        self.U = U
        self.S = S
        self.factorization_is_full = factorization_is_full
        return U, S

    @staticmethod
    def _factorize_full_eigh(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S, U = np.linalg.eigh(G)
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        S = np.maximum(S, 0.0)
        return U, S

    @staticmethod
    def _factorize_partial_eigsh(
        G: np.ndarray,
        rank: int,
        tol: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank >= G.shape[0]:
            return TSVDRegularizer._factorize_full_eigh(G)

        S, U = eigsh(G, k=rank, which="LA", tol=tol)
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        S = np.maximum(S, 0.0)
        return U, S

    def _factorize_eigsh(
        self,
        G: np.ndarray,
        initial_rank: int,
        max_rank: Optional[int],
        growth_factor: float,
        tol: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        n = G.shape[0]
        rank_val = self.rank
        rel_val = self.rel_threshold

        if rank_val is not None:
            r = int(rank_val)
            if r < 1:
                raise ValueError("rank must be >= 1")
            if r > n:
                raise ValueError("rank cannot exceed matrix size")
            if r >= n:
                U, S = self._factorize_full_eigh(G)
                return U, S, True
            U, S = self._factorize_partial_eigsh(G, r, tol)
            return U, S, False

        if rel_val is None:
            raise ValueError("rel_threshold or rank must be set for method='eigsh'")
        rel_float = float(rel_val)
        if rel_float < 0:
            raise ValueError("rel_threshold must be non-negative")
        if initial_rank < 1:
            raise ValueError("initial_rank must be >= 1")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1")

        max_rank_val = n if max_rank is None else int(max_rank)
        if max_rank_val < 1:
            raise ValueError("max_rank must be >= 1")
        if max_rank_val > n:
            raise ValueError("max_rank cannot exceed matrix size")

        k = min(max(int(initial_rank), 1), max_rank_val)
        while True:
            if k >= n:
                U, S = self._factorize_full_eigh(G)
                return U, S, True

            U, S = self._factorize_partial_eigsh(G, k, tol)
            if S.size == 0:
                raise ValueError("Empty spectrum; cannot truncate.")
            if S[0] <= 0:
                raise ValueError("Largest eigenvalue is non-positive; cannot truncate.")

            cutoff = rel_float * S[0]
            if S[-1] <= cutoff or k >= max_rank_val:
                return U, S, False

            k_next = int(np.ceil(k * growth_factor))
            if k_next <= k:
                k_next = k + 1
            k = min(k_next, max_rank_val)

    def _truncate(
        self,
        rel_threshold: Optional[float],
        rank: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.U is None or self.S is None:
            raise ValueError("Gram factorization missing. Call factorize() first.")

        rel_val = self.rel_threshold if rel_threshold is None else float(rel_threshold)
        rank_val = self.rank if rank is None else rank
        if rank_val is not None:
            r = int(rank_val)
            if r > self.S.size and not self.factorization_is_full:
                raise ValueError(
                    "Requested rank exceeds the partial factorization. "
                    "Refactorize with a larger rank or max_rank."
                )
        else:
            if self.S.size == 0:
                raise ValueError("Empty spectrum; cannot truncate.")
            if self.S[0] <= 0:
                raise ValueError("Largest eigenvalue is non-positive; cannot truncate.")
            r = int(np.sum(self.S > rel_val * self.S[0]))
            if (
                r == self.S.size
                and not self.factorization_is_full
                and self.S[-1] > rel_val * self.S[0]
            ):
                raise ValueError(
                    "Relative threshold was not reached by the partial factorization. "
                    "Refactorize with a larger max_rank or lower rel_threshold."
                )
        if r < 1:
            raise ValueError("Truncation rank is < 1. Increase rel_threshold or rank.")

        Ur = self.U[:, :r]
        Sr = self.S[:r]
        return Ur, Sr

    def solve_from_factorization(
        self,
        A: np.ndarray,
        rel_threshold: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if self.U is None or self.S is None:
            raise ValueError("Gram factorization missing. Call factorize() first.")
        if A.shape[0] != self.U.shape[0]:
            raise ValueError("A must have the same shape as the Gram factorization.")

        if rel_threshold is not None:
            self.rel_threshold = float(rel_threshold)
            self.rank = None
        if rank is not None:
            self.rank = int(rank)
            self.rel_threshold = None

        Ur, Sr = self._truncate(rel_threshold, rank)
        Sr_inv_sqrt = np.diag(1.0 / np.sqrt(Sr))
        Kr = Sr_inv_sqrt @ (Ur.conj().T @ A @ Ur) @ Sr_inv_sqrt

        self.Ur = Ur
        self.Sr = Sr
        self.Kr = Kr
        return Kr, Ur, Sr

    def lift_eigenvectors(self, W: np.ndarray) -> np.ndarray:
        if self.Ur is None:
            raise ValueError("TSVD must be solved before lifting eigenvectors")
        if self.Sr is None:
            raise ValueError("TSVD must be solved before lifting eigenvectors")
        Sr_inv_sqrt = np.diag(1.0 / np.sqrt(self.Sr))
        return self.Ur @ Sr_inv_sqrt @ W

    def gram_inverse(self) -> np.ndarray:
        if self.Ur is None or self.Sr is None:
            raise ValueError("TSVD must be solved before computing gram inverse")
        Sr_inv = np.diag(1.0 / self.Sr)
        return self.Ur @ Sr_inv @ self.Ur.conj().T
