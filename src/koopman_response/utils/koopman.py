from __future__ import annotations

from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig


def get_spectral_properties(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the sorted (decreasing orders in terms of absolute value) of eigenvalues
    and eigenvectors of the Koopman matrix.
    """
    eig_result = cast(
        tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]],
        eig(K, left=True, right=True),
    )

    eigenvalues, left_eigenvectors, right_eigenvectors = eig_result

    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    right_eigenvectors = right_eigenvectors[:, sorted_indices]
    left_eigenvectors = left_eigenvectors[:, sorted_indices]

    diag = np.diag(left_eigenvectors.T.conj() @ right_eigenvectors)
    scale_factors = 1.0 / np.sqrt(diag)
    right_eigenvectors_normalised = right_eigenvectors * scale_factors[np.newaxis, :]
    left_eigenvectors_normalised = (
        left_eigenvectors * scale_factors[np.newaxis, :].conj()
    )
    return eigenvalues, right_eigenvectors_normalised, left_eigenvectors_normalised
