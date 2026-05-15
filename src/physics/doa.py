"""Direction-of-arrival estimators."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.signal import find_peaks

from .simulator import steering_matrix_ula


def _signal_subspace(R: np.ndarray, n_sources: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (U_s, U_n) from a Hermitian covariance ``R``."""
    eigvals, eigvecs = scipy.linalg.eigh(R)  # ascending order
    U_s = eigvecs[:, -n_sources:]
    U_n = eigvecs[:, :-n_sources] if n_sources < R.shape[0] else np.zeros((R.shape[0], 0))
    return U_s, U_n


def music(
    R: np.ndarray,
    n_sources: int,
    n_elements: int,
    spacing: float,
    angle_grid_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MUSIC pseudo-spectrum in dB plus the top-K DOAs."""
    _, U_n = _signal_subspace(R, n_sources)
    A = steering_matrix_ula(n_elements, spacing, angle_grid_deg)      # (N, G)
    proj = U_n.conj().T @ A                                           # (N-K, G)
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    spectrum = 1.0 / np.maximum(denom, 1e-15)
    spectrum_db = 10.0 * np.log10(spectrum / spectrum.max())

    peaks, props = find_peaks(spectrum_db, prominence=1.0)
    if peaks.size == 0:
        doas = np.array([])
    else:
        order = np.argsort(props["prominences"])[::-1][:n_sources]
        doas = np.sort(np.asarray(angle_grid_deg)[peaks[order]])
    return spectrum_db, doas


def esprit(
    x_or_R: np.ndarray,
    n_sources: int,
    spacing: float,
) -> np.ndarray:
    """TLS-ESPRIT for a ULA. Accepts either the snapshot matrix ``X (N, L)`` or
    a covariance ``R (N, N)``. Returns DOAs in degrees.
    """
    if x_or_R.shape[0] == x_or_R.shape[1]:
        _, U_s = scipy.linalg.eigh(x_or_R, subset_by_index=[x_or_R.shape[0] - n_sources, x_or_R.shape[0] - 1])
        U_s = U_s  # already (N, K)
    else:
        # signal subspace from economy SVD of X
        U, _, _ = scipy.linalg.svd(x_or_R, full_matrices=False)
        U_s = U[:, :n_sources]

    U1 = U_s[:-1, :]
    U2 = U_s[1:, :]
    # Total least squares: SVD of [U1 U2] (N-1, 2K)
    _, _, Vh = scipy.linalg.svd(np.hstack([U1, U2]), full_matrices=False)
    V = Vh.conj().T
    V12 = V[:n_sources, n_sources:]   # (K, K)
    V22 = V[n_sources:, n_sources:]   # (K, K)
    Psi = -V12 @ scipy.linalg.inv(V22)
    eigs = scipy.linalg.eigvals(Psi)

    # eigs ~ exp(j 2pi d sin theta_k)
    phi = np.angle(eigs)
    sin_theta = phi / (2.0 * np.pi * spacing)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return np.sort(np.rad2deg(np.arcsin(sin_theta)))


def omp(
    x: np.ndarray,
    n_elements: int,
    spacing: float,
    angle_grid_deg: np.ndarray,
    sparsity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Matching Pursuit on the over-complete angular dictionary.

    Uses the mean snapshot (or a single snapshot) as the measurement vector;
    for multi-snapshot data it is often sufficient to feed ``x.mean(axis=1)``
    or to loop the caller.
    """
    if x.ndim == 1:
        y = x.astype(complex)
    else:
        # stack into a block vector -- equivalent to matching the single average snapshot
        y = x.mean(axis=1).astype(complex)

    angle_grid_deg = np.asarray(angle_grid_deg, dtype=float)
    A = steering_matrix_ula(n_elements, spacing, angle_grid_deg)
    col_norms = np.linalg.norm(A, axis=0)
    A_norm = A / col_norms

    residual = y.copy()
    support: list[int] = []
    for _ in range(sparsity):
        corr = np.abs(A_norm.conj().T @ residual)
        corr[support] = -np.inf
        idx = int(np.argmax(corr))
        support.append(idx)
        As = A_norm[:, support]
        coef, *_ = scipy.linalg.lstsq(As, y)
        residual = y - As @ coef

    doas = np.sort(angle_grid_deg[np.array(support)])
    amplitudes = np.zeros(angle_grid_deg.size)
    amplitudes[support] = np.abs(coef)
    peak = amplitudes.max() if amplitudes.max() > 0 else 1.0
    spectrum_db = 20.0 * np.log10(np.clip(amplitudes / peak, 1e-6, None))
    return spectrum_db, doas


def spatial_smoothing(
    R: np.ndarray,
    subarray_size: int,
    *,
    forward_backward: bool = True,
) -> np.ndarray:
    """Forward (and optionally backward) spatial smoothing.

    With ``subarray_size = L``, produces an ``L * L`` covariance averaged over
    ``N - L + 1`` overlapping subarrays (plus their backward-flipped conjugates
    when ``forward_backward`` is true).
    """
    N = R.shape[0]
    L = subarray_size
    if not 1 <= L <= N:
        raise ValueError("subarray_size must satisfy 1 <= L <= N")
    M = N - L + 1
    Rf = np.zeros((L, L), dtype=complex)
    for m in range(M):
        Rf += R[m : m + L, m : m + L]
    Rf /= M
    if not forward_backward:
        return Rf
    J = np.fliplr(np.eye(L))
    Rb = J @ Rf.conj() @ J
    return 0.5 * (Rf + Rb)
