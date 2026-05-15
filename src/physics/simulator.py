"""ULA / planar array data generation.

All angles are in degrees unless stated otherwise. Spacing is in wavelengths.
Returned snapshot matrix X has shape (N, L) where N = #elements, L = #snapshots.
"""

from __future__ import annotations

import numpy as np


def steering_vector_ula(n_elements: int, spacing: float, angle_deg: float) -> np.ndarray:
    """Narrowband ULA steering vector, broadside = 0 deg."""
    n = np.arange(n_elements)
    phase = 2.0 * np.pi * spacing * n * np.sin(np.deg2rad(angle_deg))
    return np.exp(1j * phase)


def steering_matrix_ula(n_elements: int, spacing: float, angles_deg: np.ndarray) -> np.ndarray:
    """Columns are steering vectors for the given angles. Shape (N, K)."""
    angles_deg = np.atleast_1d(angles_deg).astype(float)
    n = np.arange(n_elements)[:, None]
    phase = 2.0 * np.pi * spacing * n * np.sin(np.deg2rad(angles_deg))[None, :]
    return np.exp(1j * phase)


def generate_ula_data(
    n_elements: int,
    spacing: float,
    source_angles_deg,
    source_powers_db=None,
    snapshots: int = 200,
    snr_db: float = 10.0,
    *,
    coherent: bool = False,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate narrowband ULA snapshots.

    Signal model: ``X = A S + N`` with circularly-symmetric complex Gaussian
    sources ``S`` and noise ``N``. Noise variance is fixed at 1; per-source
    amplitude is set so that a 0 dB entry in ``source_powers_db`` corresponds
    to output ``snr_db`` relative to that unit noise power.

    Parameters
    ----------
    source_powers_db : array-like or None
        Per-source power offsets in dB relative to the SNR reference. ``None``
        means 0 dB (all sources at nominal SNR).
    coherent : bool
        If True, all sources share the same random snapshot stream (perfectly
        coherent). Useful for exercising spatial-smoothing code paths.

    Returns
    -------
    X : ndarray, shape (N, L), complex
    R : ndarray, shape (N, N), sample covariance ``X X^H / L``
    """
    rng = np.random.default_rng(rng)
    angles = np.atleast_1d(np.asarray(source_angles_deg, dtype=float))
    K = angles.size
    if source_powers_db is None:
        powers_db = np.zeros(K)
    else:
        powers_db = np.atleast_1d(np.asarray(source_powers_db, dtype=float))
        if powers_db.size != K:
            raise ValueError("source_powers_db must match source_angles_deg in length")

    A = steering_matrix_ula(n_elements, spacing, angles)

    # Source amplitudes: amp_k = sqrt(10^((snr_db + power_k_db) / 10)).
    amplitudes = np.sqrt(10.0 ** ((snr_db + powers_db) / 10.0))

    if coherent:
        base = (rng.standard_normal(snapshots) + 1j * rng.standard_normal(snapshots)) / np.sqrt(2)
        S = np.outer(amplitudes, base)
    else:
        S = (rng.standard_normal((K, snapshots)) + 1j * rng.standard_normal((K, snapshots))) / np.sqrt(2)
        S *= amplitudes[:, None]

    noise = (rng.standard_normal((n_elements, snapshots)) + 1j * rng.standard_normal((n_elements, snapshots))) / np.sqrt(2)

    X = A @ S + noise
    R = (X @ X.conj().T) / snapshots
    return X, R
