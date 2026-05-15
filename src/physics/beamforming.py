"""Beamforming weights and beampatterns."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.signal import windows

from .simulator import steering_matrix_ula, steering_vector_ula


def delay_and_sum(n_elements: int, spacing: float, steer_deg: float) -> np.ndarray:
    """Conventional uniform weights steered to ``steer_deg``.

    Normalized so the broadside-equivalent gain is unity: ``w^H a(steer) = 1``.
    """
    a = steering_vector_ula(n_elements, spacing, steer_deg)
    return a / n_elements


def _diag_loaded(R: np.ndarray, diag_load: float) -> np.ndarray:
    if diag_load <= 0.0:
        return R
    load = diag_load * np.trace(R).real / R.shape[0]
    return R + load * np.eye(R.shape[0])


def _capon_weights(
    R: np.ndarray,
    n_elements: int,
    spacing: float,
    steer_deg: float,
    diag_load: float,
) -> np.ndarray:
    a = steering_vector_ula(n_elements, spacing, steer_deg)
    Rl = _diag_loaded(R, diag_load)
    Rinv_a = scipy.linalg.solve(Rl, a, assume_a="pos")
    return Rinv_a / (a.conj() @ Rinv_a)


def mvdr(
    R_in: np.ndarray,
    n_elements: int,
    spacing: float,
    steer_deg: float,
    *,
    diag_load: float = 0.0,
) -> np.ndarray:
    """Minimum-Variance Distortionless Response weights.

    ``w = R_in^{-1} a / (a^H R_in^{-1} a)``, where ``R_in`` is the
    *interference-plus-noise* covariance -- the desired signal must be
    absent from the training data (e.g. estimated from a signal-free
    period). Robust to steering-vector mismatch.

    ``diag_load`` adds Tikhonov regularization as a fraction of
    ``trace(R)/N`` (set ~1e-3 when ``R`` is estimated from few snapshots).
    """
    return _capon_weights(R_in, n_elements, spacing, steer_deg, diag_load)


def mpdr(
    R_xx: np.ndarray,
    n_elements: int,
    spacing: float,
    steer_deg: float,
    *,
    diag_load: float = 0.0,
) -> np.ndarray:
    """Minimum-Power Distortionless Response (Capon) weights.

    Same closed form as MVDR -- ``w = R_xx^{-1} a / (a^H R_xx^{-1} a)`` --
    but applied to the *full* data covariance ``R_xx`` that includes the
    desired signal. This is what is computed in practice from received
    snapshots, but it self-cancels the signal of interest under any
    steering-vector mismatch (assumed look angle != true source angle,
    array calibration error, etc.).
    """
    return _capon_weights(R_xx, n_elements, spacing, steer_deg, diag_load)


def lcmv(
    R: np.ndarray,
    C: np.ndarray,
    f: np.ndarray,
    *,
    diag_load: float = 0.0,
) -> np.ndarray:
    """LCMV beamformer with linear constraint ``C^H w = f``.

    ``w = R^{-1} C (C^H R^{-1} C)^{-1} f``.
    ``C`` shape ``(N, M)``; ``f`` shape ``(M,)``.
    """
    C = np.atleast_2d(C)
    f = np.atleast_1d(f)
    Rl = _diag_loaded(R, diag_load)
    Rinv_C = scipy.linalg.solve(Rl, C, assume_a="pos")  # shape (N, M)
    G = C.conj().T @ Rinv_C                              # shape (M, M)
    mu = scipy.linalg.solve(G, f)                        # shape (M,)
    return Rinv_C @ mu


def chebychev_weights(n_elements: int, sll_db: float) -> np.ndarray:
    """Dolph-Chebychev taper.

    ``sll_db`` is the attenuation in dB of sidelobes relative to the main
    lobe, entered as a positive number (e.g. ``30`` => -30 dB).
    """
    w = windows.chebwin(n_elements, at=float(sll_db))
    return w / w.sum()


def taylor_weights(n_elements: int, nbar: int = 4, sll_db: float = 30.0) -> np.ndarray:
    """Taylor n-bar taper."""
    w = windows.taylor(n_elements, nbar=nbar, sll=float(sll_db), norm=2)
    return w / w.sum()


def beampattern(
    w: np.ndarray,
    n_elements: int,
    spacing: float,
    angle_grid_deg: np.ndarray,
) -> np.ndarray:
    """``|w^H a(theta)|^2`` in dB over the angle grid, normalized to its peak."""
    A = steering_matrix_ula(n_elements, spacing, angle_grid_deg)
    resp = np.abs(w.conj() @ A) ** 2
    peak = resp.max() if resp.max() > 0 else 1.0
    resp = resp / peak
    return 10.0 * np.log10(np.clip(resp, 1e-12, None))
