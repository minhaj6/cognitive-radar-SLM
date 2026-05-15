"""Performance metrics: SINR, sidelobe level, Rayleigh limit, 3 dB beamwidth."""

from __future__ import annotations

import numpy as np


def output_sinr(
    w: np.ndarray,
    a_signal: np.ndarray,
    R_ipn: np.ndarray,
    signal_power: float = 1.0,
) -> float:
    """Output SINR ``sigma_s^2 |w^H a_s|^2 / Re(w^H R_{i+n} w)``."""
    num = signal_power * np.abs(w.conj() @ a_signal) ** 2
    den = np.real(w.conj() @ (R_ipn @ w))
    if den <= 0.0:
        return float("inf") if num > 0 else 0.0
    return float(num / den)


def rayleigh_resolution_deg(n_elements: int, spacing: float, steer_deg: float = 0.0) -> float:
    """Approximate Rayleigh angular resolution of a ULA.

    ``Deltatheta ~ 2 / (N d cos theta_s)`` radians for half-wavelength spacing; returned in
    degrees. Guards against the endfire singularity.
    """
    cos_s = max(float(np.cos(np.deg2rad(steer_deg))), 1e-6)
    return float(np.rad2deg(2.0 / (n_elements * spacing * cos_s)))


def sidelobe_level_db(
    pattern_db: np.ndarray,
    angles_deg: np.ndarray,
    mainlobe_deg: float,
) -> float:
    """Peak sidelobe level in dB relative to the main-lobe maximum.

    Walks outward from the sample nearest ``mainlobe_deg`` until ``pattern_db``
    stops monotonically decreasing (first local minimum) on each side; that
    span is the mainlobe region and is excluded before the remaining maximum
    is returned.
    """
    pattern_db = np.asarray(pattern_db)
    angles_deg = np.asarray(angles_deg)
    n = pattern_db.size
    center = int(np.argmin(np.abs(angles_deg - mainlobe_deg)))

    left = center
    while left > 0 and pattern_db[left - 1] < pattern_db[left]:
        left -= 1
    right = center
    while right < n - 1 and pattern_db[right + 1] < pattern_db[right]:
        right += 1

    mask = np.ones(n, dtype=bool)
    mask[left : right + 1] = False
    if not mask.any():
        return float("nan")
    return float(pattern_db[mask].max())


def mainlobe_3db_width_deg(
    pattern_db: np.ndarray,
    angles_deg: np.ndarray,
    mainlobe_deg: float,
) -> float:
    """3 dB beamwidth of the main lobe (linear interpolation to the crossings)."""
    pattern_db = np.asarray(pattern_db)
    angles_deg = np.asarray(angles_deg)
    center = int(np.argmin(np.abs(angles_deg - mainlobe_deg)))
    peak = float(pattern_db[center])
    target = peak - 3.0

    def _cross(indices):
        prev_ang = prev_val = None
        for i in indices:
            v = float(pattern_db[i])
            if v <= target:
                if prev_ang is None:
                    return float(angles_deg[i])
                t = (target - prev_val) / (v - prev_val) if v != prev_val else 0.0
                return float(prev_ang + t * (float(angles_deg[i]) - prev_ang))
            prev_ang = float(angles_deg[i])
            prev_val = v
        return float("nan")

    left = _cross(range(center, -1, -1))
    right = _cross(range(center, pattern_db.size))
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(right - left)
