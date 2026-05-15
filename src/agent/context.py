"""Shared run state that tool handlers close over.

The LLM is stateless between turns 
(simulate -> look at covariance -> design weights)

LLM hanlder reads and writes a single ``RunContext`` object that lives for
the duration of one CLI invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunContext:
    run_dir: Path
    cfg: dict
    N: int
    d: float
    angle_grid: np.ndarray
    dpi: int = 150

    # Populated by ``simulate_ula``.
    X: np.ndarray | None = None
    R: np.ndarray | None = None  # full data covariance R_xx (used by MPDR)
    X_in: np.ndarray | None = None
    R_in: np.ndarray | None = None  # interference+noise covariance (used by MVDR)
    soi_index: int | None = None
    truth_angles_deg: list[float] | None = None
    truth_powers_db: list[float] | None = None
    snr_db: float | None = None
    snapshots: int | None = None
