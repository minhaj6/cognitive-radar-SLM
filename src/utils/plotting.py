from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Default to a non-interactive backend until show_all() is called. This lets
# headless runs save figures without needing a display server.
matplotlib.use("Agg", force=False)


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


def save_beampattern(
    run_dir: Path,
    name: str,
    angles_deg: np.ndarray,
    pattern_db: np.ndarray,
    *,
    title: str | None = None,
    target_deg: float | None = None,
    jammer_deg: float | None = None,
    dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(angles_deg, pattern_db, linewidth=1.2)
    if target_deg is not None:
        ax.axvline(target_deg, color="tab:green", linestyle="--", alpha=0.7, label=f"target {target_deg:g} deg")
    if jammer_deg is not None:
        ax.axvline(jammer_deg, color="tab:red", linestyle="--", alpha=0.7, label=f"jammer {jammer_deg:g} deg")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title or name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(bottom=max(-80.0, float(np.nanmin(pattern_db)) - 5.0), top=5.0)
    if target_deg is not None or jammer_deg is not None:
        ax.legend(loc="best")
    fig.tight_layout()

    out = run_dir / "plots" / f"{_slugify(name)}.png"
    fig.savefig(out, dpi=dpi)
    return out


def save_spectrum(
    run_dir: Path,
    name: str,
    angles_deg: np.ndarray,
    spectrum_db: np.ndarray,
    *,
    title: str | None = None,
    peaks_deg: list[float] | None = None,
    sparse_floor: float | None = None,
    dpi: int = 150,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if sparse_floor is None:
        # Continuous pseudo-spectrum (MUSIC-style).
        ax.plot(angles_deg, spectrum_db, linewidth=1.2)
        for peak in peaks_deg or []:
            ax.axvline(peak, color="tab:orange", linestyle=":", alpha=0.8)
    else:
        mask = spectrum_db > sparse_floor
        markerline, stemlines, _ = ax.stem(
            angles_deg[mask], spectrum_db[mask],
            bottom=sparse_floor,
            basefmt=" ", linefmt="tab:red", markerfmt="o",
        )
        plt.setp(stemlines, linewidth=1.5)
        plt.setp(markerline, color="tab:red", markersize=6)
        ax.set_ylim(sparse_floor, 5.0)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Pseudo-spectrum (dB)")
    ax.set_title(title or name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90.0, 90.0)
    fig.tight_layout()

    out = run_dir / "plots" / f"{_slugify(name)}.png"
    fig.savefig(out, dpi=dpi)
    return out


def show_all() -> None:
    if not plt.get_fignums():
        return
    # Re-render with an interactive backend. TkAgg is the common default on Linux.
    for backend in ("TkAgg", "QtAgg", "GTK3Agg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
    plt.show()
