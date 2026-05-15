"""Tool (function-calling) specifications and handler factory.

* ``TOOL_SPECS``      -- the JSON schemas surfaced to the LLM. Each entry is
                        what the model actually sees in its ``tools`` argument.
* ``build_tool_registry(ctx)`` -- returns ``{name: callable}``. Each callable
                        closes over the shared :class:`RunContext` so it can
                        read/write simulated data and save plots.

Keep the two in sync: every entry in ``TOOL_SPECS`` must appear as a key in
the dict returned by ``build_tool_registry``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..physics import analysis, beamforming, doa, simulator
from ..utils import save_beampattern, save_spectrum
from .context import RunContext


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "simulate_ula",
            "description": (
                "Generate narrowband ULA snapshot data and populate the shared "
                "sample covariance R consumed by every other tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_elements": {"type": "integer"},
                    "spacing": {"type": "number", "description": "Element spacing in wavelengths (0.5 = lambda/2)."},
                    "source_angles_deg": {"type": "array", "items": {"type": "number"}},
                    "source_powers_db": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Per-source power offsets in dB relative to SNR. Default 0 dB each.",
                    },
                    "snapshots": {"type": "integer"},
                    "snr_db": {"type": "number"},
                    "coherent": {
                        "type": "boolean",
                        "description": "True if sources are perfectly correlated (e.g. multipath). Default false.",
                    },
                    "soi_index": {
                        "type": "integer",
                        "description": "0-based index of the desired signal in source_angles_deg. When set, also produces a parallel signal-free training covariance R_in built from the remaining sources.",
                    },
                },
                "required": ["n_elements", "spacing", "source_angles_deg", "snapshots", "snr_db"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beamform_das",
            "description": (
                "Conventional delay-and-sum beamformer with optional taper. "
                "No covariance required. Returns achieved SLL, 3 dB beamwidth, "
                "Rayleigh limit, and the saved plot path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steer_deg": {"type": "number"},
                    "taper": {
                        "type": "string",
                        "enum": ["uniform", "chebychev", "taylor"],
                        "description": "uniform: narrowest beam, ~-13 dB SLL ceiling. chebychev: equiripple SLL at the requested level, narrowest beam. taylor: quasi-equiripple near-in lobes, decaying far lobes.",
                    },
                    "sll_db": {
                        "type": "number",
                        "description": "Required sidelobe attenuation in dB (positive, e.g. 30 => -30 dB). Required for chebychev/taylor.",
                    },
                },
                "required": ["steer_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beamform_mvdr",
            "description": (
                "MVDR beamformer applied to the interference-plus-noise "
                "covariance R_in. Requires simulate_ula to have been called "
                "with soi_index set so that R_in is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steer_deg": {"type": "number"},
                    "diag_load": {
                        "type": "number",
                        "description": "Tikhonov regularization as a fraction of trace(R)/N (default 1e-3).",
                    },
                },
                "required": ["steer_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beamform_mpdr",
            "description": (
                "MPDR (Capon) beamformer applied to the full data covariance "
                "R_xx (which includes the signal of interest). Requires "
                "simulate_ula."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steer_deg": {"type": "number"},
                    "diag_load": {
                        "type": "number",
                        "description": "Tikhonov regularization as a fraction of trace(R)/N (default 1e-3).",
                    },
                },
                "required": ["steer_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "beamform_lcmv",
            "description": (
                "Linearly Constrained Minimum Variance beamformer. Builds a "
                "constraint matrix with unity gain at look_deg and zero gain at "
                "every angle in null_degs, then minimizes output power subject "
                "to those equalities. Prefers R_in when simulate_ula was called "
                "with soi_index; otherwise falls back to R_xx."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "look_deg": {
                        "type": "number",
                        "description": "Target angle where unity distortionless gain is enforced.",
                    },
                    "null_degs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Angles where the response is constrained to zero. One or more.",
                    },
                    "diag_load": {
                        "type": "number",
                        "description": "Tikhonov regularization as a fraction of trace(R)/N (default 1e-3).",
                    },
                },
                "required": ["look_deg", "null_degs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_doa_music",
            "description": (
                "MUSIC pseudo-spectrum and peak picking on the sample "
                "covariance. With apply_spatial_smoothing=true, performs "
                "forward-backward smoothing before the subspace decomposition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_sources": {"type": "integer"},
                    "apply_spatial_smoothing": {
                        "type": "boolean",
                        "description": "Apply forward-backward spatial smoothing before the subspace decomposition.",
                    },
                    "subarray_size": {
                        "type": "integer",
                        "description": "Subarray length for smoothing (default N-K).",
                    },
                },
                "required": ["n_sources"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_doa_esprit",
            "description": (
                "TLS-ESPRIT on a ULA. Closed-form DOA estimates from the "
                "rotational-invariance property; no angular grid search."
            ),
            "parameters": {
                "type": "object",
                "properties": {"n_sources": {"type": "integer"}},
                "required": ["n_sources"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_doa_omp",
            "description": (
                "Orthogonal Matching Pursuit on an angular dictionary. "
                "Recovers K source angles under a K-sparse model; works "
                "with as few as L=1 snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sparsity": {"type": "integer"}},
                "required": ["sparsity"],
            },
        },
    },
]


def build_tool_registry(ctx: RunContext) -> dict[str, Callable[..., Any]]:
    """Return a name -> handler mapping. Every handler returns a JSON-safe dict."""

    def _rel(p: Path) -> str:
        return str(p.relative_to(ctx.run_dir))

    def _require_R(name: str):
        if ctx.R is None:
            return {"error": f"{name}: no covariance available. Call simulate_ula first."}
        return None

    def simulate_ula(
        n_elements,
        spacing,
        source_angles_deg,
        snapshots,
        snr_db,
        source_powers_db=None,
        coherent=False,
        soi_index=None,
    ):
        ctx.N = int(n_elements)
        ctx.d = float(spacing)
        ctx.snapshots = int(snapshots)
        ctx.snr_db = float(snr_db)
        ctx.truth_angles_deg = [float(a) for a in source_angles_deg]
        ctx.truth_powers_db = (
            [float(p) for p in source_powers_db]
            if source_powers_db is not None
            else [0.0] * len(ctx.truth_angles_deg)
        )
        X, R = simulator.generate_ula_data(
            ctx.N,
            ctx.d,
            ctx.truth_angles_deg,
            ctx.truth_powers_db,
            ctx.snapshots,
            ctx.snr_db,
            coherent=bool(coherent),
        )
        ctx.X, ctx.R = X, R

        # Optional: parallel signal-free training block (no SOI) for MVDR.
        ctx.X_in, ctx.R_in, ctx.soi_index = None, None, None
        if soi_index is not None:
            k = int(soi_index)
            if not (0 <= k < len(ctx.truth_angles_deg)):
                return {"error": f"soi_index {k} out of range for {len(ctx.truth_angles_deg)} sources"}
            ctx.soi_index = k
            angles_in = [a for i, a in enumerate(ctx.truth_angles_deg) if i != k]
            powers_in = [p for i, p in enumerate(ctx.truth_powers_db) if i != k]
            X_in, R_in = simulator.generate_ula_data(
                ctx.N,
                ctx.d,
                angles_in,
                powers_in,
                ctx.snapshots,
                ctx.snr_db,
                coherent=bool(coherent),
            )
            ctx.X_in, ctx.R_in = X_in, R_in

        eigs = np.sort(np.linalg.eigvalsh(R))[::-1]
        return {
            "ok": True,
            "n_elements": ctx.N,
            "spacing_wavelengths": ctx.d,
            "snapshots": ctx.snapshots,
            "snr_db": ctx.snr_db,
            "n_sources_truth": len(ctx.truth_angles_deg),
            "coherent": bool(coherent),
            "soi_index": ctx.soi_index,
            "has_signal_free_training": ctx.R_in is not None,
            "eigenspectrum_top_db": [
                round(float(10 * np.log10(max(e, 1e-12))), 2) for e in eigs[: min(5, eigs.size)]
            ],
            "rayleigh_resolution_deg": round(analysis.rayleigh_resolution_deg(ctx.N, ctx.d, 0.0), 3),
        }

    def beamform_das(steer_deg, taper="uniform", sll_db=None):
        steer_deg = float(steer_deg)
        taper = str(taper).lower()
        try:
            if taper == "uniform":
                w = beamforming.delay_and_sum(ctx.N, ctx.d, steer_deg)
                requested = None
            elif taper == "chebychev":
                sll = float(sll_db or 30.0)
                taper_w = beamforming.chebychev_weights(ctx.N, sll)
                w = taper_w * simulator.steering_vector_ula(ctx.N, ctx.d, steer_deg) / ctx.N
                requested = sll
            elif taper == "taylor":
                sll = float(sll_db or 30.0)
                taper_w = beamforming.taylor_weights(ctx.N, nbar=4, sll_db=sll)
                w = taper_w * simulator.steering_vector_ula(ctx.N, ctx.d, steer_deg) / ctx.N
                requested = sll
            else:
                return {"error": f"unknown taper '{taper}'. Use uniform, chebychev, or taylor."}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        p_db = beamforming.beampattern(w, ctx.N, ctx.d, ctx.angle_grid)
        sll_meas = analysis.sidelobe_level_db(p_db, ctx.angle_grid, steer_deg)
        bw_3db = analysis.mainlobe_3db_width_deg(p_db, ctx.angle_grid, steer_deg)
        rayleigh = analysis.rayleigh_resolution_deg(ctx.N, ctx.d, steer_deg)
        plot = save_beampattern(
            ctx.run_dir,
            f"das-{taper}-steer-{steer_deg:+g}",
            ctx.angle_grid,
            p_db,
            title=f"DAS ({taper}), steer = {steer_deg} deg",
            target_deg=steer_deg,
            dpi=ctx.dpi,
        )
        return {
            "algorithm": "das",
            "taper": taper,
            "steer_deg": steer_deg,
            "requested_sll_db": requested,
            "achieved_sll_db": round(float(sll_meas), 2),
            "mainlobe_3db_width_deg": round(float(bw_3db), 3),
            "rayleigh_resolution_deg": round(float(rayleigh), 3),
            "plot": _rel(plot),
        }

    def _capon_report(algorithm, R_used, steer_deg, diag_load):
        steer_deg = float(steer_deg)
        weight_fn = beamforming.mvdr if algorithm == "mvdr" else beamforming.mpdr
        try:
            w = weight_fn(R_used, ctx.N, ctx.d, steer_deg, diag_load=float(diag_load))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        p_db = beamforming.beampattern(w, ctx.N, ctx.d, ctx.angle_grid)
        sll_meas = analysis.sidelobe_level_db(p_db, ctx.angle_grid, steer_deg)
        bw_3db = analysis.mainlobe_3db_width_deg(p_db, ctx.angle_grid, steer_deg)

        jammer_deg = None
        if ctx.truth_angles_deg and ctx.truth_powers_db:
            order = sorted(
                zip(ctx.truth_powers_db, ctx.truth_angles_deg), reverse=True
            )
            for _, ang in order:
                if abs(ang - steer_deg) > max(1.5 * bw_3db, 3.0):
                    jammer_deg = ang
                    break

        target_gain = float(p_db[int(np.argmin(np.abs(ctx.angle_grid - steer_deg)))])
        null_gain = (
            float(p_db[int(np.argmin(np.abs(ctx.angle_grid - jammer_deg)))])
            if jammer_deg is not None
            else None
        )
        label = algorithm.upper()
        plot = save_beampattern(
            ctx.run_dir,
            f"{algorithm}-steer-{steer_deg:+g}",
            ctx.angle_grid,
            p_db,
            title=f"{label}, steer = {steer_deg} deg",
            target_deg=steer_deg,
            jammer_deg=jammer_deg,
            dpi=ctx.dpi,
        )
        return {
            "algorithm": algorithm,
            "steer_deg": steer_deg,
            "diag_load": float(diag_load),
            "target_gain_db": round(target_gain, 2),
            "interferer_deg": jammer_deg,
            "interferer_gain_db": round(null_gain, 2) if null_gain is not None else None,
            "peak_sll_db": round(float(sll_meas), 2),
            "mainlobe_3db_width_deg": round(float(bw_3db), 3),
            "plot": _rel(plot),
        }

    def beamform_mvdr(steer_deg, diag_load=1e-3):
        if ctx.R_in is None:
            return {
                "error": (
                    "beamform_mvdr: no signal-free training covariance available. "
                    "Re-run simulate_ula with soi_index set, or use beamform_mpdr "
                    "if you only have R_xx."
                )
            }
        return _capon_report("mvdr", ctx.R_in, steer_deg, diag_load)

    def beamform_mpdr(steer_deg, diag_load=1e-3):
        err = _require_R("beamform_mpdr")
        if err:
            return err
        return _capon_report("mpdr", ctx.R, steer_deg, diag_load)

    def beamform_lcmv(look_deg, null_degs, diag_load=1e-3):
        err = _require_R("beamform_lcmv")
        if err:
            return err
        try:
            look = float(look_deg)
            nulls = [float(a) for a in (null_degs or [])]
        except (TypeError, ValueError) as exc:
            return {"error": f"beamform_lcmv: bad arguments ({exc})"}
        if not nulls:
            return {"error": "beamform_lcmv: null_degs must contain at least one angle. Use beamform_mvdr/mpdr if no explicit nulls are required."}

        # Prefer signal-free training covariance when available.
        R_used = ctx.R_in if ctx.R_in is not None else ctx.R
        covariance_used = "R_in" if ctx.R_in is not None else "R_xx"

        # Build constraint matrix C = [a(look), a(null_1), ..., a(null_M)] with f = [1, 0, ..., 0].
        a_look = simulator.steering_vector_ula(ctx.N, ctx.d, look)
        a_nulls = [simulator.steering_vector_ula(ctx.N, ctx.d, ang) for ang in nulls]
        C = np.column_stack([a_look, *a_nulls])
        f = np.zeros(C.shape[1], dtype=complex)
        f[0] = 1.0

        try:
            w = beamforming.lcmv(R_used, C, f, diag_load=float(diag_load))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        p_db = beamforming.beampattern(w, ctx.N, ctx.d, ctx.angle_grid)
        sll_meas = analysis.sidelobe_level_db(p_db, ctx.angle_grid, look)
        bw_3db = analysis.mainlobe_3db_width_deg(p_db, ctx.angle_grid, look)

        def _gain_at(angle: float) -> float:
            return float(p_db[int(np.argmin(np.abs(ctx.angle_grid - angle)))])

        target_gain = _gain_at(look)
        null_gains = {f"{ang:+g}": round(_gain_at(ang), 2) for ang in nulls}

        # Sanity check: constraints should be satisfied exactly (before discretization).
        constraint_residual = float(np.max(np.abs(C.conj().T @ w - f)))

        plot = save_beampattern(
            ctx.run_dir,
            f"lcmv-look-{look:+g}-nulls-{'_'.join(f'{a:+g}' for a in nulls)}",
            ctx.angle_grid,
            p_db,
            title=f"LCMV, look = {look} deg, nulls = {nulls} deg",
            target_deg=look,
            dpi=ctx.dpi,
        )
        return {
            "algorithm": "lcmv",
            "look_deg": look,
            "null_degs": nulls,
            "diag_load": float(diag_load),
            "covariance_used": covariance_used,
            "target_gain_db": round(target_gain, 2),
            "null_gain_db_at": null_gains,
            "peak_sll_db": round(float(sll_meas), 2),
            "mainlobe_3db_width_deg": round(float(bw_3db), 3),
            "constraint_residual": round(constraint_residual, 6),
            "plot": _rel(plot),
        }

    def estimate_doa_music(n_sources, apply_spatial_smoothing=False, subarray_size=None):
        err = _require_R("estimate_doa_music")
        if err:
            return err
        K = int(n_sources)
        if apply_spatial_smoothing:
            L = int(subarray_size) if subarray_size else max(K + 1, ctx.N - K)
            if L >= ctx.N:
                L = ctx.N - 1
            R_used = doa.spatial_smoothing(ctx.R, L, forward_backward=True)
            N_used = L
            note = f"forward-backward spatial smoothing, L={L}"
        else:
            R_used = ctx.R
            N_used = ctx.N
            note = "no smoothing"
        if K >= N_used:
            return {"error": f"MUSIC requires K < N_effective ({K} >= {N_used})"}
        spec_db, doas = doa.music(R_used, K, N_used, ctx.d, ctx.angle_grid)
        plot = save_spectrum(
            ctx.run_dir,
            f"music-K{K}{'-ss' if apply_spatial_smoothing else ''}",
            ctx.angle_grid,
            spec_db,
            title=f"MUSIC (K={K}, {note})",
            peaks_deg=[float(x) for x in doas],
            dpi=ctx.dpi,
        )
        return {
            "algorithm": "music",
            "n_sources": K,
            "spatial_smoothing": bool(apply_spatial_smoothing),
            "subarray_size": N_used if apply_spatial_smoothing else None,
            "estimated_doas_deg": [round(float(x), 3) for x in doas],
            "plot": _rel(plot),
        }

    def estimate_doa_esprit(n_sources):
        if ctx.X is None and ctx.R is None:
            return {"error": "estimate_doa_esprit: no data available. Call simulate_ula first."}
        data = ctx.X if ctx.X is not None else ctx.R
        K = int(n_sources)
        try:
            doas = doa.esprit(data, K, ctx.d)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        # ESPRIT is closed-form so there is no native spectrum, but render the
        # estimates as stems on the angle axis so the run still produces a plot.
        floor = -60.0
        spec_db = np.full(ctx.angle_grid.size, -120.0)
        for ang in doas:
            idx = int(np.argmin(np.abs(ctx.angle_grid - float(ang))))
            spec_db[idx] = 0.0
        plot = save_spectrum(
            ctx.run_dir,
            f"esprit-K{K}",
            ctx.angle_grid,
            spec_db,
            title=f"ESPRIT (K={K}) -- estimated DOAs",
            sparse_floor=floor,
            dpi=ctx.dpi,
        )
        return {
            "algorithm": "esprit",
            "n_sources": K,
            "estimated_doas_deg": [round(float(x), 3) for x in doas],
            "plot": _rel(plot),
        }

    def estimate_doa_omp(sparsity):
        if ctx.X is None:
            return {"error": "estimate_doa_omp: no snapshot data. Call simulate_ula first."}
        K = int(sparsity)
        coarse = np.arange(-90.0, 90.0 + 1e-9, 0.25)
        try:
            spec_db, doas = doa.omp(ctx.X, ctx.N, ctx.d, coarse, K)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        plot = save_spectrum(
            ctx.run_dir,
            f"omp-K{K}",
            coarse,
            spec_db,
            title=f"OMP (K={K})",
            peaks_deg=[float(x) for x in doas],
            sparse_floor=-60.0,
            dpi=ctx.dpi,
        )
        return {
            "algorithm": "omp",
            "sparsity": K,
            "estimated_doas_deg": [round(float(x), 3) for x in doas],
            "plot": _rel(plot),
        }

    return {
        "simulate_ula": simulate_ula,
        "beamform_das": beamform_das,
        "beamform_mvdr": beamform_mvdr,
        "beamform_mpdr": beamform_mpdr,
        "beamform_lcmv": beamform_lcmv,
        "estimate_doa_music": estimate_doa_music,
        "estimate_doa_esprit": estimate_doa_esprit,
        "estimate_doa_omp": estimate_doa_omp,
    }
