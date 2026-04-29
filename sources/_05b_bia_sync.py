# _05b_bia_sync.py - BIA SYNCHRONIZATION
# Estimate and apply the temporal lag between BIA and the master EMG-torque timeline.
#
# BIA is recorded by the LIRMM prototype device on its own internal clock, independent
# of the Delsys system clock used by EMG and torque. Even though _05a_ converts BIA
# timestamps to session seconds via ts_ref, small clock drift between devices means
# the BIA signal is still slightly offset from the master grid. This step finds that
# offset (the "lag") and corrects it by shifting every BIA time_index by the same
# integer number of master grid samples.
#
# Lag estimation algorithm
#   1. Build a grid of candidate lags from lag_min_s to lag_max_s in LAG_STEP_S steps.
#   2. For each candidate, shift BIA time_index values and restrict to the anchor
#      sequence (MVC_REF). MVC_REF is chosen because it contains well-defined VC events
#      with a strong |Z| response: impedance magnitude rises clearly during maximal
#      contraction, making VC-in vs. VC-out contrast the most reliable sync signal.
#   3. Score the candidate as abs(mean |Z| inside VC – mean |Z| outside VC).
#      e.g. mean |Z| in VC = 380 Ω, out VC = 310 Ω → score = 70 Ω.
#      The true lag maximises this contrast.
#   4. Take the candidate with the highest score as the automatic estimate.
#   5. Add optional manual_nudge_s for fine adjustment after visual inspection.
#   6. Apply the total shift to both bia2 and bia4; samples pushed outside
#      [0, master_n) are dropped. SEQ, VC, and VC_count are re-stamped from the
#      shifted time_index positions.
#
# Scoring uses 4PT impedance (bia4) because it cancels electrode contact impedance
# and gives a cleaner |Z| signal. The same integer-sample shift is applied to 2PT
# (bia2) because both configurations are recorded simultaneously by the same device.
#
# Inputs
#   ctx keys  : CACHE_DIR, master_index_grid (with SEQ, VC, VC_count),
#               bia2_compact_df, bia4_compact_df, bia4_freqs_hz, torque_compact_df
#   parameters: lag_min_s, lag_max_s — search window bounds (session seconds)
#               manual_nudge_s      — extra shift added after visual inspection (default 0)
#               torque_col          — which torque column to plot (default "torque_raw")
#               force_recompute     — bypass cache and re-run the lag search
#
# Outputs (also written to ctx)
#   - bia2_compact_aligned_df : 2PT table with shifted time_index and refreshed SEQ/VC labels
#   - bia4_compact_aligned_df : 4PT table (same layout)
#
# Cache
#   - 05b_bia2_compact_aligned.parquet
#   - 05b_bia4_compact_aligned.parquet

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Constants ---

# Step size of the lag candidate grid in seconds.
# 0.05 s is well below the BIA sampling interval (~0.1 s), giving sub-sample resolution
# without excessive iterations. Finer steps improve score smoothness but rarely change
# the outcome since the score curve is broad relative to LAG_STEP_S.
LAG_STEP_S = 0.05

# Protocol sequence used as the anchor window for lag scoring.
# MVC_REF is chosen because its repeated maximal contractions produce the largest |Z|
# swings in the dataset, making the VC-in vs. VC-out contrast robust to noise.
# Changing this requires re-validating the lag search across all subjects.
ANCHOR_SEQ = "MVC_REF"

# BIA frequency bin used for lag scoring; must be present in bia4_freqs_hz.
# 9760 Hz is the highest frequency recorded by the LIRMM device. High-frequency
# impedance reflects primarily resistive tissue properties with minimal capacitive
# contribution, giving the most stable |Z| baseline and the clearest contraction signal.
TARGET_HZ = 9760


def run_bia_sync(
    *,
    ctx: dict,
    lag_min_s: float,
    lag_max_s: float,
    manual_nudge_s: float = 0.0,
    torque_col: str = "torque_raw",
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Find and apply the temporal lag between BIA and the master EMG-torque timeline.

    The lag is estimated by scanning candidate shifts and scoring each using the |Z|
    contrast between VC and non-VC samples within the MVC_REF window. An optional
    manual_nudge_s can be added after visual inspection of the QC plot.

    The same shift is applied to both bia2 and bia4 (both were recorded simultaneously
    by the same device). Samples shifted outside the master grid bounds are dropped.
    SEQ, VC, and VC_count columns are re-stamped from the shifted time_index.

    Returns (bia2_compact_aligned_df, bia4_compact_aligned_df, fig).
    fig is None on a cache hit (no plot is redrawn from cache).
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    bia2_compact_df   = ctx["bia2_compact_df"]
    bia4_compact_df   = ctx["bia4_compact_df"]
    bia4_freqs_hz     = np.asarray(ctx["bia4_freqs_hz"], dtype=float)
    torque_compact_df = ctx["torque_compact_df"]

    # --- Cache paths ---
    CACHE_BIA2_ALIGNED = cache_dir / "05b_bia2_compact_aligned.parquet"
    CACHE_BIA4_ALIGNED = cache_dir / "05b_bia4_compact_aligned.parquet"

    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_BIA_2_ALIGNED"] = CACHE_BIA2_ALIGNED
    ctx["parquet_path"]["CACHE_BIA_4_ALIGNED"] = CACHE_BIA4_ALIGNED

    # --- Cache check ---
    if CACHE_BIA2_ALIGNED.exists() and CACHE_BIA4_ALIGNED.exists() and not force_recompute:
        bia2_compact_aligned_df = pd.read_parquet(CACHE_BIA2_ALIGNED)
        bia4_compact_aligned_df = pd.read_parquet(CACHE_BIA4_ALIGNED)
        ctx["bia2_compact_aligned_df"] = bia2_compact_aligned_df
        ctx["bia4_compact_aligned_df"] = bia4_compact_aligned_df
        print(
            "[05b_bia_sync] Cache hit — loaded aligned BIA from cache.\n"
            "  Set force_recompute=True to re-run the lag search."
        )
        return bia2_compact_aligned_df, bia4_compact_aligned_df, None

    # --- Master grid arrays ---
    # Extracted once so the lag-scoring loop can index into them cheaply.
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    # Median inter-sample interval is used to estimate sampling rate because occasional
    # large between-sequence gaps would inflate the mean and distort the Hz estimate.
    master_fs_hz      = 1.0 / float(np.median(np.diff(master_time_ref_s)))
    master_n          = len(master_index_grid)
    master_seq        = master_index_grid["SEQ"].to_numpy()
    master_seq_index  = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc         = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count   = master_index_grid["VC_count"].to_numpy(dtype=int)

    # --- Anchor window mask ---
    # Boolean mask over the full master grid selecting only ANCHOR_SEQ samples.
    # The lag search evaluates only BIA samples that fall inside this window after
    # shifting, so a longer anchor sequence → more BIA samples scored → more reliable result.
    mvc_mask_master = (master_seq == ANCHOR_SEQ)

    # --- Frequency column names ---
    # Confirm TARGET_HZ exists in the BIA data before building column names.
    # The explicit check here gives a clearer message than the KeyError pandas would
    # raise when accessing the constructed column name.
    freq_match_idx = np.where(bia4_freqs_hz == TARGET_HZ)[0]
    if freq_match_idx.size == 0:
        raise ValueError(
            f"[05b_bia_sync] TARGET_HZ={TARGET_HZ} not found in bia4_freqs_hz: "
            f"{bia4_freqs_hz.tolist()}"
        )
    freq_tag    = str(int(TARGET_HZ))
    bia4_R_col  = f"bia4_R_ohm__f_{freq_tag}Hz"
    bia4_Xc_col = f"bia4_Xc_ohm__f_{freq_tag}Hz"
    bia2_R_col  = f"bia2_R_ohm__f_{freq_tag}Hz"
    bia2_Xc_col = f"bia2_Xc_ohm__f_{freq_tag}Hz"

    # --- Build bia4 |Z| ---
    bia4_time_index_raw = bia4_compact_df["time_index"].to_numpy(dtype=int)
    bia4_R              = bia4_compact_df[bia4_R_col].to_numpy(dtype=float)
    bia4_Xc             = bia4_compact_df[bia4_Xc_col].to_numpy(dtype=float)
    # |Z| = sqrt(R² + Xc²): impedance magnitude. np.hypot is numerically safer than
    # manual sqrt(R**2 + Xc**2) for very large or very small values.
    bia4_absZ           = np.hypot(bia4_R, bia4_Xc)

    # --- Lag search ---
    # Convert candidate lag times to integer sample offsets via np.rint so each shift
    # is a whole number of master grid samples.
    # e.g. lag=0.50 s at 2148 Hz → shift = round(0.50 × 2148) = 1074 samples.
    lag_candidates_s         = np.arange(float(lag_min_s), float(lag_max_s) + 1e-12, LAG_STEP_S)
    if lag_candidates_s.size < 3:
        raise ValueError(
            f"[05b_bia_sync] Lag grid has fewer than 3 candidates — "
            f"check lag_min_s={lag_min_s}, lag_max_s={lag_max_s}, LAG_STEP_S={LAG_STEP_S}."
        )
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

    scores = np.full(lag_candidates_s.shape, np.nan, dtype=float)

    for lag_idx, shift_samples in enumerate(shift_candidates_samples):
        bia4_time_shifted = bia4_time_index_raw + shift_samples

        # Drop BIA samples that fall outside the master grid after shifting.
        # This happens near the edges of the search window and is expected.
        valid_shifted = (bia4_time_shifted >= 0) & (bia4_time_shifted < master_n)
        if np.sum(valid_shifted) < 10:
            continue

        shifted_valid   = bia4_time_shifted[valid_shifted].astype(int, copy=False)
        absZ_valid      = bia4_absZ[valid_shifted]

        in_anchor       = mvc_mask_master[shifted_valid]
        if np.sum(in_anchor) < 10:
            continue

        anchor_indices  = shifted_valid[in_anchor]
        anchor_absZ     = absZ_valid[in_anchor]
        anchor_vc_count = master_vc_count[anchor_indices]

        in_vc  = (anchor_vc_count > 0)
        out_vc = ~in_vc

        if np.sum(in_vc) < 3 or np.sum(out_vc) < 3:
            continue

        # Score = |mean(|Z|_in_VC) – mean(|Z|_out_VC)|.
        # The lag that maximises this contrast is the one where BIA events align best
        # with the torque-derived VC labels.
        scores[lag_idx] = abs(
            float(np.nanmean(anchor_absZ[in_vc])) - float(np.nanmean(anchor_absZ[out_vc]))
        )

    if not np.any(np.isfinite(scores)):
        raise ValueError(
            "[05b_bia_sync] All lag scores are NaN — check anchor SEQ labeling, "
            "VC detection, lag search window, and BIA-to-master mapping."
        )

    # nanargmax is used because some score entries may be NaN (skipped candidates).
    best_score_idx      = int(np.nanargmax(scores))
    best_lag_s_auto     = float(lag_candidates_s[best_score_idx])
    best_shift_samples  = int(shift_candidates_samples[best_score_idx])

    total_lag_s         = best_lag_s_auto + manual_nudge_s
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))

    # --- Apply shift ---

    def _shift_drop_relabel(compact_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Shift time_index by total_shift_samples, drop samples outside [0, master_n),
        and re-stamp SEQ, SEQ_index, VC, VC_count from the master grid.

        Samples at the edges of the recording are dropped when the lag shifts them
        before session start or after session end — they have no master context to
        label against, so keeping them would produce invalid SEQ/VC values.

        Returns (aligned_df, stats_dict) counting total / kept / dropped samples.
        """
        time_index_raw     = compact_df["time_index"].to_numpy(dtype=int)
        time_index_shifted = time_index_raw + total_shift_samples

        valid   = (time_index_shifted >= 0) & (time_index_shifted < master_n)
        n_valid = int(np.sum(valid))

        if n_valid == 0:
            raise ValueError(
                "[05b_bia_sync] Shift produced zero valid samples — "
                "check lag sign, search window, and manual nudge."
            )

        df_out           = compact_df.loc[valid].copy().reset_index(drop=True)
        time_index_valid = time_index_shifted[valid].astype(int, copy=False)

        df_out["time_index"] = time_index_valid
        df_out["SEQ_index"]  = master_seq_index[time_index_valid]
        df_out["SEQ"]        = master_seq[time_index_valid]
        df_out["VC"]         = master_vc[time_index_valid]
        df_out["VC_count"]   = master_vc_count[time_index_valid]

        df_out = df_out.sort_values("time_index", kind="mergesort").reset_index(drop=True)

        stats = {
            "n_total":     time_index_shifted.size,
            "n_valid":     n_valid,
            "n_drop_low":  int(np.sum(time_index_shifted < 0)),
            "n_drop_high": int(np.sum(time_index_shifted >= master_n)),
        }
        return df_out, stats

    bia2_compact_aligned_df, bia2_stats = _shift_drop_relabel(bia2_compact_df)
    bia4_compact_aligned_df, bia4_stats = _shift_drop_relabel(bia4_compact_df)

    # --- Write cache ---
    bia2_compact_aligned_df.to_parquet(CACHE_BIA2_ALIGNED, index=False)
    bia4_compact_aligned_df.to_parquet(CACHE_BIA4_ALIGNED, index=False)

    print(
        f"[05b_bia_sync] Done — auto lag={best_lag_s_auto:+.3f}s, "
        f"nudge={manual_nudge_s:+.3f}s, total={total_lag_s:+.3f}s "
        f"({total_shift_samples:+d} samples at {master_fs_hz:.0f} Hz)\n"
        f"  bia2: {bia2_stats['n_valid']}/{bia2_stats['n_total']} samples kept "
        f"(dropped {bia2_stats['n_drop_low']} low, {bia2_stats['n_drop_high']} high)\n"
        f"  bia4: {bia4_stats['n_valid']}/{bia4_stats['n_total']} samples kept "
        f"(dropped {bia4_stats['n_drop_low']} low, {bia4_stats['n_drop_high']} high)"
    )

    # --- QC plot ---
    # 2×2 layout: left column = before sync, right column = after sync.
    # Top row shows torque + VC borders in both columns (torque is not shifted;
    # it is identical left and right and serves as the alignment reference).
    # Bottom row shows |Z| at TARGET_HZ before and after the lag correction.
    # Orange vertical lines mark VC start/stop boundaries.
    # Correct alignment: |Z| peaks or drops should line up with the orange markers.
    plt.close("all")

    def _draw_vc_borders(ax):
        for s, e in zip(starts, stops):
            if s >= mvc_i0 and e <= mvc_i1:
                ax.axvline(s, color="orange", linewidth=2)
                ax.axvline(e, color="orange", linewidth=2)

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_values     = torque_compact_df[torque_col].to_numpy(dtype=float)

    mvc_idx = np.where(mvc_mask_master)[0]
    mvc_i0  = int(mvc_idx[0])
    mvc_i1  = int(mvc_idx[-1])

    # Build VC span edges within the anchor window for the orange marker lines.
    mask_vc_anchor = mvc_mask_master & (master_vc_count > 0)
    d      = np.diff(mask_vc_anchor.astype(int))
    starts = np.where(d ==  1)[0] + 1
    stops  = np.where(d == -1)[0]
    if mask_vc_anchor[0]:
        starts = np.r_[0, starts]
    if mask_vc_anchor[-1]:
        stops  = np.r_[stops, len(mask_vc_anchor) - 1]

    torque_in_anchor = (torque_time_index >= mvc_i0) & (torque_time_index <= mvc_i1)

    # Reuse already-extracted bia4 arrays to avoid a second DataFrame read.
    raw_in_anchor = (bia4_time_index_raw >= mvc_i0) & (bia4_time_index_raw <= mvc_i1)
    x_raw         = bia4_time_index_raw[raw_in_anchor]
    y_raw         = bia4_absZ[raw_in_anchor]
    # Sort by time_index so the line plot connects samples in chronological order.
    o = np.argsort(x_raw)
    x_raw, y_raw = x_raw[o], y_raw[o]

    bia4_aln_time_index = bia4_compact_aligned_df["time_index"].to_numpy(dtype=int)
    bia4_aln_absZ       = np.hypot(
        bia4_compact_aligned_df[bia4_R_col].to_numpy(dtype=float),
        bia4_compact_aligned_df[bia4_Xc_col].to_numpy(dtype=float),
    )
    aln_in_anchor = (bia4_aln_time_index >= mvc_i0) & (bia4_aln_time_index <= mvc_i1)
    x_aln         = bia4_aln_time_index[aln_in_anchor]
    y_aln         = bia4_aln_absZ[aln_in_anchor]
    o = np.argsort(x_aln)
    x_aln, y_aln = x_aln[o], y_aln[o]

    lag_txt = (
        f" (lag {total_lag_s:+.2f}s | auto {best_lag_s_auto:+.2f}s "
        f"+ nudge {manual_nudge_s:+.2f}s)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
    fig.suptitle(f"BIA sync QC — {run_id}", y=0.995)

    ax = axes[0, 0]
    ax.plot(torque_time_index[torque_in_anchor], torque_values[torque_in_anchor], alpha=0.85)
    _draw_vc_borders(ax)
    ax.set_title(f"BEFORE — {ANCHOR_SEQ} torque + VC borders")
    ax.set_ylabel("Torque")

    ax = axes[1, 0]
    ax.plot(x_raw, y_raw, linewidth=1.2, alpha=0.85)
    ax.scatter(x_raw, y_raw, s=14, alpha=0.85)
    _draw_vc_borders(ax)
    ax.set_title(f"|Z| bia4 @ {freq_tag} Hz — BEFORE (raw)")
    ax.set_ylabel("|Z| (Ω)")
    ax.set_xlabel("Master time_index")

    ax = axes[0, 1]
    ax.plot(torque_time_index[torque_in_anchor], torque_values[torque_in_anchor], alpha=0.85)
    _draw_vc_borders(ax)
    ax.set_title(f"AFTER — {ANCHOR_SEQ} torque + VC borders (reference, unchanged)")
    ax.set_ylabel("Torque")

    ax = axes[1, 1]
    ax.plot(x_aln, y_aln, linewidth=1.2, alpha=0.85)
    ax.scatter(x_aln, y_aln, s=14, alpha=0.85)
    _draw_vc_borders(ax)
    ax.set_title(f"|Z| bia4 @ {freq_tag} Hz — AFTER{lag_txt}")
    ax.set_ylabel("|Z| (Ω)")
    ax.set_xlabel("Master time_index")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # --- Update ctx ---
    ctx["bia2_compact_aligned_df"] = bia2_compact_aligned_df
    ctx["bia4_compact_aligned_df"] = bia4_compact_aligned_df

    return bia2_compact_aligned_df, bia4_compact_aligned_df, fig
