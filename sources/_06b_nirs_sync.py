# _06b_nirs_sync.py - NIRS SYNCHRONIZATION
# Estimate and apply the temporal lag between NIRS and the master EMG-torque timeline.
#
# NIRS is recorded by the PortaLite device on its own clock, independent of the Delsys
# system. After _06a_ converts NIRS timestamps to session seconds via ts_ref, small clock
# drift between devices means the NIRS signal is still slightly offset from the master
# grid. This step finds that offset (the "lag") and corrects it by shifting every NIRS
# time_index by the same integer number of master grid samples.
#
# Lag estimation algorithm
#   1. Build a grid of candidate lags from lag_min_s to lag_max_s in LAG_STEP_S steps.
#   2. For each candidate, shift NIRS time_index values and restrict to the anchor
#      sequence (MVC_REF). MVC_REF contains well-defined voluntary contractions that
#      produce a clear haemoglobin response, making it the most reliable sync window.
#   3. Within the anchor, z-score normalise the NIRS signal across all samples.
#   4. At each VC onset, compute a weighted slope of the z-score in a 1 s post-onset
#      window (NIRS_SLOPE_WINDOW_S). Weights decrease linearly (1.0 → 0.0) to emphasise
#      the initial slope change — the lag affects timing of the onset, not the plateau.
#   5. Score = median of per-onset slope features (median is robust to missed contractions
#      in MVC_REF where the subject did not reach target force).
#   6. The slope sign is handled for both signal directions:
#        HHb (preferred per fNIRS protocol): rises during contraction → positive slope
#        O2Hb: falls during contraction → code uses abs(dz) on negative slopes
#   7. Take the candidate with the maximum score as the automatic estimate.
#   8. Add optional manual_nudge_s for fine adjustment after visual inspection.
#   9. Apply total shift; drop samples outside [0, master_n); re-stamp SEQ/VC/VC_count.
#
# Inputs
#   ctx keys  : CACHE_DIR, master_index_grid (with SEQ, VC, VC_count),
#               nirs_compact_df, torque_compact_df
#   parameters: nirs_signal_col — which NIRS column to score (e.g. "nirs_hhb_tx1")
#               lag_min_s, lag_max_s — search window bounds (session seconds)
#               manual_nudge_s      — extra shift added after visual inspection (default 0)
#               torque_col          — torque column to plot (default "torque_raw")
#               force_recompute     — bypass cache and re-run the lag search
#
# Outputs (also written to ctx)
#   - nirs_compact_aligned_df : NIRS table with shifted time_index and refreshed SEQ/VC labels
#
# Cache
#   - 06b_nirs_compact_aligned.parquet

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Constants ---

# Protocol sequence used as the anchor window for lag scoring.
# MVC_REF is chosen because its maximal voluntary contractions produce the clearest
# haemoglobin response, making VC onset timing detectable even at low SNR.
ANCHOR_SEQ = "MVC_REF"

# Step size of the lag candidate grid in seconds.
# 0.05 s is well below the NIRS inter-sample interval (0.1 s at 10 Hz), providing
# sub-sample resolution without excessive iterations.
LAG_STEP_S = 0.05

# Duration of the post-onset window used to compute the slope score (seconds).
# 1 s is long enough to capture the initial haemoglobin change while staying within
# the typical contraction duration for MVC_REF (~3–5 s).
NIRS_SLOPE_WINDOW_S = 1.0

# Minimum number of samples required inside a post-onset window to compute a slope.
# Windows shorter than this are skipped; they arise at the edges of the anchor sequence.
NIRS_MIN_WINDOW_SAMPLES = 12


def run_nirs_sync(
    *,
    ctx: dict,
    nirs_signal_col: str,
    lag_min_s: float,
    lag_max_s: float,
    manual_nudge_s: float = 0.0,
    torque_col: str = "torque_raw",
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, Any]:
    """
    Find and apply the temporal lag between NIRS and the master EMG-torque timeline.

    The lag is estimated by scoring candidate shifts using the haemoglobin onset slope
    within MVC_REF. An optional manual_nudge_s can be added after visual inspection.
    The same shift is applied to the full NIRS DataFrame; out-of-bounds samples are
    dropped and SEQ/VC/VC_count are re-stamped from the shifted time_index.

    Returns (nirs_compact_aligned_df, fig). fig is None on a cache hit.
    """

    # --- Resolve ctx ---
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    nirs_compact_df   = ctx["nirs_compact_df"]
    torque_compact_df = ctx["torque_compact_df"]

    # --- Cache paths ---
    CACHE_NIRS_ALIGNED = cache_dir / "06b_nirs_compact_aligned.parquet"

    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_NIRS_ALIGNED"] = CACHE_NIRS_ALIGNED

    # --- Cache check ---
    if CACHE_NIRS_ALIGNED.exists() and not force_recompute:
        nirs_compact_aligned_df = pd.read_parquet(CACHE_NIRS_ALIGNED)
        ctx["nirs_compact_aligned_df"] = nirs_compact_aligned_df
        print(
            "[06b_nirs_sync] Cache hit — loaded aligned NIRS from cache.\n"
            "  Set force_recompute=True to re-run the lag search."
        )
        return nirs_compact_aligned_df, None

    # --- Master grid arrays ---
    # Extracted once; the scoring loop indexes into them on every candidate iteration.
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    # Median inter-sample interval for fs_hz — large between-sequence gaps inflate mean.
    master_fs_hz      = 1.0 / float(np.median(np.diff(master_time_ref_s)))
    master_n          = len(master_index_grid)
    master_seq        = master_index_grid["SEQ"].to_numpy()
    master_seq_index  = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc         = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count   = master_index_grid["VC_count"].to_numpy(dtype=int)

    # --- Anchor window mask ---
    master_anchor_mask = (master_seq == ANCHOR_SEQ)

    # --- Lag search setup ---
    # Convert candidate lag seconds to integer sample offsets.
    # e.g. lag=1.0 s at 2148 Hz → shift = round(1.0 × 2148) = 2148 samples.
    lag_candidates_s         = np.arange(float(lag_min_s), float(lag_max_s) + 1e-12, LAG_STEP_S)
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

    # Precompute per-sample window size for the slope scoring (in master grid samples).
    nirs_slope_window_samples = int(round(NIRS_SLOPE_WINDOW_S * master_fs_hz))

    nirs_time_index = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_signal     = nirs_compact_df[nirs_signal_col].to_numpy(dtype=float)

    # --- Score lag candidates ---

    def score_for_shift(shift_samples: int) -> float:
        """
        Score one candidate lag by measuring haemoglobin onset slope within MVC_REF.

        For each VC onset in the anchor window, the slope of the z-scored NIRS signal
        in the first NIRS_SLOPE_WINDOW_S seconds is computed with a linearly decreasing
        weight (1.0 → 0.0). Weights emphasise the initial rise/fall over the plateau.

        Returns the median per-onset slope, or -inf if the shift window has too few samples
        to compute a reliable score (expected near the edges of the search range).
        """
        shifted = nirs_time_index + shift_samples

        in_master = (shifted >= 0) & (shifted < master_n)
        if not np.any(in_master):
            return -np.inf

        shifted_ok = shifted[in_master]
        signal_ok  = nirs_signal[in_master]

        in_anchor = master_anchor_mask[shifted_ok]
        if np.sum(in_anchor) < 10:
            return -np.inf

        anchor_idx    = shifted_ok[in_anchor]
        signal_anchor = signal_ok[in_anchor]

        vc_count_anchor = master_vc_count[anchor_idx]
        if vc_count_anchor.size < 3:
            return -np.inf

        # Z-score normalise within the anchor window so slope magnitudes are comparable
        # across subjects and signal columns with different absolute scales.
        mu = float(np.nanmean(signal_anchor))
        sd = float(np.nanstd(signal_anchor))
        if not np.isfinite(sd) or sd == 0.0:
            return -np.inf
        z_anchor = (signal_anchor - mu) / sd

        # Detect VC onset positions: indices where VC_count transitions from 0 to > 0.
        # np.r_[False, in_vc[:-1]] is a one-sample-lagged copy; ANDing with in_vc selects
        # rising edges (first sample inside each contraction).
        in_vc          = vc_count_anchor > 0
        onset_mask     = in_vc & (~np.r_[False, in_vc[:-1]])
        onset_positions = np.flatnonzero(onset_mask)
        if onset_positions.size < 2:
            return -np.inf

        n = z_anchor.size
        slope_features = []

        for p in onset_positions:
            right = min(n, p + nirs_slope_window_samples)
            if (right - p) < NIRS_MIN_WINDOW_SAMPLES:
                continue

            dz = np.diff(z_anchor[p:right])
            if dz.size == 0:
                continue

            # Linearly decreasing weight (1.0 at onset → 0.0 at window end) to emphasise
            # the initial slope change. The lag shifts the onset timing, not the plateau.
            weight = np.linspace(1.0, 0.0, dz.size, endpoint=True)

            # Negative dz = falling signal (O2Hb). Positive dz = rising (HHb).
            # The code handles both: prefer negative-slope features when they exist,
            # otherwise fall back to abs(dz) which captures any directional change.
            neg = dz < 0
            if np.any(neg):
                slope_feature = float(np.nanmax((-dz[neg]) * weight[neg]))
            else:
                slope_feature = float(np.nanmax(np.abs(dz) * weight))

            if np.isfinite(slope_feature):
                slope_features.append(slope_feature)

        if len(slope_features) < 2:
            return -np.inf

        # Median over onsets — robust to one or two contractions where the subject did
        # not reach full force (common in MVC_REF warm-up repetitions).
        return float(np.nanmedian(np.array(slope_features, dtype=float)))

    scores  = np.array([score_for_shift(s) for s in shift_candidates_samples], dtype=float)
    n_finite = int(np.isfinite(scores).sum())
    if n_finite == 0:
        raise ValueError(
            "[06b_nirs_sync] No finite lag scores — check anchor SEQ labeling, "
            "VC detection, lag search window, and NIRS-to-master mapping."
        )

    # nanargmax is used because some score entries may be -inf (skipped candidates).
    best_idx            = int(np.nanargmax(scores))
    best_lag_s_auto     = float(lag_candidates_s[best_idx])
    total_lag_s         = best_lag_s_auto + manual_nudge_s
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))

    # --- Apply shift ---
    df_aligned             = nirs_compact_df.copy()
    time_shifted           = df_aligned["time_index"].to_numpy(dtype=int) + total_shift_samples
    in_range               = (time_shifted >= 0) & (time_shifted < master_n)
    n_drop_low             = int(np.sum(time_shifted < 0))
    n_drop_high            = int(np.sum(time_shifted >= master_n))
    df_aligned             = df_aligned.loc[in_range].copy()

    ti                     = time_shifted[in_range].astype(int, copy=False)
    df_aligned["time_index"]  = ti
    # Direct numpy array indexing rather than master_index_grid.loc[ti, col] — the
    # .loc form assumes a RangeIndex and silently gives wrong results if the master
    # grid was ever filtered or reindexed.
    df_aligned["SEQ_index"] = master_seq_index[ti]
    df_aligned["SEQ"]       = master_seq[ti]
    df_aligned["VC"]        = master_vc[ti]
    df_aligned["VC_count"]  = master_vc_count[ti]
    df_aligned              = df_aligned.sort_values("time_index", kind="mergesort").reset_index(drop=True)

    # --- Write cache ---
    df_aligned.to_parquet(CACHE_NIRS_ALIGNED, index=False)

    n_kept = len(df_aligned)
    print(
        f"[06b_nirs_sync] Done — auto lag={best_lag_s_auto:+.3f}s, "
        f"nudge={manual_nudge_s:+.3f}s, total={total_lag_s:+.3f}s "
        f"({total_shift_samples:+d} samples at {master_fs_hz:.0f} Hz)\n"
        f"  {n_kept}/{len(nirs_compact_df)} rows kept "
        f"(dropped {n_drop_low} low, {n_drop_high} high)"
    )

    # --- QC plot ---
    # 3-panel layout (shared x-axis = session time in seconds):
    #   Panel 1 (overview): full-session torque + NIRS before and after sync on twin y-axes.
    #                        Orange shading marks VC spans within MVC_REF.
    #   Panel 2 (zoom before): torque + unsynced NIRS restricted to the anchor window.
    #   Panel 3 (zoom after): torque + synced NIRS restricted to the anchor window.
    # Correct alignment: NIRS onset slope should line up with VC boundary shading.
    plt.close("all")

    # --- Plot style constants ---
    marker_size_overview = 1.6
    markevery_overview   = 8
    nirs_before_color    = "blue"
    nirs_after_color     = "red"
    vc_fill_alpha        = 0.18
    vc_edge_lw           = 2.0
    vc_face_rgba         = (1.0, 0.5, 0.0, vc_fill_alpha)
    vc_edge_rgba         = (1.0, 0.5, 0.0, 1.0)

    # --- Anchor window extents ---
    anchor_time_indices  = np.where(master_anchor_mask)[0]
    anchor_start_index   = int(anchor_time_indices[0])
    anchor_end_index     = int(anchor_time_indices[-1])
    anchor_start_time_s  = float(master_time_ref_s[anchor_start_index])
    anchor_end_time_s    = float(master_time_ref_s[anchor_end_index])

    # --- VC spans within anchor (numpy diff — consistent with _04_ and _05b_ patterns) ---
    mask_vc_anchor  = master_anchor_mask & (master_vc_count > 0)
    d               = np.diff(mask_vc_anchor.astype(int))
    vc_span_starts  = np.where(d ==  1)[0] + 1
    vc_span_stops   = np.where(d == -1)[0]
    if mask_vc_anchor[0]:
        vc_span_starts = np.r_[0, vc_span_starts]
    if mask_vc_anchor[-1]:
        vc_span_stops  = np.r_[vc_span_stops, len(mask_vc_anchor) - 1]
    vc_spans_in_anchor = list(zip(vc_span_starts.tolist(), vc_span_stops.tolist()))

    # --- Data arrays for plotting ---
    torque_time_indices = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_time_s       = master_time_ref_s[torque_time_indices]
    torque_values       = torque_compact_df[torque_col].to_numpy(dtype=float)

    nirs_before_time_idx = nirs_compact_df["time_index"].to_numpy(dtype=int)
    nirs_before_time_s   = master_time_ref_s[nirs_before_time_idx]
    nirs_before_values   = nirs_compact_df[nirs_signal_col].to_numpy(dtype=float)
    order = np.argsort(nirs_before_time_s)
    nirs_before_time_s   = nirs_before_time_s[order]
    nirs_before_values   = nirs_before_values[order]

    nirs_after_time_idx = df_aligned["time_index"].to_numpy(dtype=int)
    nirs_after_time_s   = master_time_ref_s[nirs_after_time_idx]
    nirs_after_values   = df_aligned[nirs_signal_col].to_numpy(dtype=float)
    order = np.argsort(nirs_after_time_s)
    nirs_after_time_s   = nirs_after_time_s[order]
    nirs_after_values   = nirs_after_values[order]

    # Shared NIRS y-axis limits from combined before+after to keep both traces comparable.
    all_nirs = np.concatenate([nirs_before_values, nirs_after_values])
    nirs_p01 = float(np.nanpercentile(all_nirs, 1))
    nirs_p99 = float(np.nanpercentile(all_nirs, 99))
    use_nirs_ylim = np.isfinite(nirs_p01) and np.isfinite(nirs_p99) and (nirs_p99 > nirs_p01)
    if use_nirs_ylim:
        nirs_span     = nirs_p99 - nirs_p01
        nirs_ylim_lo  = nirs_p01 - 0.05 * nirs_span
        nirs_ylim_hi  = nirs_p99 + 0.05 * nirs_span

    torque_in_anchor = (
        (torque_time_indices >= anchor_start_index) &
        (torque_time_indices <= anchor_end_index)
    )

    def _shade_vc_spans(ax, first_label=None):
        """Draw orange VC span shading on ax. first_label sets legend label on the first span."""
        for i, (start_idx, stop_idx) in enumerate(vc_spans_in_anchor):
            ax.axvspan(
                master_time_ref_s[start_idx],
                master_time_ref_s[stop_idx],
                facecolor=vc_face_rgba,
                edgecolor=vc_edge_rgba,
                linewidth=vc_edge_lw,
                label=first_label if i == 0 else None,
            )

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 10))

    ax1_torque = fig.add_subplot(3, 1, 1)
    ax1_torque.set_title(
        f"NIRS sync overview (anchor={ANCHOR_SEQ}) — OVERLAY "
        f"| auto={best_lag_s_auto:.3f}s | manual={manual_nudge_s:.3f}s | total={total_lag_s:.3f}s"
    )
    ax1_torque.axvspan(anchor_start_time_s, anchor_end_time_s, facecolor=(0, 0, 0, 0.05), edgecolor="none")
    _shade_vc_spans(ax1_torque, first_label="VC")
    ax1_torque.plot(torque_time_s, torque_values, linewidth=0.9, label=torque_col)
    ax1_torque.set_ylabel("Torque")
    ax1_torque.set_xlabel("time_ref_s (s)")

    ax1_nirs = ax1_torque.twinx()
    ax1_nirs.plot(
        nirs_before_time_s, nirs_before_values,
        linewidth=1.0, color=nirs_before_color,
        marker="o", markersize=marker_size_overview, markeredgewidth=0.0,
        markevery=markevery_overview, label=f"NIRS BEFORE ({nirs_signal_col})",
    )
    ax1_nirs.plot(
        nirs_after_time_s, nirs_after_values,
        linewidth=1.0, color=nirs_after_color,
        marker="o", markersize=marker_size_overview, markeredgewidth=0.0,
        markevery=markevery_overview, label="NIRS AFTER",
    )
    ax1_nirs.set_ylabel("NIRS")
    if use_nirs_ylim:
        ax1_nirs.set_ylim(nirs_ylim_lo, nirs_ylim_hi)
    h1, l1 = ax1_torque.get_legend_handles_labels()
    h2, l2 = ax1_nirs.get_legend_handles_labels()
    ax1_torque.legend(h1 + h2, l1 + l2, loc="upper left")

    ax2_torque = fig.add_subplot(3, 1, 2)
    ax2_torque.set_title(f"BEFORE (unsynced) — zoom on {ANCHOR_SEQ}")
    _shade_vc_spans(ax2_torque)
    ax2_torque.plot(
        master_time_ref_s[torque_time_indices[torque_in_anchor]],
        torque_values[torque_in_anchor],
        linewidth=0.9,
    )
    ax2_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax2_torque.set_ylabel("Torque")
    ax2_torque.set_xlabel("time_ref_s (s)")

    ax2_nirs = ax2_torque.twinx()
    before_in_anchor = (
        (nirs_before_time_idx >= anchor_start_index) &
        (nirs_before_time_idx <= anchor_end_index)
    )
    zoom_before_t   = master_time_ref_s[nirs_before_time_idx[before_in_anchor]]
    zoom_before_v   = nirs_compact_df.loc[before_in_anchor, nirs_signal_col].to_numpy(dtype=float)
    order           = np.argsort(zoom_before_t)
    ax2_nirs.plot(
        zoom_before_t[order], zoom_before_v[order],
        linewidth=1.1, color=nirs_before_color,
        marker="o", markersize=marker_size_overview, markeredgewidth=0.0,
        markevery=max(1, markevery_overview // 2),
    )
    ax2_nirs.set_ylabel("NIRS")

    ax3_torque = fig.add_subplot(3, 1, 3)
    ax3_torque.set_title(f"AFTER (synced) — zoom on {ANCHOR_SEQ}")
    _shade_vc_spans(ax3_torque)
    ax3_torque.plot(
        master_time_ref_s[torque_time_indices[torque_in_anchor]],
        torque_values[torque_in_anchor],
        linewidth=0.9,
    )
    ax3_torque.set_xlim(anchor_start_time_s, anchor_end_time_s)
    ax3_torque.set_ylabel("Torque")
    ax3_torque.set_xlabel("time_ref_s (s)")

    ax3_nirs = ax3_torque.twinx()
    after_in_anchor = (
        (nirs_after_time_idx >= anchor_start_index) &
        (nirs_after_time_idx <= anchor_end_index)
    )
    zoom_after_t    = master_time_ref_s[nirs_after_time_idx[after_in_anchor]]
    zoom_after_v    = df_aligned.loc[after_in_anchor, nirs_signal_col].to_numpy(dtype=float)
    order           = np.argsort(zoom_after_t)
    ax3_nirs.plot(
        zoom_after_t[order], zoom_after_v[order],
        linewidth=1.1, color=nirs_after_color,
        marker="o", markersize=marker_size_overview, markeredgewidth=0.0,
        markevery=max(1, markevery_overview // 2),
    )
    ax3_nirs.set_ylabel("NIRS")

    plt.tight_layout()
    plt.show()

    # --- Update ctx ---
    ctx["nirs_compact_aligned_df"] = df_aligned

    return df_aligned, fig
