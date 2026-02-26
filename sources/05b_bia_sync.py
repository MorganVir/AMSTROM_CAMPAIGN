# ============================================================
# 06_bia_sync — estimate lag in MVC_REF using real BIA samples only
# - Objective (legacy v1): maximize |mean(|Z| in VC) - mean(|Z| out VC)| within MVC_REF
# - Apply best shift to bia2 + bia4 by shifting time_index, then relabel from master
# - Out-of-range after shift: DROP (never clip)
# - Cache (COMMIT only): 3 fixed parquets under CACHE_06_BIA_SYNC.parent
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Required inputs (fail loud)
# ----------------------------
assert "RUN_ID" in globals() and isinstance(RUN_ID, str) and RUN_ID, "RUN_ID missing/invalid"

assert "master_index_grid" in globals() and isinstance(master_index_grid, pd.DataFrame), "master_index_grid missing/invalid"
assert "bia2_compact_df" in globals() and isinstance(bia2_compact_df, pd.DataFrame), "bia2_compact_df missing/invalid"
assert "bia4_compact_df" in globals() and isinstance(bia4_compact_df, pd.DataFrame), "bia4_compact_df missing/invalid"

assert "bia4_freqs_hz" in globals(), "bia4_freqs_hz missing (from Step 05a)"
bia4_freqs_hz = np.asarray(bia4_freqs_hz, dtype=float)
assert bia4_freqs_hz.ndim == 1 and bia4_freqs_hz.size > 0, "bia4_freqs_hz must be 1D non-empty"

assert "CACHE_05b_BIA_SYNC" in globals(), "CACHE_05b_BIA_SYNC missing"
CACHE_06_BIA_SYNC = Path(CACHE_05b_BIA_SYNC)

# Dashboard-owned knobs/flags (no defaults here)
assert "BIA_SYNC_ANCHOR_SEQ" in globals(), "BIA_SYNC_ANCHOR_SEQ missing"
assert "BIA_SYNC_TARGET_HZ" in globals(), "BIA_SYNC_TARGET_HZ missing"
assert "BIA_SYNC_LAG_MIN_S" in globals(), "BIA_SYNC_LAG_MIN_S missing"
assert "BIA_SYNC_LAG_MAX_S" in globals(), "BIA_SYNC_LAG_MAX_S missing"
assert "BIA_SYNC_LAG_STEP_S" in globals(), "BIA_SYNC_LAG_STEP_S missing"
assert "BIA_SYNC_MANUAL_NUDGE_S" in globals(), "BIA_SYNC_MANUAL_NUDGE_S missing"


anchor_seq_name = str(BIA_SYNC_ANCHOR_SEQ)
target_hz = float(BIA_SYNC_TARGET_HZ)

# ----------------------------
# Cache paths (fixed filenames)
# ----------------------------
cache_dir = CACHE_06_BIA_SYNC.parent
cache_dir.mkdir(parents=True, exist_ok=True)

P_BIA2_ALIGNED = cache_dir / "05b_bia2_compact_aligned.parquet"
P_BIA4_ALIGNED = cache_dir / "05b_bia4_compact_aligned.parquet"


# Cache hit: load and return
if P_BIA2_ALIGNED.exists() and P_BIA4_ALIGNED.exists():
    bia2_compact_aligned_df_out = pd.read_parquet(P_BIA2_ALIGNED)
    bia4_compact_aligned_df_out = pd.read_parquet(P_BIA4_ALIGNED)
    print("Cache found. Skipping sync and loading aligned BIA parquets instead.")

    # minimal schema asserts
    for df_name, df in [("bia2_aligned", bia2_compact_aligned_df_out), ("bia4_aligned", bia4_compact_aligned_df_out)]:
        for col in ["time_index", "SEQ_index", "VC", "VC_count"]:
            assert col in df.columns, f"{df_name}: missing {col}"

else:
    # ----------------------------
    # Master grid contracts
    # ----------------------------
    for col in ["time_index", "time_ref_s", "SEQ", "SEQ_index", "VC", "VC_count"]:
        assert col in master_index_grid.columns, f"master_index_grid missing required column: {col}"

    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    assert master_time_ref_s.size >= 3, "master_index_grid too short"

    master_time_deltas = np.diff(master_time_ref_s)
    assert np.all(np.isfinite(master_time_deltas)), "master time_ref_s contains non-finite deltas"
    median_dt = float(np.median(master_time_deltas))
    assert median_dt > 0, "master time_ref_s not strictly increasing (median dt <= 0)"

    master_fs_hz = 1.0 / median_dt
    master_n = int(len(master_index_grid))

    master_seq = master_index_grid["SEQ"].to_numpy()
    master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_vc = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count = master_index_grid["VC_count"].to_numpy(dtype=int)

    # ----------------------------
    # Anchor window mask (MVC_REF)
    # ----------------------------
    mvc_mask_master = (master_seq == anchor_seq_name)
    if int(np.sum(mvc_mask_master)) == 0:
        raise ValueError(f"Anchor SEQ '{anchor_seq_name}' not found in master_index_grid['SEQ'].")

    # ----------------------------
    # Frequency must exist exactly (no approximation)
    # ----------------------------
    freq_match_idx = np.where(bia4_freqs_hz == target_hz)[0]
    if freq_match_idx.size == 0:
        raise ValueError(f"BIA_SYNC_TARGET_HZ={target_hz} not in bia4_freqs_hz bins: {bia4_freqs_hz.tolist()}")
    freq_used_hz = float(target_hz)

    # ----------------------------
    # Resolve R/Xc columns for that exact frequency (fail loud if missing)
    # ----------------------------
    freq_tag = str(int(freq_used_hz))  # your freqs are integer-valued
    bia4_R_col = f"bia4_R_ohm__f_{freq_tag}Hz"
    bia4_Xc_col = f"bia4_Xc_ohm__f_{freq_tag}Hz"
    bia2_R_col = f"bia2_R_ohm__f_{freq_tag}Hz"
    bia2_Xc_col = f"bia2_Xc_ohm__f_{freq_tag}Hz"

    for df_name, df, cols in [
        ("bia4_compact_df", bia4_compact_df, [bia4_R_col, bia4_Xc_col, "time_index"]),
        ("bia2_compact_df", bia2_compact_df, [bia2_R_col, bia2_Xc_col, "time_index"]),
    ]:
        for col in cols:
            assert col in df.columns, f"{df_name}: missing column {col}"

    # ----------------------------
    # Build |Z| from real samples only
    # ----------------------------
    bia4_time_index_raw = bia4_compact_df["time_index"].to_numpy(dtype=int)
    bia4_R = bia4_compact_df[bia4_R_col].to_numpy(dtype=float)
    bia4_Xc = bia4_compact_df[bia4_Xc_col].to_numpy(dtype=float)
    bia4_absZ = np.sqrt(bia4_R * bia4_R + bia4_Xc * bia4_Xc)

    # Basic sanity: Step 05 mapping should already be in range
    if bia4_time_index_raw.size == 0:
        raise ValueError("bia4_compact_df has zero rows.")
    if int(np.min(bia4_time_index_raw)) < 0 or int(np.max(bia4_time_index_raw)) >= master_n:
        raise ValueError("BIA time_index out of master range before sync (Step 05 mapping broken).")

    # ----------------------------
    # Lag candidate grid (in samples)
    # ----------------------------
    lag_min_s = float(BIA_SYNC_LAG_MIN_S)
    lag_max_s = float(BIA_SYNC_LAG_MAX_S)
    lag_step_s = float(BIA_SYNC_LAG_STEP_S)
    manual_nudge_s = float(BIA_SYNC_MANUAL_NUDGE_S)

    lag_candidates_s = np.arange(lag_min_s, lag_max_s + 1e-12, lag_step_s)
    if lag_candidates_s.size < 3:
        raise ValueError("Lag grid too small. Check min/max/step.")
    shift_candidates_samples = np.rint(lag_candidates_s * master_fs_hz).astype(int)

    # ----------------------------
    # Score lag using ONLY shifted BIA samples inside MVC_REF
    # score = abs(mean(absZ in VC) - mean(absZ out VC))
    # ----------------------------
    scores = np.full(lag_candidates_s.shape, np.nan, dtype=float)

    for lag_idx, shift_samples in enumerate(shift_candidates_samples):
        bia4_time_index_shifted = bia4_time_index_raw + int(shift_samples)

        valid_shifted = (bia4_time_index_shifted >= 0) & (bia4_time_index_shifted < master_n)
        if int(np.sum(valid_shifted)) < 10:
            continue

        shifted_indices_valid = bia4_time_index_shifted[valid_shifted].astype(int, copy=False)
        absZ_valid = bia4_absZ[valid_shifted]

        # keep only samples whose shifted index falls in MVC_REF
        in_mvc_ref = mvc_mask_master[shifted_indices_valid]
        if int(np.sum(in_mvc_ref)) < 10:
            continue

        mvc_indices = shifted_indices_valid[in_mvc_ref]
        mvc_absZ = absZ_valid[in_mvc_ref]

        # split by VC_count at those master indices
        mvc_vc_count = master_vc_count[mvc_indices]
        in_vc = (mvc_vc_count > 0)
        out_vc = ~in_vc

        if int(np.sum(in_vc)) < 3 or int(np.sum(out_vc)) < 3:
            continue

        mean_in_vc = float(np.nanmean(mvc_absZ[in_vc]))
        mean_out_vc = float(np.nanmean(mvc_absZ[out_vc]))
        scores[lag_idx] = abs(mean_in_vc - mean_out_vc)

    if not np.any(np.isfinite(scores)):
        raise ValueError("All lag scores are NaN (check MVC_REF labeling, VC detection, lag window, BIA mapping).")

    best_score_idx = int(np.nanargmax(scores))
    best_lag_s_auto = float(lag_candidates_s[best_score_idx])
    best_shift_samples_auto = int(shift_candidates_samples[best_score_idx])

    total_lag_s = best_lag_s_auto + manual_nudge_s
    total_shift_samples = int(np.rint(total_lag_s * master_fs_hz))

    # ----------------------------
    # Apply shift to bia2/bia4, DROP out-of-range, relabel from master
    # ----------------------------
    def _shift_drop_and_relabel(compact_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        time_index_raw = compact_df["time_index"].to_numpy(dtype=int)
        time_index_shifted = time_index_raw + int(total_shift_samples)

        valid = (time_index_shifted >= 0) & (time_index_shifted < master_n)

        n_total = int(time_index_shifted.size)
        n_valid = int(np.sum(valid))
        n_drop_low = int(np.sum(time_index_shifted < 0))
        n_drop_high = int(np.sum(time_index_shifted >= master_n))

        if n_valid == 0:
            raise ValueError("Shift produced zero valid samples (check sign, lag window, manual nudge).")

        df_out = compact_df.loc[valid].copy().reset_index(drop=True)
        time_index_valid = time_index_shifted[valid].astype(int, copy=False)

        df_out["time_index"] = time_index_valid
        df_out["SEQ_index"] = master_seq_index[time_index_valid]
        df_out["VC"] = master_vc[time_index_valid]
        df_out["VC_count"] = master_vc_count[time_index_valid]

        df_out = df_out.sort_values("time_index", kind="mergesort").reset_index(drop=True)

        stats = {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_drop_low": n_drop_low,
            "n_drop_high": n_drop_high,
        }
        return df_out, stats

    bia2_compact_aligned_df_out, bia2_stats = _shift_drop_and_relabel(bia2_compact_df)
    bia4_compact_aligned_df_out, bia4_stats = _shift_drop_and_relabel(bia4_compact_df)

    # ----------------------------
    # TUNNING PLOT
    # ----------------------------
    # ---- hard contracts (minimal) ----
    assert "torque_compact_df" in globals() and isinstance(torque_compact_df, pd.DataFrame), "torque_compact_df missing/invalid"
    assert "time_index" in torque_compact_df.columns, "torque_compact_df missing time_index"
    assert "TORQUE_COL" in globals(), "TORQUE_COL missing"
    TORQUE_COL = str(TORQUE_COL)
    assert TORQUE_COL in torque_compact_df.columns, f"torque_compact_df missing {TORQUE_COL}"

    # MVC_REF window on master
    mvc_mask = (master_index_grid["SEQ"].to_numpy() == "MVC_REF")
    mvc_idx = np.where(mvc_mask)[0]
    assert mvc_idx.size > 0, "MVC_REF not found in master_index_grid['SEQ']"
    mvc_i0 = int(mvc_idx[0])
    mvc_i1 = int(mvc_idx[-1])

    # VC borders within MVC_REF
    vc_mask = (master_index_grid["VC_count"].to_numpy(dtype=int) > 0)
    mask_vc_mvc = mvc_mask & vc_mask
    d = np.diff(mask_vc_mvc.astype(int))
    starts = np.where(d == 1)[0] + 1
    stops  = np.where(d == -1)[0]
    if mask_vc_mvc[0]:
        starts = np.r_[0, starts]
    if mask_vc_mvc[-1]:
        stops = np.r_[stops, len(mask_vc_mvc) - 1]

    # torque points restricted to MVC_REF (plot directly, no master-length fill)
    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_values = torque_compact_df[TORQUE_COL].to_numpy(dtype=float)
    torque_in_mvc = (torque_time_index >= mvc_i0) & (torque_time_index <= mvc_i1)

    # hard freq
    freq_tag = "9760"
    bia4_R_col = f"bia4_R_ohm__f_{freq_tag}Hz"
    bia4_Xc_col = f"bia4_Xc_ohm__f_{freq_tag}Hz"

    for df_name, df in [("bia4_compact_df", bia4_compact_df), ("bia4_compact_aligned_df_out", bia4_compact_aligned_df_out)]:
        assert "time_index" in df.columns, f"{df_name} missing time_index"
        assert bia4_R_col in df.columns, f"{df_name} missing {bia4_R_col}"
        assert bia4_Xc_col in df.columns, f"{df_name} missing {bia4_Xc_col}"

    # ---- extract BIA scatter in MVC_REF (raw) ----
    bia4_raw_time_index = bia4_compact_df["time_index"].to_numpy(dtype=int)
    bia4_raw_absZ = np.sqrt(
        bia4_compact_df[bia4_R_col].to_numpy(dtype=float) ** 2
        + bia4_compact_df[bia4_Xc_col].to_numpy(dtype=float) ** 2
    )
    raw_in_mvc = (bia4_raw_time_index >= mvc_i0) & (bia4_raw_time_index <= mvc_i1)
    x_raw = bia4_raw_time_index[raw_in_mvc]
    y_raw = bia4_raw_absZ[raw_in_mvc]
    order = np.argsort(x_raw)
    x_raw = x_raw[order]
    y_raw = y_raw[order]

    # ---- extract BIA scatter in MVC_REF (aligned) ----
    bia4_aln_time_index = bia4_compact_aligned_df_out["time_index"].to_numpy(dtype=int)
    bia4_aln_absZ = np.sqrt(
        bia4_compact_aligned_df_out[bia4_R_col].to_numpy(dtype=float) ** 2
        + bia4_compact_aligned_df_out[bia4_Xc_col].to_numpy(dtype=float) ** 2
    )
    aln_in_mvc = (bia4_aln_time_index >= mvc_i0) & (bia4_aln_time_index <= mvc_i1)
    x_aln = bia4_aln_time_index[aln_in_mvc]
    y_aln = bia4_aln_absZ[aln_in_mvc]
    order = np.argsort(x_aln)
    x_aln = x_aln[order]
    y_aln = y_aln[order]

    # lag text
    lag_txt = f" (lag {total_lag_s:+0.2f}s | auto {best_lag_s_auto:+0.2f}s + nudge {manual_nudge_s:+0.2f}s)"

    # ---- plot: 2 columns (BEFORE vs AFTER), each torque + bia ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")

    # BEFORE torque
    ax = axes[0, 0]
    ax.plot(torque_time_index[torque_in_mvc], torque_values[torque_in_mvc], alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title("BEFORE (Step 5 raw) — MVC_REF torque + VC borders")
    ax.set_ylabel("Torque")

    # BEFORE bia
    ax = axes[1, 0]
    ax.plot(x_raw, y_raw, linewidth=1.2, alpha=0.85)
    ax.scatter(x_raw, y_raw, s=14, alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title("|Z| bia4 @ 9760 Hz — BEFORE")
    ax.set_ylabel("|Z| (ohm)")
    ax.set_xlabel("Master time_index")

    # AFTER torque
    ax = axes[0, 1]
    ax.plot(torque_time_index[torque_in_mvc], torque_values[torque_in_mvc], alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title("AFTER (Step 5b aligned) — MVC_REF torque + VC borders")
    ax.set_ylabel("Torque")

    # AFTER bia
    ax = axes[1, 1]
    ax.plot(x_aln, y_aln, linewidth=1.2, alpha=0.85)
    ax.scatter(x_aln, y_aln, s=14, alpha=0.85)
    for s, e in zip(starts, stops):
        if s >= mvc_i0 and e <= mvc_i1:
            ax.axvline(s, color="orange", linewidth=2)
            ax.axvline(e, color="orange", linewidth=2)
    ax.set_title(f"|Z| bia4 @ 9760 Hz — AFTER{lag_txt}")
    ax.set_ylabel("|Z| (ohm)")
    ax.set_xlabel("Master time_index")

    plt.tight_layout()
    bia_sync_qc_fig_out = fig
    plt.show()



