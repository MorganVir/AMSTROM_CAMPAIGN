# _07_myoton_import_sync.py - MYOTON IMPORT AND SYNC
# Load MyotonPRO vibration data from CSV and map each test to the master timeline.
#
# The MyotonPRO fires a series of brief mechanical taps against the muscle surface and
# records the resulting oscillation to estimate tissue mechanical properties. Each test
# site yields MYO_TAPS_PER_TEST (=5) individual taps; rows 0–4 are test 1, rows 5–9
# test 2, and so on. Measurement timestamps come from the device clock, stored in the
# "Measurement time" column of the export CSV.
#
# Five mechanical metrics are imported per tap:
#   Frequency  : oscillation frequency (Hz) — proxy for muscle tone / tension
#   Stiffness  : dynamic stiffness (N/m)
#   Decrement  : logarithmic decrement — describes oscillation damping
#   Relaxation : relaxation time (ms)
#   Creep      : tissue creep — slow deformation under sustained load
#
# Alignment
#   Wall-clock timestamps are converted to session seconds relative to ts_ref (Delsys
#   anchor), then snapped to the nearest master grid point via nearest-neighbour search.
#   Taps that fall outside the master timeline window are dropped.
#
# Output rows
#   The returned DataFrame contains two interleaved row types (row_type column):
#     "tap"  — one row per individual tap (MYO_TAPS_PER_TEST per test)
#     "test" — one row per test, holding the mean of its constituent tap metrics
#
# Inputs
#   ctx keys  : RUN_ID, CACHE_DIR, master_index_grid, torque_compact_df
#   parameter : raw_myoton_dir — folder containing <RUN_ID>_Myoton.csv
#               ts_ref          — session reference timestamp (Delsys clock anchor)
#               force_recompute — bypass cache and re-read the CSV
#   files     : data/raw_signal/myoton/<RUN_ID>_Myoton.csv
#
# Outputs
#   return value : myoton_compact_df — tap and test-mean rows, mapped to time_index
#                                       with SEQ/VC/VC_count from the master grid
#                  fig               — matplotlib Figure of the QC plot (None on cache hit)
#
# Cache
#   - 07_myoton_compact.parquet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Number of individual taps the MyotonPRO fires per test site.
# Rows 0–4 in the CSV belong to test 1, rows 5–9 to test 2, etc.
MYO_TAPS_PER_TEST = 5

# Myoton exports semicolon-delimited CSV (not comma-delimited).
CSV_SEP = ";"

# Torque trace downsampling factor for the QC plot.
# Full-resolution torque (~2148 Hz) makes the plot slow; factor of 50 keeps
# the shape clearly readable at interactive speed.
PLOT_DECIM = 50

# Scatter marker sizes in the QC plot (matplotlib s= units, roughly pixels²).
TAP_MARKER_SIZE  = 18   # individual tap dots
TEST_MARKER_SIZE = 60   # per-test mean diamonds


def run_myoton_load_sync(
    ctx,
    raw_myoton_dir,
    ts_ref,
    force_recompute,
):
    """
    Import a MyotonPRO CSV file and align tap measurements to the master timeline.

    Each tap row is snapped to its nearest master grid time_index via nearest-neighbour
    search, then stamped with SEQ/VC/VC_count from the master grid. A second pass
    aggregates the MYO_TAPS_PER_TEST taps of each test into a mean row. Both tap and
    test-mean rows are returned together in one compact DataFrame with a row_type column.

    The QC plot draws all five Myoton metrics on separate panels sharing an x-axis,
    each overlaid on the decimated torque trace for visual alignment inspection.

    Returns (myoton_compact_df, fig). fig is None on a cache hit.
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    torque_compact_df = ctx["torque_compact_df"]

    # --- Cache paths ---
    CACHE_MYOTON = cache_dir / "07_myoton_compact.parquet"
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_MYOTON"] = CACHE_MYOTON

    # --- Cache check ---
    if CACHE_MYOTON.exists() and not force_recompute:
        print("[07_myoton_import] Cache hit — loaded from cache. Set force_recompute=True to re-import.")
        return pd.read_parquet(CACHE_MYOTON), None

    # --- Load CSV ---
    myo_raw_path = raw_myoton_dir / f"{run_id}_Myoton.csv"
    print(f"[07_myoton_import] Loading: {myo_raw_path.name}")
    myo_raw_df = pd.read_csv(myo_raw_path, sep=CSV_SEP, engine="python")

    # --- Parse timestamps and metrics ---
    myo_raw_df["myo_ts"] = pd.to_datetime(
        myo_raw_df["Measurement time"],
        format="mixed",
        dayfirst=True,
    )

    # Metric column names in the CSV and their prefixed pipeline equivalents.
    # e.g. "Stiffness" in the CSV becomes "myo_Stiffness" in the compact DataFrame.
    myo_metric_base_names = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
    myo_metric_cols       = [f"myo_{name}" for name in myo_metric_base_names]

    for base_name, prefixed_name in zip(myo_metric_base_names, myo_metric_cols):
        myo_raw_df[prefixed_name] = pd.to_numeric(myo_raw_df[base_name], errors="coerce")

    # --- Tap structure ---
    # Keep only the columns needed downstream; reset to a clean RangeIndex so that
    # the integer-division test assignment below works correctly on row position.
    myo_taps_df = myo_raw_df[["myo_ts"] + myo_metric_cols].copy().reset_index(drop=True)

    # Assign each tap to its test by integer-dividing the row position by MYO_TAPS_PER_TEST.
    # e.g. rows 0–4 → test 1 (0//5 + 1), rows 5–9 → test 2 (5//5 + 1).
    myo_taps_df["myo_test_id"]     = (myo_taps_df.index // MYO_TAPS_PER_TEST) + 1
    # Modulo gives 0–4 within each test group on a clean RangeIndex; +1 makes it 1-based.
    myo_taps_df["myo_tap_in_test"] = (myo_taps_df.index % MYO_TAPS_PER_TEST) + 1

    # Convert wall-clock timestamps to session seconds anchored at ts_ref.
    myo_taps_df["myo_time_s"] = (myo_taps_df["myo_ts"] - ts_ref).dt.total_seconds()

    # --- Extract master arrays ---
    # Extracted once here so the in-range filter, nearest-neighbour mapping, and QC
    # plot all access them via direct array indexing (avoids repeated .to_numpy() calls).
    master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    master_time_index = master_index_grid["time_index"].to_numpy(dtype=int)
    master_seq_index  = master_index_grid["SEQ_index"].to_numpy(dtype=int)
    master_seq        = master_index_grid["SEQ"].to_numpy(dtype=object)
    master_vc         = master_index_grid["VC"].to_numpy(dtype=int)
    master_vc_count   = master_index_grid["VC_count"].to_numpy(dtype=int)

    # --- In-range filter ---
    # Drop taps that fall before or after the master timeline; they cannot be assigned
    # a valid time_index and would produce out-of-bounds indices in the mapping step.
    myo_time_s  = myo_taps_df["myo_time_s"].to_numpy(dtype=float)
    in_range    = (myo_time_s >= master_time_ref_s[0]) & (myo_time_s <= master_time_ref_s[-1])
    myo_taps_df = myo_taps_df.loc[in_range].copy().reset_index(drop=True)
    myo_time_s  = myo_time_s[in_range]

    # --- Map to master ---
    # Nearest-neighbour matching: for each tap, find the two surrounding master grid
    # points and assign to whichever is closer. Ties go to the left (earlier) point.
    # Clipping to [1, n-1] ensures cand-1 and cand are always valid indices; the
    # in-range filter above guarantees no tap falls outside the grid.
    cand           = np.searchsorted(master_time_ref_s, myo_time_s, side="left")
    cand           = np.clip(cand, 1, len(master_time_ref_s) - 1)
    left           = master_time_ref_s[cand - 1]
    right          = master_time_ref_s[cand]
    choose_left    = (myo_time_s - left) <= (right - myo_time_s)
    myo_time_index = np.where(choose_left, cand - 1, cand).astype(int)

    # --- Stamp master keys ---
    myo_taps_df["time_index"] = myo_time_index
    myo_taps_df["SEQ_index"]  = master_seq_index[myo_time_index]
    myo_taps_df["SEQ"]        = master_seq[myo_time_index]
    myo_taps_df["VC"]         = master_vc[myo_time_index]
    myo_taps_df["VC_count"]   = master_vc_count[myo_time_index]

    # --- Per-test aggregation ---
    # One test-mean row per test site: metric columns averaged over MYO_TAPS_PER_TEST taps;
    # time and grid columns computed as medians so they represent the middle tap's position.
    # time_index is rounded after the median because fractional indices are not valid
    # master grid keys.
    agg_dict = {
        "myo_ts":     ("myo_ts",     "median"),
        "myo_time_s": ("myo_time_s", "median"),
        "time_index": ("time_index", "median"),
        "SEQ_index":  ("SEQ_index",  "median"),
        "SEQ":        ("SEQ",        "first"),
        "VC":         ("VC",         "median"),
        "VC_count":   ("VC_count",   "median"),
    }
    for metric_col in myo_metric_cols:
        agg_dict[metric_col] = (metric_col, "mean")

    myo_tests_df = (
        myo_taps_df
        .groupby("myo_test_id", as_index=False)
        .agg(**agg_dict)
    )
    myo_tests_df["time_index"] = myo_tests_df["time_index"].round().astype(int)

    # --- Pack compact DataFrame ---
    # Interleave tap rows and test-mean rows in one DataFrame with a row_type discriminator.
    # myo_tap_in_test is NaN for test rows — they aggregate all taps, so no single
    # tap number applies.
    taps_keep = [
        "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
        "myo_test_id", "myo_tap_in_test", "myo_time_s",
    ] + myo_metric_cols

    tests_keep = [
        "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
        "myo_test_id", "myo_time_s",
    ] + myo_metric_cols

    taps_out = myo_taps_df[taps_keep].copy()
    taps_out.insert(0, "row_type", "tap")

    tests_out = myo_tests_df[tests_keep].copy()
    tests_out["myo_tap_in_test"] = np.nan
    tests_out.insert(0, "row_type", "test")

    col_order = [
        "row_type", "time_index", "SEQ_index", "SEQ", "VC", "VC_count",
        "myo_test_id", "myo_tap_in_test", "myo_time_s",
    ] + myo_metric_cols

    myoton_compact_df = pd.concat([taps_out, tests_out], ignore_index=True)[col_order]

    # --- Cache write ---
    myoton_compact_df.to_parquet(CACHE_MYOTON, index=False)
    print(f"[07_myoton_import] Done — {len(myoton_compact_df)} rows written to {CACHE_MYOTON.name}.")

    # --- QC plot ---
    # Five panels (one per Myoton metric) sharing an x-axis. Each panel overlays the
    # decimated torque trace (left axis, grey) with tap dots and test-mean diamonds
    # (right axis, coloured). Orange bands mark the VC regions within each SEQ.
    plt.close("all")

    taps_df  = myoton_compact_df[myoton_compact_df["row_type"] == "tap"]
    tests_df = myoton_compact_df[myoton_compact_df["row_type"] == "test"]

    torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
    torque_background = torque_compact_df["torque_raw"].to_numpy(dtype=float)

    # Build per-SEQ VC bounding boxes for the orange background shading.
    # For each sequence, find the first and last master grid point that belongs to a VC,
    # then store the corresponding time_index values as (start, end) for axvspan.
    master_vc_mask = master_vc_count > 0
    seq_change_idx = np.where(np.diff(master_seq_index) != 0)[0] + 1
    seq_bounds     = np.r_[0, seq_change_idx, len(master_seq_index)]
    seq_boxes      = []
    for seq_start, seq_end in zip(seq_bounds[:-1], seq_bounds[1:]):
        vc_local = master_vc_mask[seq_start:seq_end]
        if not np.any(vc_local):
            continue
        vc_idx   = np.where(vc_local)[0]
        vc_first = seq_start + vc_idx[0]
        vc_last  = seq_start + vc_idx[-1]
        seq_boxes.append((master_time_index[vc_first], master_time_index[vc_last]))

    n_panels = len(myo_metric_cols)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, 3.2 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    # Pre-extract shared x-axis arrays so each panel iteration avoids redundant .to_numpy() calls.
    taps_x = taps_df["time_index"].to_numpy(dtype=int)
    tests_x = tests_df["time_index"].to_numpy(dtype=int)
    first_ax_right = None

    for ax_left, metric_col in zip(axes, myo_metric_cols):
        ax_left.plot(
            torque_time_index[::PLOT_DECIM],
            torque_background[::PLOT_DECIM],
            color="0.85",
            linewidth=1,
            zorder=1,
        )

        for x_start, x_end in seq_boxes:
            ax_left.axvspan(
                x_start, x_end,
                facecolor="orange",
                alpha=0.06,
                edgecolor="none",
                linewidth=0,
                zorder=5,
            )

        ax_left.set_ylabel("Torque", color="0.5")
        ax_left.tick_params(axis="y", labelcolor="0.5")

        ax_right = ax_left.twinx()
        if first_ax_right is None:
            first_ax_right = ax_right

        tests_y = tests_df[metric_col].to_numpy(dtype=float)
        ax_right.scatter(
            taps_x,
            taps_df[metric_col].to_numpy(dtype=float),
            s=TAP_MARKER_SIZE,
            alpha=0.8,
            label="taps",
            zorder=20,
        )
        ax_right.scatter(
            tests_x,
            tests_y,
            s=TEST_MARKER_SIZE,
            marker="D",
            label="test mean",
            zorder=21,
        )
        ax_right.plot(
            tests_x,
            tests_y,
            linewidth=1,
            alpha=0.35,
            zorder=19,
        )

        ax_right.set_ylabel(metric_col)
        ax_right.grid(alpha=0.25)

    axes[-1].set_xlabel("time_index (master grid)")
    first_ax_right.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    return myoton_compact_df, fig
