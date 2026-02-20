# 09_myoton_import_sync — discrete MYOTON import + map to master grid

import numpy as np
import pandas as pd

# ----------------------------
# REQUIRED INPUTS (asserts only)
# ----------------------------
assert "RUN_ID" in globals() and isinstance(RUN_ID, str) and RUN_ID, "RUN_ID missing/invalid"
assert "RAW_MYOTON_DIR" in globals()
assert "ts_ref" in globals(), "ts_ref missing (absolute session anchor)"
assert "master_index_grid" in globals(), "master_index_grid missing"

for col in ["time_ref_s", "SEQ_index", "VC", "VC_count"]:
    assert col in master_index_grid.columns, f"master_index_grid missing column: {col}"

# ----------------------------
# Params
# ----------------------------
myo_raw_path = RAW_MYOTON_DIR / f"{RUN_ID}_Myoton.csv"
myo_taps_per_test = 5
myo_csv_sep = ";"

assert myo_taps_per_test > 0, "MYOTON_TAPS_PER_TEST must be > 0"

# Metric definitions
myo_metric_base_names = ["Frequency", "Stiffness", "Decrement", "Relaxation", "Creep"]
myo_metric_cols = [f"myo_{name}" for name in myo_metric_base_names]

# ----------------------------
# Load
# ----------------------------

print(f"Loading MYOTON file: {getattr(myo_raw_path, 'name', str(myo_raw_path))}")
myo_raw_df = pd.read_csv(myo_raw_path, sep=myo_csv_sep, engine="python")

assert "Measurement time" in myo_raw_df.columns, "MYOTON csv missing 'Measurement time' column"

myo_raw_df["myo_ts"] = pd.to_datetime(
    myo_raw_df["Measurement time"],
    format="mixed",
    dayfirst=True,
)
assert myo_raw_df["myo_ts"].notna().all(), "Myoton timestamp parsing failed (NaT present)"

for base_name, prefixed_name in zip(myo_metric_base_names, myo_metric_cols):
    assert base_name in myo_raw_df.columns, f"MYOTON csv missing metric column: {base_name}"
    myo_raw_df[prefixed_name] = pd.to_numeric(myo_raw_df[base_name], errors="coerce")

# One row per tap
myo_taps_df = myo_raw_df[["myo_ts"] + myo_metric_cols].copy().reset_index(drop=True)

# Tap/test structure
myo_taps_df["myo_test_id"] = (myo_taps_df.index // myo_taps_per_test) + 1
myo_taps_df["myo_tap_in_test"] = myo_taps_df.groupby("myo_test_id").cumcount() + 1

# Timestamp → seconds from session anchor
myo_taps_df["myo_time_s"] = (myo_taps_df["myo_ts"] - ts_ref).dt.total_seconds()

# ----------------------------
# Nearest mapping (NO silent clipping)
# ----------------------------
master_time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
assert master_time_ref_s.size >= 2, "master_index_grid too small for nearest mapping"
assert np.all(np.diff(master_time_ref_s) >= 0), "master_index_grid time_ref_s must be monotonic non-decreasing"

myo_time_s = myo_taps_df["myo_time_s"].to_numpy(dtype=float)

t_min = float(master_time_ref_s[0])
t_max = float(master_time_ref_s[-1])

in_range = (myo_time_s >= t_min) & (myo_time_s <= t_max)
dropped_out_of_range = int((~in_range).sum())

# Drop out-of-range explicitly
if dropped_out_of_range > 0:
    myo_taps_df = myo_taps_df.loc[in_range].copy().reset_index(drop=True)
    myo_time_s = myo_time_s[in_range]

# Map in-range
idx_right = np.searchsorted(master_time_ref_s, myo_time_s, side="left")
n = int(master_time_ref_s.size)

# idx_right can be 0..n-1 for in-range (with side="left" and inclusive max)
assert (idx_right >= 0).all() and (idx_right < n).all(), "Unexpected searchsorted index out of bounds"

idx_left = np.maximum(idx_right - 1, 0)

# If idx_right == 0, nearest is 0
is_first = idx_right == 0
choose_right = np.zeros_like(idx_right, dtype=bool)

# For others, compare distances between left/right candidates
not_first = ~is_first
dr = np.abs(master_time_ref_s[idx_right[not_first]] - myo_time_s[not_first])
dl = np.abs(myo_time_s[not_first] - master_time_ref_s[idx_left[not_first]])
choose_right[not_first] = dr < dl

myo_time_index = np.where(choose_right, idx_right, idx_left).astype(int)
assert (myo_time_index >= 0).all() and (myo_time_index < n).all(), "Nearest mapping produced invalid indices"

# Attach shared keys
myo_taps_df["time_index"] = myo_time_index
myo_taps_df["SEQ_index"] = master_index_grid["SEQ_index"].to_numpy(dtype=int)[myo_time_index]
myo_taps_df["VC"] = master_index_grid["VC"].to_numpy(dtype=int)[myo_time_index]
myo_taps_df["VC_count"] = master_index_grid["VC_count"].to_numpy(dtype=int)[myo_time_index]

# ----------------------------
# Per-test means (1 row per test)
# ----------------------------
agg_dict = {
    "myo_ts": ("myo_ts", "median"),
    "myo_time_s": ("myo_time_s", "median"),
    "time_index": ("time_index", "median"),
    "SEQ_index": ("SEQ_index", "median"),
    "VC": ("VC", "median"),
    "VC_count": ("VC_count", "median"),
}
for metric_col in myo_metric_cols:
    agg_dict[metric_col] = (metric_col, "mean")

myo_tests_df = (
    myo_taps_df
    .groupby("myo_test_id", as_index=False)
    .agg(**agg_dict)
)
myo_tests_df["time_index"] = myo_tests_df["time_index"].round().astype(int)

# ----------------------------
# Pack compact (taps + tests) with explicit row_type
# ----------------------------
taps_keep = [
    "time_index", "SEQ_index", "VC", "VC_count",
    "myo_test_id", "myo_tap_in_test", "myo_time_s",
] + myo_metric_cols

tests_keep = [
    "time_index", "SEQ_index", "VC", "VC_count",
    "myo_test_id", "myo_time_s",
] + myo_metric_cols

taps_out = myo_taps_df[taps_keep].copy()
taps_out.insert(0, "row_type", "tap")

tests_out = myo_tests_df[tests_keep].copy()
tests_out["myo_tap_in_test"] = np.nan  # keep schema stable
tests_out.insert(0, "row_type", "test")

# Order columns consistently
col_order = ["row_type", "time_index", "SEQ_index", "VC", "VC_count",
             "myo_test_id", "myo_tap_in_test", "myo_time_s"] + myo_metric_cols
myoton_compact_df_out = pd.concat([taps_out, tests_out], ignore_index=True)[col_order]



#region PLOTING MYOTON OVER TORQUE

import numpy as np
import matplotlib.pyplot as plt

plt.close("all")  # prevents leftover empty figure rows from previous runs


# QC SETTINGS

MYOTON_QC_MASTER_DECIM = 50
MYOTON_QC_TAP_SIZE = 18
MYOTON_QC_TEST_SIZE = 60

MYOTON_QC_METRICS = [
    "myo_Frequency",
    "myo_Stiffness",
    "myo_Decrement",
    "myo_Relaxation",
    "myo_Creep",
]


# Split taps / tests

myoton_taps_df = myoton_compact_df_out[myoton_compact_df_out["row_type"] == "tap"]
myoton_tests_df = myoton_compact_df_out[myoton_compact_df_out["row_type"] == "test"]


# Torque background (x/y from torque_compact_df)

torque_time_index = torque_compact_df["time_index"].to_numpy(dtype=int)
torque_background = torque_compact_df["torque_raw"].to_numpy(dtype=float)


# Build VC hull per SEQ (master grid)
master_time_index = master_index_grid["time_index"].to_numpy(dtype=int)
master_seq_index = master_index_grid["SEQ_index"].to_numpy(dtype=int)
master_vc_mask = master_index_grid["VC_count"].to_numpy(dtype=int) > 0

seq_change_idx = np.where(np.diff(master_seq_index) != 0)[0] + 1
seq_bounds = np.r_[0, seq_change_idx, len(master_seq_index)]

seq_boxes = []
for seq_start, seq_end in zip(seq_bounds[:-1], seq_bounds[1:]):
    vc_local = master_vc_mask[seq_start:seq_end]
    if not np.any(vc_local):
        continue
    vc_first = seq_start + np.where(vc_local)[0][0]
    vc_last = seq_start + np.where(vc_local)[0][-1]
    seq_boxes.append((master_time_index[vc_first], master_time_index[vc_last]))


# Figure
n_panels = len(MYOTON_QC_METRICS)
fig_myoton_qc, axes_myoton_qc = plt.subplots(
    n_panels, 1,
    figsize=(13, 3.2 * n_panels),
    sharex=True,
)

if n_panels == 1:
    axes_myoton_qc = [axes_myoton_qc]

legend_axis_right = None

for ax_left, metric_col in zip(axes_myoton_qc, MYOTON_QC_METRICS):

    # Torque on LEFT axis
    ax_left.plot(
        torque_time_index[::MYOTON_QC_MASTER_DECIM],
        torque_background[::MYOTON_QC_MASTER_DECIM],
        color="0.85",
        linewidth=1,
        zorder=1,
    )

    # VC hull boxes (fill only, no border)
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

    # MYOTON on RIGHT axis
    ax_right = ax_left.twinx()
    if legend_axis_right is None:
        legend_axis_right = ax_right

    ax_right.scatter(
        myoton_taps_df["time_index"].to_numpy(dtype=int),
        myoton_taps_df[metric_col].to_numpy(dtype=float),
        s=MYOTON_QC_TAP_SIZE,
        alpha=0.8,
        label="taps",
        zorder=20,
    )

    ax_right.scatter(
        myoton_tests_df["time_index"].to_numpy(dtype=int),
        myoton_tests_df[metric_col].to_numpy(dtype=float),
        s=MYOTON_QC_TEST_SIZE,
        marker="D",
        label="test mean",
        zorder=21,
    )

    ax_right.plot(
        myoton_tests_df["time_index"].to_numpy(dtype=int),
        myoton_tests_df[metric_col].to_numpy(dtype=float),
        linewidth=1,
        alpha=0.35,
        zorder=19,
    )

    ax_right.set_ylabel(metric_col)
    ax_right.grid(alpha=0.25)

axes_myoton_qc[-1].set_xlabel("time_index (master grid)")
if legend_axis_right is not None:
    legend_axis_right.legend(loc="upper right")

plt.tight_layout()
plt.show()

#endregion