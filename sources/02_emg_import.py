# 02_emg_import module
# Output : master_index_grid, emg_compact_df, torque_compact_df, ts_ref


from pathlib import Path
import csv
import numpy as np
import pandas as pd
import re

import os

assert "RAW_EMG_DIR" in globals(), "RAW_EMG_DIR missing"
assert isinstance(RAW_EMG_DIR, Path), "RAW_EMG_DIR must be a pathlib.Path"


assert "RUN_ID" in globals(), "RUN_ID missing"
assert isinstance(RUN_ID, str) and len(RUN_ID) >= 7, "RUN_ID must start with subject key"
assert "RAW_ROOT" in globals(), "RAW_ROOT missing"
assert isinstance(RAW_ROOT, Path), "RAW_ROOT must be a pathlib.Path"


subj_key = RUN_ID[:7]


assert "participant_row" in globals(), "participant_row missing"

delsys_ts_init = participant_row.iloc[12] #deterministic - always here if the template is used.
delsys_ts_init_dt = pd.to_datetime(delsys_ts_init)


date_match = re.search(r"\d{8}", RUN_ID)
assert date_match is not None, "No YYYYMMDD date found in RUN_ID"
exp_date = date_match.group(0)


# EMG CSV path 
CSV_PATH = RAW_EMG_DIR / f"{subj_key}_{exp_date}_DELSYS.csv"

assert isinstance(CSV_PATH, Path), "CSV_PATH must be a pathlib.Path"
assert CSV_PATH.exists(), f"EMG CSV not found: {CSV_PATH}"

print(f"Loading DELSYS CSV: {CSV_PATH.name}")



# Detect header row (start of numeric table)
def detect_header_row(csv_path: Path, max_lines: int = 50) -> int: # look for line starting with "X[s]," in the first 50 lines 
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as file_handle:
        for line_index, line in enumerate(file_handle):
            if line_index >= max_lines:
                break
            if line.lstrip().startswith("X[s],"):
                return line_index
    raise ValueError("Header row not found (expected line starting with 'X[s],').")

header_row = detect_header_row(CSV_PATH)


#  Read the header line and parse as CSV 

with open(CSV_PATH, "r", encoding="utf-8", errors="ignore", newline="") as file_handle:
    for _ in range(header_row):
        next(file_handle)
    header_line = next(file_handle)

column_names = next(csv.reader([header_line]))

def first_index_containing(substrings: list[str], label: str) -> int:
    for substring in substrings:
        hit_indices = [i for i, name in enumerate(column_names) if substring in name]
        if hit_indices:
            return hit_indices[0]
    raise ValueError(f"Missing {label}. Tried: {substrings}")


#  Resolve required signal indices
# - torque signal
# - emg signals: EMG 6 / EMG 8 / EMG 10
# - paired time column: immediately BEFORE the signal column
torque_signal_index = first_index_containing(["Analog.D"], "torque signal (Analog.D)")
emg6_signal_index = first_index_containing(["EMG 6"], "EMG 6")
emg8_signal_index = first_index_containing(["EMG 8"], "EMG 8")
emg10_signal_index = first_index_containing(["EMG 10"], "EMG 10")

# paired time columns (time col is immediately BEFORE the signal col)
torque_time_index = torque_signal_index - 1
emg_time_index = emg6_signal_index - 1


# Load raw data (only the columns of interest, as numeric, with coercion to handle non-numeric garbage)

def first_name_containing(substrings: list[str], label: str) -> str:
    idx = first_index_containing(substrings, label)
    return column_names[idx]

def time_name_before(signal_col_name: str) -> str:
    idx = column_names.index(signal_col_name)
    if idx <= 0:
        raise ValueError(f"No time column before {signal_col_name}")
    return column_names[idx - 1]


# ----------------------------
# Load raw data by INDEX (robust to duplicate "X[s]" headers)
# ----------------------------
usecols_idx = sorted({
    torque_time_index, torque_signal_index,
    emg_time_index, emg6_signal_index, emg8_signal_index, emg10_signal_index
})

raw = pd.read_csv(
    CSV_PATH,
    header=None,
    skiprows=header_row + 1,
    usecols=usecols_idx,
)

# Map original file indices -> stable names
colmap = {
    torque_time_index: "torque_time_s",
    torque_signal_index: "torque_raw",
    emg_time_index: "emg_time_s",
    emg6_signal_index: "emg6",
    emg8_signal_index: "emg8",
    emg10_signal_index: "emg10",
}
raw = raw.rename(columns=colmap)

# Torque dataframe
biodex_raw = raw[["torque_time_s", "torque_raw"]].copy()
assert "torque_raw" in biodex_raw.columns

# EMG dataframe
emg_raw = raw[["emg_time_s", "emg6", "emg8", "emg10"]].copy()






# Reference grid (time_REF = EMG because highest hz of all signals), uniform from mean dt

emg_time_s = pd.to_numeric(emg_raw["emg_time_s"], errors="coerce").to_numpy()
emg_valid = ~np.isnan(emg_time_s)

emg_time_s = emg_time_s[emg_valid]
emg6 = pd.to_numeric(emg_raw["emg6"], errors="coerce").to_numpy()[emg_valid]
emg8 = pd.to_numeric(emg_raw["emg8"], errors="coerce").to_numpy()[emg_valid]
emg10 = pd.to_numeric(emg_raw["emg10"], errors="coerce").to_numpy()[emg_valid]

fs_ref_hz = 1.0 / np.mean(np.diff(emg_time_s))
dt_ref_s = 1.0 / fs_ref_hz
n_ref = len(emg_time_s)

time_index = np.arange(n_ref, dtype=int)
time_ref_s = time_index * dt_ref_s

# GRID ONLY 
master_index_grid = pd.DataFrame({
    "time_index": time_index,
    "time_ref_s": time_ref_s,
})

# Store ts_ref ONCE (metadata) 
ts_ref = delsys_ts_init_dt  # anchor == DELSYS_TS_init 

# --- EMG table (dense on ref grid) ---
emg_compact_df = pd.DataFrame({
    "time_index": time_index,
    "emg6": emg6,
    "emg8": emg8,
    "emg10": emg10,
})


# Torque -> ref grid (nearest EMG timestamp) as its OWN table 
torque_time_s = pd.to_numeric(biodex_raw["torque_time_s"], errors="coerce").to_numpy()
torque_raw = pd.to_numeric(biodex_raw["torque_raw"], errors="coerce").to_numpy()

torque_valid = np.isfinite(torque_time_s) & np.isfinite(torque_raw)
torque_time_s = torque_time_s[torque_valid]
torque_raw = torque_raw[torque_valid]

ins = np.searchsorted(emg_time_s, torque_time_s, side="left")
right = np.clip(ins, 0, n_ref - 1)
left  = np.clip(ins - 1, 0, n_ref - 1)

right_dist = np.abs(emg_time_s[right] - torque_time_s)
left_dist  = np.abs(emg_time_s[left] - torque_time_s)

nearest_idx = np.where(left_dist <= right_dist, left, right).astype(int)

torque_compact_df = pd.DataFrame({
    "time_index": nearest_idx,
    "torque_raw": torque_raw,
})

# last-wins if multiple torque samples map to same time_index
torque_compact_df = (
    torque_compact_df
    .sort_values("time_index", kind="mergesort")
    .drop_duplicates(subset=["time_index"], keep="last")
    .reset_index(drop=True)
)

