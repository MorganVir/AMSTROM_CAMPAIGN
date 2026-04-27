# _02_emg_import.py - EMG IMPORT AND MASTER TIMELINE CONSTRUCTION
# Second step of the preprocessing pipeline.
# Loads raw EMG and torque signals from the Delsys CSV and builds the shared time grid
# that all other modalities will be aligned to.
#
# Inputs
#   ctx keys  : CACHE_DIR, participants_df
#   parameter : raw_emg_dir (Path to folder containing <RUN_ID>_DELSYS.csv)
#               run_id      (full RUN_ID string, e.g. "011BeSa_20251023")
#
# Outputs (ctx keys set)
#   - master_index_grid  : regular time backbone used by all other modalities for alignment
#   - emg_compact_df     : EMG channels (emg6, emg8, emg10) indexed on time_index
#   - torque_compact_df  : torque signal mapped onto the EMG grid via nearest-neighbour
#   - ts_ref             : session reference timestamp (from Delsys init in participants_df)
#
# Cache
#   - 02_master_index_grid.parquet
#   - 02_emg_compact.parquet
#   - 02_torque_compact.parquet
#
# Notes
#   - The Delsys CSV has a variable-length preamble before the numeric table starts.
#     The header row is detected by scanning for a line starting with "X[s],".
#   - The master timeline is a perfect regular grid reconstructed from the mean EMG sampling
#     interval (~2148 Hz). Mean is used here (not median) because EMG timestamps are nearly
#     jitter-free and there are no large inter-sequence gaps in the raw Delsys time column.
#   - Torque is recorded at a lower rate (~100 Hz) on a separate Delsys clock and must be
#     resampled onto the EMG grid via nearest-neighbour index matching.
#   - All subsequent modalities (BIA, NIRS, Myoton) follow the same nearest-neighbour pattern.

from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd


def run_emg_import(
    *,
    ctx: dict,
    raw_emg_dir: Path,
    run_id: str,
    participants_df: pd.DataFrame,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Load raw EMG and torque from the Delsys CSV and build the shared master time grid.

    Returns (master_index_grid, emg_compact_df, torque_compact_df, ts_ref).
    All four are also written into ctx so downstream steps can access them without re-passing.

    The master grid is a regular integer-indexed timeline (time_index, time_ref_s) built from
    the mean EMG sampling interval. All modalities align to this grid via time_index.

    Set force_recompute=True to bypass the cache and re-read from the raw CSV.
    """

    # --- Resolve session reference timestamp ---
    # ts_ref is the Delsys system clock at acquisition start.
    # It anchors time_ref_s to wall-clock time and is used by downstream steps to
    # cross-reference events recorded on other system clocks (BIA, NIRS).
    cache_dir: Path = ctx["CACHE_DIR"]
    ts_ref = pd.to_datetime(participants_df.loc[0, "delsys_ts_init"])

    # --- Register cache paths ---
    # Paths are stored in ctx["parquet_path"] so the notebook dashboard can reference them
    # without hardcoding filenames (e.g. for displaying cache status or forcing re-runs).
    csv_path = raw_emg_dir / f"{run_id}_DELSYS.csv"
    CACHE_MASTER  = cache_dir / "02_master_index_grid.parquet"
    CACHE_EMG     = cache_dir / "02_emg_compact.parquet"
    CACHE_TORQUE  = cache_dir / "02_torque_compact.parquet"

    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_MASTER"] = CACHE_MASTER
    ctx["parquet_path"]["CACHE_EMG"]    = CACHE_EMG
    ctx["parquet_path"]["CACHE_TORQUE"] = CACHE_TORQUE

    # --- Cache check ---
    # All three cache files must exist together — a partial cache is treated as a miss.
    if CACHE_MASTER.exists() and CACHE_EMG.exists() and CACHE_TORQUE.exists() and not force_recompute:
        master_index_grid  = pd.read_parquet(CACHE_MASTER)
        emg_compact_df     = pd.read_parquet(CACHE_EMG)
        torque_compact_df  = pd.read_parquet(CACHE_TORQUE)
        ctx["master_index_grid"] = master_index_grid
        ctx["emg_compact_df"]    = emg_compact_df
        ctx["torque_compact_df"] = torque_compact_df
        ctx["ts_ref"]            = ts_ref
        print(f"[02_emg_import] Cache found — loaded from:\n  {CACHE_MASTER}\n  {CACHE_EMG}\n  {CACHE_TORQUE}")
        return master_index_grid, emg_compact_df, torque_compact_df, ts_ref

    # --- Detect header row and parse column layout ---

    def detect_header_row(csv_path: Path, max_lines: int = 50) -> int:
        """
        Scan the CSV preamble to find the row where the numeric table begins.
        The Delsys CSV starts with metadata lines before the actual signal columns.
        The data header is identified by a line starting with 'X[s],' (time column marker).
        Raises ValueError if not found within the first max_lines lines.
        """
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            for line_index, line in enumerate(fh):
                if line_index >= max_lines:
                    break
                if line.lstrip().startswith("X[s],"):
                    return line_index
        raise ValueError(f"Header row not found in {csv_path} (expected line starting with 'X[s],').")

    def first_index_containing(substrings: list[str], label: str) -> int:
        """
        Return the index of the first column name that contains any of the given substrings.
        Tries each substring in order and returns on first match.
        Raises ValueError with a descriptive message if none are found.
        Used to locate EMG and torque columns by partial name match, since Delsys column
        names can vary slightly between firmware versions.
        """
        for substring in substrings:
            hit = [i for i, name in enumerate(column_names) if substring in name]
            if hit:
                return hit[0]
        raise ValueError(f"Column not found for '{label}'. Tried substrings: {substrings}")

    header_row = detect_header_row(csv_path)

    # Read only the header line to extract column names before loading the full CSV.
    # This avoids reading the entire file twice just to get the column layout.
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        for _ in range(header_row):
            next(fh)
        header_line = next(fh)

    column_names = next(csv.reader([header_line]))

    # Locate the signal columns by partial name match.
    # Each signal has a paired time column at (signal_index - 1).
    torque_signal_index = first_index_containing(["Analog.D"], "torque signal (Analog.D)")
    emg6_signal_index   = first_index_containing(["EMG 6"],    "EMG 6")
    emg8_signal_index   = first_index_containing(["EMG 8"],    "EMG 8")
    emg10_signal_index  = first_index_containing(["EMG 10"],   "EMG 10")

    torque_time_index = torque_signal_index - 1
    emg_time_index    = emg6_signal_index - 1

    # Load only the columns we need to keep memory usage low (the full CSV is large).
    usecols_idx = sorted({
        torque_time_index, torque_signal_index,
        emg_time_index, emg6_signal_index, emg8_signal_index, emg10_signal_index,
    })

    # --- Read CSV ---
    raw = pd.read_csv(
        csv_path,
        header=None,
        skiprows=header_row + 1,
        usecols=usecols_idx,
    )

    colmap = {
        torque_time_index:  "torque_time_s",
        torque_signal_index: "torque_raw",
        emg_time_index:     "emg_time_s",
        emg6_signal_index:  "emg6",
        emg8_signal_index:  "emg8",
        emg10_signal_index: "emg10",
    }
    raw = raw.rename(columns=colmap)

    biodex_raw = raw[["torque_time_s", "torque_raw"]].copy()
    emg_raw    = raw[["emg_time_s", "emg6", "emg8", "emg10"]].copy()

    # --- Build EMG reference grid ---
    # The master timeline is reconstructed as a perfect regular grid at the mean EMG sampling rate.
    # We drop NaN rows first (Delsys can emit a few null rows at the start/end of the recording).

    emg_time_s = pd.to_numeric(emg_raw["emg_time_s"], errors="coerce").to_numpy()
    emg_valid  = ~np.isnan(emg_time_s)

    emg_time_s = emg_time_s[emg_valid]
    emg6  = pd.to_numeric(emg_raw["emg6"],  errors="coerce").to_numpy()[emg_valid]
    emg8  = pd.to_numeric(emg_raw["emg8"],  errors="coerce").to_numpy()[emg_valid]
    emg10 = pd.to_numeric(emg_raw["emg10"], errors="coerce").to_numpy()[emg_valid]

    # Estimate true sampling rate from the mean inter-sample interval.
    # Mean is appropriate here because EMG timestamps are nearly jitter-free.
    fs_ref_hz = 1.0 / float(np.mean(np.diff(emg_time_s)))
    dt_ref_s  = 1.0 / fs_ref_hz
    n_ref     = len(emg_time_s)

    # time_index: integer sample counter starting at 0 — the universal join key across all modalities.
    # time_ref_s: continuous time in seconds from session start, on the perfect regular grid.
    time_index  = np.arange(n_ref, dtype=int)
    time_ref_s  = time_index * dt_ref_s

    master_index_grid = pd.DataFrame({"time_index": time_index, "time_ref_s": time_ref_s})

    emg_compact_df = pd.DataFrame({
        "time_index": time_index,
        "emg6":  emg6,
        "emg8":  emg8,
        "emg10": emg10,
    })

    # --- Map torque onto the EMG grid (nearest-neighbour) ---
    # Torque is sampled at ~100 Hz on a separate Delsys clock, so its timestamps do not fall
    # exactly on EMG grid points. Each torque sample is assigned to the closest EMG time_index.

    torque_time_s = pd.to_numeric(biodex_raw["torque_time_s"], errors="coerce").to_numpy()
    torque_raw    = pd.to_numeric(biodex_raw["torque_raw"],    errors="coerce").to_numpy()

    # Drop any rows where either the time or the signal value is non-finite.
    torque_valid  = np.isfinite(torque_time_s) & np.isfinite(torque_raw)
    torque_time_s = torque_time_s[torque_valid]
    torque_raw    = torque_raw[torque_valid]

    # For each torque timestamp, find the two nearest EMG grid points (left and right)
    # and assign it to whichever is closer.
    ins   = np.searchsorted(emg_time_s, torque_time_s, side="left")
    right = np.clip(ins,     0, n_ref - 1)
    left  = np.clip(ins - 1, 0, n_ref - 1)

    right_dist  = np.abs(emg_time_s[right] - torque_time_s)
    left_dist   = np.abs(emg_time_s[left]  - torque_time_s)
    nearest_idx = np.where(left_dist <= right_dist, left, right).astype(int)

    torque_compact_df = pd.DataFrame({"time_index": nearest_idx, "torque_raw": torque_raw})

    # If two torque samples map to the same grid point (possible at boundaries), keep the last one
    # after sorting by time_index, which corresponds to the temporally later torque reading.
    torque_compact_df = (
        torque_compact_df
        .sort_values("time_index", kind="mergesort")
        .drop_duplicates(subset=["time_index"], keep="last")
        .reset_index(drop=True)
    )

    # --- Write cache ---
    master_index_grid.to_parquet(CACHE_MASTER, index=False)
    emg_compact_df.to_parquet(CACHE_EMG, index=False)
    torque_compact_df.to_parquet(CACHE_TORQUE, index=False)

    # --- Update ctx ---
    ctx["master_index_grid"] = master_index_grid
    ctx["emg_compact_df"]    = emg_compact_df
    ctx["torque_compact_df"] = torque_compact_df
    ctx["ts_ref"]            = ts_ref

    print(f"[02_emg_import] Done — cached outputs to:\n  {CACHE_MASTER}\n  {CACHE_EMG}\n  {CACHE_TORQUE}")
    return master_index_grid, emg_compact_df, torque_compact_df, ts_ref
