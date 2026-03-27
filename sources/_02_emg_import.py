# from __future__ import annotations

from pathlib import Path
import csv
import json
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
    02 — EMG import
    Outputs:
      master_index_grid, emg_compact_df, torque_compact_df, ts_ref

    Caches (step-owned, in ctx['CACHE_DIR']):
      02_master_index_grid.parquet
      02_emg_compact.parquet
      02_torque_compact.parquet
    """

    # Fail-loud inputs
    if "CACHE_DIR" not in ctx or ctx["CACHE_DIR"] is None:
        raise KeyError("[02_emg_import] missing ctx['CACHE_DIR']")
    cache_dir: Path = ctx["CACHE_DIR"]
    if not isinstance(cache_dir, Path):
        raise TypeError(f"[02_emg_import] ctx['CACHE_DIR'] must be pathlib.Path, got {type(cache_dir)}")

    if not isinstance(raw_emg_dir, Path):
        raise TypeError("raw_emg_dir must be pathlib.Path")
    if not raw_emg_dir.exists():
        raise FileNotFoundError(f"RAW_EMG_DIR not found: {raw_emg_dir}")

    if not isinstance(run_id, str) or len(run_id) < 7:
        raise ValueError("run_id invalid (expected RUN_ID string with >= 7 chars)")

    if not isinstance(participants_df, pd.DataFrame):
        raise TypeError("participants_df must be a pandas DataFrame")

    #definit the time reference (ts_ref) as the delsys_ts_init timestamp from participants_df
    assert "delsys_ts_init" in participants_df.columns, "participants_df missing 'delsys_ts_init'"
    ts_ref = pd.to_datetime(participants_df.loc[0, "delsys_ts_init"])
    assert pd.notna(ts_ref), "participants_df.loc[0, 'delsys_ts_init'] is NaT or invalid timestamp"

    # Cache paths
    csv_path = raw_emg_dir / f"{run_id}_DELSYS.csv"
    CACHE_MASTER = cache_dir / "02_master_index_grid.parquet"
    CACHE_EMG = cache_dir / "02_emg_compact.parquet"
    CACHE_TORQUE = cache_dir / "02_torque_compact.parquet"

    # register specific cache paths (so dashboard commit can reuse them without hardcoding )
    ctx.setdefault("parquet_path", {})
    ctx["parquet_path"]["CACHE_MASTER"] = CACHE_MASTER
    ctx["parquet_path"]["CACHE_EMG"] = CACHE_EMG
    ctx["parquet_path"]["CACHE_TORQUE"] = CACHE_TORQUE

   
    # Cache hit
    if (CACHE_MASTER.exists() and CACHE_EMG.exists() and CACHE_TORQUE.exists()) and (not force_recompute):
        master_index_grid = pd.read_parquet(CACHE_MASTER)
        emg_compact_df = pd.read_parquet(CACHE_EMG)
        torque_compact_df = pd.read_parquet(CACHE_TORQUE)
        ctx["master_index_grid"] = master_index_grid
        ctx["emg_compact_df"] = emg_compact_df
        ctx["torque_compact_df"] = torque_compact_df
        ctx["ts_ref"] = ts_ref
        print(f"[02_emg_import] Cache found : stored files have been loaded.\n{CACHE_MASTER}, \n{CACHE_EMG}, \n{CACHE_TORQUE}")
        return master_index_grid, emg_compact_df, torque_compact_df, ts_ref

  
    # Resolve delsys_ts_init -> ts_ref 
    assert isinstance(participants_df, pd.DataFrame), "participants_df missing/invalid"
    assert participants_df.shape[0] == 1, f"Expected 1-row participants_df, got shape={participants_df.shape}"
    assert "delsys_ts_init" in participants_df.columns, "participants_df missing column: delsys_ts_init"

    ts_ref = pd.to_datetime(participants_df.loc[0, "delsys_ts_init"])

    # -------------------------
    # Detect header row (start of numeric table)
    # -------------------------
    def detect_header_row(csv_path: Path, max_lines: int = 50) -> int:
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            for line_index, line in enumerate(fh):
                if line_index >= max_lines:
                    break
                if line.lstrip().startswith("X[s],"):
                    return line_index
        raise ValueError("Header row not found (expected line starting with 'X[s],').")

    header_row = detect_header_row(csv_path)

    # Read header line and parse as CSV
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        for _ in range(header_row):
            next(fh)
        header_line = next(fh)

    column_names = next(csv.reader([header_line]))

    def first_index_containing(substrings: list[str], label: str) -> int:
        for substring in substrings:
            hit = [i for i, name in enumerate(column_names) if substring in name]
            if hit:
                return hit[0]
        raise ValueError(f"Missing {label}. Tried: {substrings}")

    # Resolve indices
    torque_signal_index = first_index_containing(["Analog.D"], "torque signal (Analog.D)")
    emg6_signal_index = first_index_containing(["EMG 6"], "EMG 6")
    emg8_signal_index = first_index_containing(["EMG 8"], "EMG 8")
    emg10_signal_index = first_index_containing(["EMG 10"], "EMG 10")

    torque_time_index = torque_signal_index - 1
    emg_time_index = emg6_signal_index - 1

    usecols_idx = sorted({
        torque_time_index, torque_signal_index,
        emg_time_index, emg6_signal_index, emg8_signal_index, emg10_signal_index
    })

    raw = pd.read_csv(
        csv_path,
        header=None,
        skiprows=header_row + 1,
        usecols=usecols_idx,
    )

    colmap = {
        torque_time_index: "torque_time_s",
        torque_signal_index: "torque_raw",
        emg_time_index: "emg_time_s",
        emg6_signal_index: "emg6",
        emg8_signal_index: "emg8",
        emg10_signal_index: "emg10",
    }
    raw = raw.rename(columns=colmap)

    biodex_raw = raw[["torque_time_s", "torque_raw"]].copy()
    emg_raw = raw[["emg_time_s", "emg6", "emg8", "emg10"]].copy()


    # Build EMG reference grid

    emg_time_s = pd.to_numeric(emg_raw["emg_time_s"], errors="coerce").to_numpy()
    emg_valid = ~np.isnan(emg_time_s)

    emg_time_s = emg_time_s[emg_valid]
    emg6 = pd.to_numeric(emg_raw["emg6"], errors="coerce").to_numpy()[emg_valid]
    emg8 = pd.to_numeric(emg_raw["emg8"], errors="coerce").to_numpy()[emg_valid]
    emg10 = pd.to_numeric(emg_raw["emg10"], errors="coerce").to_numpy()[emg_valid]

    fs_ref_hz = 1.0 / float(np.mean(np.diff(emg_time_s)))
    dt_ref_s = 1.0 / fs_ref_hz
    n_ref = int(len(emg_time_s))

    time_index = np.arange(n_ref, dtype=int)
    time_ref_s = time_index * dt_ref_s

    master_index_grid = pd.DataFrame({"time_index": time_index, "time_ref_s": time_ref_s})

    emg_compact_df = pd.DataFrame({
        "time_index": time_index,
        "emg6": emg6,
        "emg8": emg8,
        "emg10": emg10,
    })


    # Torque -> ref grid (nearest EMG timestamp)
    torque_time_s = pd.to_numeric(biodex_raw["torque_time_s"], errors="coerce").to_numpy()
    torque_raw = pd.to_numeric(biodex_raw["torque_raw"], errors="coerce").to_numpy()

    torque_valid = np.isfinite(torque_time_s) & np.isfinite(torque_raw)
    torque_time_s = torque_time_s[torque_valid]
    torque_raw = torque_raw[torque_valid]

    ins = np.searchsorted(emg_time_s, torque_time_s, side="left")
    right = np.clip(ins, 0, n_ref - 1)
    left = np.clip(ins - 1, 0, n_ref - 1)

    right_dist = np.abs(emg_time_s[right] - torque_time_s)
    left_dist = np.abs(emg_time_s[left] - torque_time_s)

    nearest_idx = np.where(left_dist <= right_dist, left, right).astype(int)

    torque_compact_df = pd.DataFrame({"time_index": nearest_idx, "torque_raw": torque_raw})
    torque_compact_df = (
        torque_compact_df
        .sort_values("time_index", kind="mergesort")
        .drop_duplicates(subset=["time_index"], keep="last")
        .reset_index(drop=True)
    )


    # Commit
    master_index_grid.to_parquet(CACHE_MASTER, index=False)
    emg_compact_df.to_parquet(CACHE_EMG, index=False)
    torque_compact_df.to_parquet(CACHE_TORQUE, index=False)


    # CTX updates (corridor plumbing)
    ctx["master_index_grid"] = master_index_grid
    ctx["emg_compact_df"] = emg_compact_df
    ctx["torque_compact_df"] = torque_compact_df
    ctx["ts_ref"] = ts_ref
    print(f"[02_emg_import] Cached outputs to :\n {CACHE_MASTER}, \n{CACHE_EMG}, \n{CACHE_TORQUE}")

    return master_index_grid, emg_compact_df, torque_compact_df, ts_ref