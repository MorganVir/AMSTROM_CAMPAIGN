# _03b_sequence_labeling.py - SEQUENCE LABEL APPLICATION
# Apply SEQ phase labels from the picker JSON to the master grid, EMG, and torque tables.
# This step consumes the boundary times saved by _03a_ and stamps every row in the
# master grid with the protocol phase it belongs to (e.g. "MVC_REF", "EX_DYN", "REST2").
#
# Inputs
#   ctx keys : RUN_ID, CACHE_DIR, master_index_grid, emg_compact_df, torque_compact_df
#   cache    : 03a_sequence_picked.json  (boundary times written by the interactive picker)
#
# Outputs (ctx keys updated in place)
#   - master_index_grid  : SEQ and SEQ_index columns added
#   - emg_compact_df     : SEQ and SEQ_index columns propagated via time_index
#   - torque_compact_df  : SEQ and SEQ_index columns propagated via time_index
#
# Cache (overwrites step-02 caches after labeling)
#   - 02_master_index_grid.parquet
#   - 02_emg_compact.parquet
#   - 02_torque_compact.parquet
#
# Notes
#   - Overwrites the step-02 parquet caches in place because SEQ is part of the base grid
#     contract and all downstream steps expect it to be present in those files.
#   - SEQ_index is an integer position into seq_order (0 = "INIT", 1 = first sequence, etc.).
#     It is kept alongside SEQ so downstream code can sort or compare phases by protocol order.
#   - Labeling uses numpy searchsorted on the picked boundary times — O(N log M) where
#     N is grid length and M is number of sequences.

from pathlib import Path
import json
import numpy as np


def run_apply_label_to_grid(
    ctx,
    seq_order,
    force_recompute,
):
    """
    Stamp SEQ phase labels onto the master grid using boundary times from 03a_sequence_picked.json.

    For each row in master_index_grid, assigns the protocol phase label (SEQ) based on
    which interval between picker boundaries it falls into. EMG and torque tables receive
    the same labels via their time_index column (direct numpy array lookup — no join needed).

    Overwrites the 02_* parquet caches with the labeled versions.
    Skips if SEQ is already present in master_index_grid and force_recompute is False.

    Returns (master_index_grid, emg_compact_df, torque_compact_df).
    """

    # --- Resolve ctx ---
    run_id            = ctx["RUN_ID"]
    cache_dir         = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    emg_compact_df    = ctx["emg_compact_df"]
    torque_compact_df = ctx["torque_compact_df"]

    # --- Skip check ---
    # SEQ is already present if _03b_ was run previously and the cache was loaded by _02_.
    if "SEQ" in master_index_grid.columns and not force_recompute:
        print("[03b_sequence_labeling] SEQ already in master_index_grid — skipping. Set force_recompute=True to re-label.")
        return master_index_grid, emg_compact_df, torque_compact_df

    # --- Load picker JSON ---
    pick_json_path = cache_dir / "03a_sequence_picked.json"
    with open(pick_json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    # Guard against applying boundaries that were picked for a different seq_order.
    # This can happen if the operator changes seq_order in the notebook between _03a_ and _03b_.
    if payload["seq_order"] != list(seq_order):
        raise ValueError(
            f"[03b_sequence_labeling] seq_order mismatch between JSON ({payload['seq_order']}) "
            f"and current seq_order ({list(seq_order)}). "
            f"Re-run _03a_ with the current seq_order before applying labels."
        )

    picked_times_s = np.array(payload["picked_times_s"], dtype=float)
    expected_n_bounds = len(seq_order) - 1

    if len(picked_times_s) != expected_n_bounds:
        raise ValueError(
            f"[03b_sequence_labeling] Expected {expected_n_bounds} boundaries "
            f"({len(seq_order)} sequences), got {len(picked_times_s)}. "
            f"Finish picking in _03a_ before running _03b_."
        )

    # --- Label master grid ---
    # searchsorted maps each grid timestamp to the index of the sequence it falls into.
    # Example with seq_order = ["INIT", "MVC_REF", "REST1", "END"] and 3 boundaries [t0, t1, t2]:
    #   time < t0  → index 0 → "INIT"
    #   t0 ≤ time < t1 → index 1 → "MVC_REF"
    #   t1 ≤ time < t2 → index 2 → "REST1"
    #   time ≥ t2  → index 3 → "END"
    time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
    seq_index  = np.searchsorted(picked_times_s, time_ref_s, side="right").astype(int)
    seq_names  = np.array(seq_order, dtype=object)

    master_index_grid["SEQ_index"] = seq_index
    master_index_grid["SEQ"]       = seq_names[seq_index]

    # --- Propagate SEQ to EMG and torque via time_index ---
    # Each EMG/torque row carries a time_index into the master grid.
    # Using it as a direct numpy array index is equivalent to a join on time_index,
    # but faster since time_index values are guaranteed to be valid grid positions.
    master_seq_index_np = master_index_grid["SEQ_index"].to_numpy()
    master_seq_np       = master_index_grid["SEQ"].to_numpy()

    emg_idx    = emg_compact_df["time_index"].to_numpy(dtype=int)
    torque_idx = torque_compact_df["time_index"].to_numpy(dtype=int)

    emg_compact_df["SEQ_index"]    = master_seq_index_np[emg_idx]
    emg_compact_df["SEQ"]          = master_seq_np[emg_idx]

    torque_compact_df["SEQ_index"] = master_seq_index_np[torque_idx]
    torque_compact_df["SEQ"]       = master_seq_np[torque_idx]

    print(f"[03b_sequence_labeling] SEQ labels applied: {list(master_index_grid['SEQ'].unique())}")
    print(f"[03b_sequence_labeling] Boundaries used: {len(picked_times_s)} / {expected_n_bounds}")

    # --- Overwrite 02 caches with labeled versions ---
    # The 02 caches are the canonical source for downstream steps; SEQ must be baked in.
    cache_02_master = cache_dir / "02_master_index_grid.parquet"
    cache_02_emg    = cache_dir / "02_emg_compact.parquet"
    cache_02_torque = cache_dir / "02_torque_compact.parquet"

    master_index_grid.to_parquet(cache_02_master, index=False)
    emg_compact_df.to_parquet(cache_02_emg, index=False)
    torque_compact_df.to_parquet(cache_02_torque, index=False)

    print("[03b_sequence_labeling] Done — 02_* caches updated with SEQ columns.")
    return master_index_grid, emg_compact_df, torque_compact_df
