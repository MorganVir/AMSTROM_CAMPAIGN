# 03b_sequence_labeling.py
# Apply SEQ labels to master_index_grid (and propagate to EMG / torque)

from pathlib import Path
import json
import numpy as np


def run_apply_label_to_grid(
    ctx,
    seq_order,
    force_recompute,
):
    """
    Apply SEQ labels to master_index_grid using 03a_sequence_picked.json.

    Parameters
    ----------
    ctx : dict
        Must contain:
            - RUN_ID
            - CACHE_DIR
            - master_index_grid
            - emg_compact_df
            - torque_compact_df

    seq_order : list[str]
        Ordered SEQ labels

    force_recompute : bool
        If False and SEQ already present, skip
    """

    assert isinstance(ctx, dict)
    required = [
        "RUN_ID",
        "CACHE_DIR",
        "master_index_grid",
        "emg_compact_df",
        "torque_compact_df",
    ]
    for key in required:
        assert key in ctx, f"{key} missing from ctx"

    run_id = ctx["RUN_ID"]
    cache_dir = ctx["CACHE_DIR"]
    master_index_grid = ctx["master_index_grid"]
    emg_compact_df = ctx["emg_compact_df"]
    torque_compact_df = ctx["torque_compact_df"]

    assert isinstance(cache_dir, Path)


    # Skip logic
    if "SEQ" in master_index_grid.columns and not force_recompute:
        print("[03b_sequence_labeling] SEQ already present in master_index_grid. Skipping. Force recompute or delete cache to re-apply SEQ labels.")
        return master_index_grid, emg_compact_df, torque_compact_df


    # Load boundary JSON (generated from previous step)
    pick_json_path = cache_dir / "03a_sequence_picked.json"
    assert pick_json_path.exists(), f"Missing pick file: {pick_json_path}"

    with open(pick_json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload["run_id"] == run_id, (
        payload["run_id"],
        run_id,
    )

    assert payload["seq_order"] == list(seq_order), (
        "seq_order mismatch between JSON and current SEQ_ORDER ! recompute 03a_sequence_picker with current SEQ_ORDER",
    )

    picked_times_s = np.array(payload["picked_times_s"], dtype=float)

    expected_n_bounds = len(seq_order) - 1
    assert len(picked_times_s) == expected_n_bounds
    assert np.all(np.isfinite(picked_times_s))
    assert np.all(np.diff(picked_times_s) > 0)


    # Label master b
    time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)

    seq_index = np.searchsorted(
        picked_times_s,
        time_ref_s,
        side="right",
    ).astype(int)

    seq_names = np.array(seq_order, dtype=object)

    master_index_grid["SEQ_index"] = seq_index
    master_index_grid["SEQ"] = seq_names[seq_index]


    # Propagate via time_index
    master_seq_index_np = master_index_grid["SEQ_index"].to_numpy()
    master_seq_np = master_index_grid["SEQ"].to_numpy()

    emg_idx = emg_compact_df["time_index"].to_numpy(dtype=int)
    torque_idx = torque_compact_df["time_index"].to_numpy(dtype=int)

    emg_compact_df["SEQ_index"] = master_seq_index_np[emg_idx]
    emg_compact_df["SEQ"] = master_seq_np[emg_idx]

    torque_compact_df["SEQ_index"] = master_seq_index_np[torque_idx]
    torque_compact_df["SEQ"] = master_seq_np[torque_idx]

    unique_seq = list(master_index_grid["SEQ"].unique())
    print("SEQ labels present:", unique_seq)
    print("Applied boundaries:", len(picked_times_s), "Expected:", expected_n_bounds)


    # Commit (overwrite 02 caches !)
    cache_02_master = cache_dir / "02_master_index_grid.parquet"
    cache_02_emg = cache_dir / "02_emg_compact.parquet"
    cache_02_torque = cache_dir / "02_torque_compact.parquet"

    master_index_grid.to_parquet(cache_02_master, index=False)
    emg_compact_df.to_parquet(cache_02_emg, index=False)
    torque_compact_df.to_parquet(cache_02_torque, index=False)

    print("[03b_sequence_labeling] Updated caches with SEQ columns.")

    return master_index_grid, emg_compact_df, torque_compact_df