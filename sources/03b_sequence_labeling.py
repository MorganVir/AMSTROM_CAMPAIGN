# 04b_seq_labeling - apply SEQ labels to master_index_grid
# Input : master_index_grid, RUN_ID, CACHE_DIR, SEQ_ORDER
# Reads : 03a_sequence_picked.json
# Output: master_index_grid with SEQ_index + SEQ


from pathlib import Path
import json
import numpy as np

assert "RUN_ID" in globals(), "RUN_ID missing"
assert "CACHE_DIR" in globals(), "CACHE_DIR missing"
assert isinstance(CACHE_DIR, Path), "CACHE_DIR must be a pathlib.Path"

assert "master_index_grid" in globals(), "master_index_grid missing"
assert "SEQ_ORDER" in globals(), "SEQ_ORDER missing"

PICK_JSON_PATH = CACHE_DIR / "03a_sequence_picked.json"
assert PICK_JSON_PATH.exists(), f"Missing pick file: {PICK_JSON_PATH}"

# ----------------------------
# Load picks
# ----------------------------
with open(PICK_JSON_PATH, "r", encoding="utf-8") as file_handle:
    pick_payload = json.load(file_handle)

picked_run_id = pick_payload.get("run_id", None)
assert picked_run_id == RUN_ID, f"run_id mismatch in JSON: {picked_run_id} vs {RUN_ID}"

picked_seq_order = pick_payload.get("seq_order", None)
assert picked_seq_order == list(SEQ_ORDER), "seq_order mismatch between JSON and current SEQ_ORDER"

picked_times_s = np.array(pick_payload.get("picked_times_s", []), dtype=float)

expected_n_bounds = len(SEQ_ORDER) - 1
assert len(picked_times_s) == expected_n_bounds, (len(picked_times_s), expected_n_bounds)
assert np.all(np.isfinite(picked_times_s)), "picked_times_s contains non-finite values"
assert np.all(np.diff(picked_times_s) > 0), "picked_times_s must be strictly increasing"


# Labeling 
# idx=0 for INIT (t < bounds[0])
# idx=1 for WU ...
# idx=len(bounds) for END (t >= bounds[-1])

time_ref_s = master_index_grid["time_ref_s"].to_numpy(dtype=float)
seq_index = np.searchsorted(picked_times_s, time_ref_s, side="right").astype(int)

seq_names = np.array(SEQ_ORDER, dtype=object)
master_index_grid["SEQ_index"] = seq_index
master_index_grid["SEQ"] = seq_names[seq_index]

# print
unique_seq = list(master_index_grid["SEQ"].unique())
print("SEQ labels present:", unique_seq)
print("Applied boundaries:", len(picked_times_s), "Expected:", expected_n_bounds)

del time_ref_s, seq_index, seq_names, picked_times_s
