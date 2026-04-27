# _09_export.py - FINAL EXPORT
# Write final QC pdf and selected parquet files from step caches.
#
# This is the last step of the Python preprocessing pipeline.
# It takes the intermediate cache files produced by steps _01_ to _07_,
# renames them to their canonical downstream names, stamps metadata columns
# (run_id, period_s, sequence_valid flags), and writes the final parquet files
# that R will read for analysis.
#
# Inputs
#   export_final_qc_pdf:
#     - qc_figs   : list of matplotlib Figure objects already built upstream
#     - run_id    : participant run identifier (e.g. "011BeSa_20251023")
#
#   export_final_parquet:
#     - ctx with RUN_ID and CACHE_DIR
#     - export_selection = "all" or list of specific cache parquet names to export
#     - sequence_valid_by_target (optional): validity overrides per modality and sequence
#
# Outputs
#   - results/QC_EXPORT/QC_PLOT_EXPORT_<RUN_ID>.pdf
#   - results/DATA_EXPORT/<RUN_ID>/<renamed parquet files>
#
# Notes
#   - Cache filenames use step-prefixed names (e.g. "02_emg_compact.parquet").
#     Exported filenames use clean downstream names (e.g. "preprocessed_emg.parquet").
#   - Every exported table receives a run_id column so downstream R can identify the participant.
#   - The export folder always uses the full RUN_ID (not the 7-char subject key).

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# ── Constants ─────────────────────────────────────────────────────────────────

# Aliases for the two cache files that need special handling during export
# (EMG gets per-channel validity flags; master grid gets a period_s column).
PARQUET_MASTER_INDEX_GRID = "02_master_index_grid.parquet"
PARQUET_EMG_COMPACT = "02_emg_compact.parquet"

# Full list of cache parquet files to include in the default "all" export.
# Order here determines the order they are processed and reported.
PARQUET_EXPORT_LIST = [
    "01_participants.parquet",
    PARQUET_EMG_COMPACT,
    PARQUET_MASTER_INDEX_GRID,
    "02_torque_compact.parquet",
    "05a_bia2_freqs_hz.parquet",
    "05a_bia4_freqs_hz.parquet",
    "05b_bia2_compact_aligned.parquet",
    "05b_bia4_compact_aligned.parquet",
    "06b_nirs_compact_aligned.parquet",
    "07_myoton_compact.parquet",
]

# Mapping from internal cache name → canonical downstream name used by R.
# Files not listed here keep their cache name unchanged.
PARQUET_EXPORT_RENAME = {
    "01_participants.parquet":          "participant.parquet",
    PARQUET_MASTER_INDEX_GRID:          "master_index_grid.parquet",
    PARQUET_EMG_COMPACT:                "preprocessed_emg.parquet",
    "02_torque_compact.parquet":        "preprocessed_torque.parquet",
    "05a_bia2_freqs_hz.parquet":        "preprocessed_bia2_freqs_hz.parquet",
    "05a_bia4_freqs_hz.parquet":        "preprocessed_bia4_freqs_hz.parquet",
    "05b_bia2_compact_aligned.parquet": "preprocessed_bia2.parquet",
    "05b_bia4_compact_aligned.parquet": "preprocessed_bia4.parquet",
    "06b_nirs_compact_aligned.parquet": "preprocessed_nirs.parquet",
    "07_myoton_compact.parquet":        "preprocessed_myoton.parquet",
}

# All modality targets that can receive a sequence_valid flag.
# EMG targets are per-channel (emg6/8/10 → three separate sequence_valid columns).
# All other targets are per-parquet (one sequence_valid column per file).
# This split is intentional and reflected in EMG_SEQUENCE_VALID_COLUMNS vs PARQUET_SEQUENCE_VALID_TARGET.
SEQUENCE_VALID_TARGETS = [
    "emg6",
    "emg8",
    "emg10",
    "torque",
    "bia",
    "nirs",
    "myoton",
]

# Maps each EMG channel target to its output column name in the exported EMG file.
# Downstream R uses these columns to filter out sequences where a specific channel was bad.
EMG_SEQUENCE_VALID_COLUMNS = {
    "emg6":  "sequence_valid_emg6",
    "emg8":  "sequence_valid_emg8",
    "emg10": "sequence_valid_emg10",
}

# Maps each non-EMG cache file to its modality target key.
# Used to look up the correct validity map when stamping the sequence_valid column.
# BIA2 and BIA4 share the same "bia" validity map since they are recorded simultaneously.
PARQUET_SEQUENCE_VALID_TARGET = {
    "02_torque_compact.parquet":        "torque",
    "05b_bia2_compact_aligned.parquet": "bia",
    "05b_bia4_compact_aligned.parquet": "bia",
    "06b_nirs_compact_aligned.parquet": "nirs",
    "07_myoton_compact.parquet":        "myoton",
}


# ── Helper functions ───────────────────────────────────────────────────────────

def build_master_seq_lookup(master_index_grid: pd.DataFrame) -> tuple[dict[int, str], list[str]]:
    """
    Extract an ordered SEQ lookup from the master grid.
    Returns (SEQ_index → SEQ label dict, ordered list of SEQ labels).
    Used to resolve integer indices back to protocol phase labels when stamping validity flags.
    """
    # Keep one row per unique sequence, then sort by SEQ_index to preserve protocol order.
    # mergesort is used to guarantee a stable sort (same output regardless of original row order).
    seq_pairs = master_index_grid[["SEQ", "SEQ_index"]].drop_duplicates().copy()
    seq_pairs["SEQ"] = seq_pairs["SEQ"].astype(str)
    seq_pairs["SEQ_index"] = seq_pairs["SEQ_index"].astype(int)

    seq_pairs = seq_pairs.sort_values("SEQ_index", kind="mergesort").reset_index(drop=True)

    # Build the integer→label lookup and the plain ordered list of labels.
    seq_index_to_seq = dict(zip(seq_pairs["SEQ_index"], seq_pairs["SEQ"]))
    seq_values = list(seq_pairs["SEQ"])
    return seq_index_to_seq, seq_values


def build_sequence_valid_maps(
    *,
    ctx: dict,
    sequence_valid_overrides: dict | None = None,
) -> dict[str, dict[str, int]]:
    """
    Build a {target → {SEQ → 0|1}} validity map, defaulting to 1 (valid) for all sequences.
    Overrides are applied on top to mark specific target/SEQ combinations as 0 (excluded).
    Overrides come from the notebook export cell where the user flags bad or incomplete sequences.
    """
    master_index_grid = ctx["master_index_grid"]
    _, seq_values = build_master_seq_lookup(master_index_grid)

    # Default: every target is valid (1) for every sequence in the protocol.
    # This means "include all data" unless the user explicitly overrides something.
    sequence_valid_by_target = {
        target: {seq: 1 for seq in seq_values} for target in SEQUENCE_VALID_TARGETS
    }

    if sequence_valid_overrides is None:
        return sequence_valid_by_target

    # Apply per-target overrides: typically used to mark a corrupted or missing sequence as 0.
    # Example: {"nirs": {"EX_DYN": 0}} would flag NIRS during the dynamic exercise as invalid.
    for target_key, overrides_for_target in sequence_valid_overrides.items():
        target = str(target_key)
        for seq_key, raw_value in overrides_for_target.items():
            seq = str(seq_key)
            sequence_valid_by_target[target][seq] = int(raw_value)

    return sequence_valid_by_target


def resolve_seq_series(
    *,
    table: pd.DataFrame,
    parquet_name: str,
    seq_index_to_seq: dict[int, str],
) -> pd.Series | None:
    """
    Return a string SEQ Series for a table, or None if no sequence column is present.
    Prefers the resolved SEQ label; falls back to mapping from SEQ_index if SEQ is absent.
    Returns None for tables that carry no sequence information (e.g. participant metadata).
    """
    has_seq = "SEQ" in table.columns
    has_seq_index = "SEQ_index" in table.columns

    # Tables like participant.parquet have no time-series structure — skip them.
    if not has_seq and not has_seq_index:
        return None

    # Most tables written by _09_ already carry a resolved string SEQ column — use it directly.
    if has_seq:
        return pd.Series(table["SEQ"]).astype(str)

    # SEQ_index fallback: some cache tables carry the integer index rather than the resolved label
    # (e.g. tables written before _03b_ runs the labeling pass). Map integers back to labels here.
    seq_index_series = pd.Series(table["SEQ_index"]).astype(int)
    seq_from_index = seq_index_series.map(seq_index_to_seq)
    return seq_from_index


def append_sequence_valid_columns(
    *,
    table: pd.DataFrame,
    parquet_name: str,
    sequence_valid_by_target: dict[str, dict[str, int]] | None,
    seq_index_to_seq: dict[int, str],
) -> pd.DataFrame:
    """
    Stamp sequence_valid flags onto a cache table before export.
    EMG gets one flag per channel (sequence_valid_emg6/8/10).
    All other modalities get a single sequence_valid column.
    Returns the original table unchanged if no validity map is provided or the table has no SEQ column.
    """
    # If no validity map was built (user passed None), skip flagging entirely.
    if sequence_valid_by_target is None:
        return table

    # Resolve which sequence each row belongs to. Returns None for non-time-series tables.
    seq_series = resolve_seq_series(
        table=table,
        parquet_name=parquet_name,
        seq_index_to_seq=seq_index_to_seq,
    )
    if seq_series is None:
        return table

    table_out = table.copy()

    # EMG branch: one validity column per channel so downstream R can filter each channel independently.
    # A sequence might be valid for emg6 but invalid for emg10 (e.g. one electrode came loose).
    if parquet_name == PARQUET_EMG_COMPACT:
        for target, column_name in EMG_SEQUENCE_VALID_COLUMNS.items():
            sequence_valid = seq_series.map(sequence_valid_by_target[target])
            table_out[column_name] = sequence_valid.astype(int)
        return table_out

    # Generic branch: one sequence_valid column per file (torque, BIA, NIRS, Myoton).
    # Tables not listed in PARQUET_SEQUENCE_VALID_TARGET (e.g. participant metadata) are returned unchanged.
    target = PARQUET_SEQUENCE_VALID_TARGET.get(parquet_name)
    if target is None:
        return table

    sequence_valid = seq_series.map(sequence_valid_by_target[target])
    table_out["sequence_valid"] = sequence_valid.astype(int)
    return table_out


def infer_master_period_s(table: pd.DataFrame) -> float:
    """
    Estimate the nominal sampling period from the master grid time column.
    Uses median over consecutive diffs to stay robust against occasional irregular gaps in the EMG clock.
    """
    # Compute time differences between consecutive samples (in seconds).
    # The median is used instead of mean because a few large gaps (e.g. at sequence boundaries)
    # would skew the mean, whereas the median reflects the typical inter-sample interval.
    master_time_ref_s = table["time_ref_s"].to_numpy(dtype=float)
    master_time_deltas = np.diff(master_time_ref_s)

    period_s = float(np.median(master_time_deltas))
    return period_s


# ── Export functions ───────────────────────────────────────────────────────────

def export_final_qc_pdf(
    *,
    qc_figs: list,
    run_id: str,
    results_dir: Path = Path("results"),
) -> Path:
    """
    Bundle all QC figures for the current run into a single PDF file.
    Figures must already be built before calling this function — it only handles the writing.
    Output path: results/QC_EXPORT/QC_PLOT_EXPORT_<RUN_ID>.pdf
    """
    qc_export_root = results_dir / "QC_EXPORT"
    qc_export_root.mkdir(parents=True, exist_ok=True)

    qc_pdf_path = qc_export_root / f"QC_PLOT_EXPORT_{run_id}.pdf"

    # PdfPages appends each figure as a separate page in the PDF.
    with PdfPages(qc_pdf_path) as pdf:
        for fig in qc_figs:
            pdf.savefig(fig)

    print(f"[QC EXPORT] PDF created -> {qc_pdf_path.resolve()}")
    return qc_pdf_path


def export_final_parquet(
    *,
    ctx: dict,
    export_selection="all",
    sequence_valid_by_target: dict[str, dict[str, int]] | None = None,
    run_id: str | None = None,
    cache_dir: Path | None = None,
    results_dir: Path = Path("results"),
) -> Path:
    """
    Export selected cache parquet tables to results/DATA_EXPORT/<RUN_ID>/.
    For each file: appends sequence_valid flags, stamps run_id, and (for master_index_grid) computes period_s.
    Cache filenames are renamed to their canonical downstream names via PARQUET_EXPORT_RENAME.
    """

    # --- Resolve parameters from ctx if not passed explicitly ---
    # run_id and cache_dir can be passed directly for testing, otherwise they come from the pipeline context.
    if run_id is None:
        run_id = ctx["RUN_ID"]
    if cache_dir is None:
        cache_dir = ctx["CACHE_DIR"]

    # Pre-build the SEQ label lookup once so it can be reused across all files in the loop.
    seq_index_to_seq, _ = build_master_seq_lookup(ctx["master_index_grid"])

    # --- Resolve file list ---
    # None is treated as "all" so notebook callers can pass an unset variable without error.
    if export_selection == "all" or export_selection is None:
        parquet_names = list(PARQUET_EXPORT_LIST)
    else:
        # Partial export: the user selected only specific cache files (e.g. to re-export one modality).
        parquet_names = [str(name) for name in export_selection]

    data_export_root = results_dir / "DATA_EXPORT" / run_id
    data_export_root.mkdir(parents=True, exist_ok=True)

    print("Parquet files selected for export:")
    for parquet_name in parquet_names:
        export_name = PARQUET_EXPORT_RENAME.get(parquet_name, parquet_name)
        print(f"- {parquet_name} -> {export_name}")

    # --- Per-file export loop ---
    # For each selected cache file: read → stamp validity → add metadata → write → log.
    exported_columns_by_file: dict[str, list[str]] = {}
    for parquet_name in parquet_names:
        source_path = cache_dir / parquet_name

        # Read from cache and stamp sequence_valid flags.
        # The result may be the same object as source_table if no flags were added.
        source_table = pd.read_parquet(source_path)
        exported_table = append_sequence_valid_columns(
            table=source_table,
            parquet_name=parquet_name,
            sequence_valid_by_target=sequence_valid_by_target,
            seq_index_to_seq=seq_index_to_seq,
        )
        # Track which sequence_valid columns were added so the summary print stays accurate.
        added_validity_columns = [
            column_name
            for column_name in exported_table.columns
            if column_name not in source_table.columns and column_name.startswith("sequence_valid")
        ]

        # Add metadata columns.
        # Copy first: append_sequence_valid_columns may return the original object if nothing was added.
        period_s = None
        exported_table = exported_table.copy()
        if parquet_name == PARQUET_MASTER_INDEX_GRID:
            # period_s is the nominal inter-sample interval in seconds (e.g. ~0.000466 s at ~2148 Hz).
            # It is derived here rather than upstream so it always reflects the final exported grid.
            period_s = infer_master_period_s(exported_table)
            exported_table["period_s"] = period_s
        # run_id is added to every exported file so downstream R can always identify the participant,
        # even when tables from multiple runs are pooled together in the cohort analysis.
        exported_table["run_id"] = run_id

        # Write to DATA_EXPORT under the canonical downstream filename.
        export_name = PARQUET_EXPORT_RENAME.get(parquet_name, parquet_name)
        export_path = data_export_root / export_name
        exported_table.to_parquet(export_path, index=False)

        # Log what was written and which extra columns were added, for traceability.
        print(f"[DATA EXPORT] Wrote -> {export_path}")
        summary_parts = [
            f"RUN_ID={run_id}",
            f"sequence_valid_added={'yes' if len(added_validity_columns) != 0 else 'no'}",
            f"period_s_added={'yes' if period_s is not None else 'no'}",
        ]
        if len(added_validity_columns) != 0:
            summary_parts.append(f"sequence_valid_columns={added_validity_columns}")
        if period_s is not None:
            summary_parts.append(f"period_s={period_s:.15f}")
        print(f"[DATA EXPORT CHECK] {export_name} | " + " | ".join(summary_parts))

        # Record the final column list for the summary printed after the loop.
        exported_columns_by_file[export_name] = list(exported_table.columns)

    print("\nExported parquet files and their columns:")
    for export_name, columns in exported_columns_by_file.items():
        print(f"- {export_name}: columns = {columns}")

    print(f"\n[DATA EXPORT COMPLETE] -> {data_export_root.resolve()}")
    return data_export_root
