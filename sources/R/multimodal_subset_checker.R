# multimodal_subset_checker.R -- builds the subset registry every notebook reads to decide which
# runs are in scope. For each (analysis_block x run) it records included 0/1 + a reason, then
# derives the canonical `cohort_final` set (four-way modality intersection minus the subject-level
# exclusions in R/config/cohort_config.R). Writes the registry parquet + csv. Run from the project
# root after (re)processing runs; the notebooks read this output and never recompute it.
library(arrow)
library(dplyr)
library(tibble)

export_root <- here::here("results", "DATA_EXPORT")
subset_root <- here::here("results", "R_ANALYSIS", "subset_registry")
dir.create(subset_root, recursive = TRUE, showWarnings = FALSE)

critical_seqs <- c("MVC_REF", "SVC_REF", "EX_DYN", "EX_STA", "MVC_RECOV_DYN", "MVC_RECOV_STA")
bia_target_freqs_hz <- c(9760L, 48800L, 97600L)

# Which analysis blocks exist, and whether each is "frozen" (ready to gate runs) or still a
# placeholder. Non-frozen blocks are always recorded as excluded.
analysis_block_registry <- tibble::tribble(
  ~analysis_family, ~analysis_block,  ~block_frozen, ~detail,
  "emg",           "emg_cohort",     TRUE,          "Existing EMG pathway already established.",
  "bia",           "bia_individual", TRUE,          "Provisional first-pass BIA starter block. Revisit after bibliography freeze.",
  "bia",           "bia_cohort",     TRUE,          "Provisional first-pass BIA starter block. Revisit after bibliography freeze.",
  "nirs",          "nirs_individual", TRUE,         "",
  "nirs",          "nirs_cohort",     FALSE,        "Waiting for bibliography freeze.",
  "myoton",        "myoton_individual", FALSE,      "Waiting for bibliography freeze.",
  "myoton",        "myoton_cohort",     FALSE,      "Waiting for bibliography freeze.",
  "cross",         "emg_bia_cross",  FALSE,         "Waiting for bibliography freeze of the EMG x BIA summary mapping."
)

run_ids <- list.dirs(export_root, recursive = FALSE, full.names = FALSE)
run_ids <- run_ids[nzchar(run_ids)]

if (length(run_ids) == 0L) {
  stop("No run folders found under: ", normalizePath(export_root, mustWork = FALSE))
}

# Build one registry row (family / block / run / included 0-1 / reason / detail).
make_result <- function(analysis_family, analysis_block, run_id, included, reason, detail = "") {
  return(tibble::tibble(
    analysis_family = analysis_family,
    analysis_block  = analysis_block,
    RUN_ID          = run_id,
    included        = as.integer(included),
    reason          = reason,
    detail          = detail
  ))
}

# Which of file_names are missing for this run (character(0) = all present).
check_files_exist <- function(run_id, file_names) {
  missing <- file_names[!file.exists(file.path(export_root, run_id, file_names))]
  return(if (length(missing) == 0L) {
    character(0)
  } else {
    missing
  })
}

# read_parquet that returns the error object instead of throwing.
safe_read <- function(path, cols) {
  return(tryCatch(
    arrow::read_parquet(path, col_select = cols),
    error = function(e) e
  ))
}

# NULL if the file exposes all required_cols, otherwise the read-error message.
check_required_columns <- function(run_id, file_name, required_cols) {
  df_or_err <- safe_read(file.path(export_root, run_id, file_name), required_cols)
  if (inherits(df_or_err, "error")) {
    return(conditionMessage(df_or_err))
  }
  return(NULL)
}

# Gate on sequence validity: all critical sequences present and seq_valid_col != 0.
check_validity_file <- function(run_id, file_name, seq_valid_col) {
  df_or_err <- safe_read(
    file.path(export_root, run_id, file_name),
    c("SEQ", seq_valid_col)
  )

  if (inherits(df_or_err, "error")) {
    return(list(ok = FALSE, reason = "read_error", detail = conditionMessage(df_or_err)))
  }

  validity <- df_or_err |>
    dplyr::distinct(.data$SEQ, .data[[seq_valid_col]]) |>
    dplyr::filter(.data$SEQ %in% critical_seqs)

  missing_seqs <- setdiff(critical_seqs, validity$SEQ)
  if (length(missing_seqs) > 0L) {
    return(list(
      ok = FALSE,
      reason = "missing_sequence",
      detail = paste(missing_seqs, collapse = ", ")
    ))
  }

  bad_seqs <- validity |>
    dplyr::filter(.data[[seq_valid_col]] == 0L) |>
    dplyr::pull(.data$SEQ)

  if (length(bad_seqs) > 0L) {
    return(list(
      ok = FALSE,
      reason = "invalid_sequence",
      detail = paste(bad_seqs, collapse = ", ")
    ))
  }

  return(list(ok = TRUE, reason = "ok", detail = ""))
}

# EMG eligibility: file present + VL (emg10) valid across the critical sequences.
check_emg_cohort_block <- function(run_id, analysis_family, analysis_block) {
  missing <- check_files_exist(run_id, c("preprocessed_emg.parquet"))
  if (length(missing) > 0L) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_file", paste(missing, collapse = ", ")))
  }

  validity_check <- check_validity_file(run_id, "preprocessed_emg.parquet", "sequence_valid_emg10")
  if (!isTRUE(validity_check$ok)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, validity_check$reason, validity_check$detail))
  }

  return(make_result(analysis_family, analysis_block, run_id, 1L, "ok", ""))
}

# BIA eligibility: required files + the target-frequency columns present, and sequence_valid.
check_bia_block <- function(run_id, analysis_family, analysis_block, block_detail) {
  required_files <- c(
    "master_index_grid.parquet",
    "participant.parquet",
    "preprocessed_torque.parquet",
    "preprocessed_bia4.parquet"
  )

  missing <- check_files_exist(run_id, required_files)
  if (length(missing) > 0L) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_file", paste(missing, collapse = ", ")))
  }

  required_bia_cols <- c(
    "time_index", "SEQ", "VC", "VC_count", "sequence_valid",
    unlist(lapply(bia_target_freqs_hz, function(freq_hz) {
      c(
        paste0("bia4_R_ohm__f_", freq_hz, "Hz"),
        paste0("bia4_Xc_ohm__f_", freq_hz, "Hz"),
        paste0("bia4_PhA_deg__f_", freq_hz, "Hz")
      )
    }))
  )

  bia_column_error <- check_required_columns(run_id, "preprocessed_bia4.parquet", required_bia_cols)
  if (!is.null(bia_column_error)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_column", bia_column_error))
  }

  master_column_error <- check_required_columns(run_id, "master_index_grid.parquet", c("time_index", "time_ref_s", "SEQ", "VC", "VC_count"))
  if (!is.null(master_column_error)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_column", master_column_error))
  }

  torque_column_error <- check_required_columns(run_id, "preprocessed_torque.parquet", c("time_index", "torque_raw", "SEQ", "VC", "VC_count", "sequence_valid"))
  if (!is.null(torque_column_error)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_column", torque_column_error))
  }

  participant_column_error <- check_required_columns(run_id, "participant.parquet", c("participant_id"))
  if (!is.null(participant_column_error)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_column", participant_column_error))
  }

  validity_check <- check_validity_file(run_id, "preprocessed_bia4.parquet", "sequence_valid")
  if (!isTRUE(validity_check$ok)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, validity_check$reason, validity_check$detail))
  }

  return(make_result(analysis_family, analysis_block, run_id, 1L, "ok", block_detail))
}

# NIRS eligibility: required files + the ΔHHb column present, and sequence_valid.
check_nirs_block <- function(run_id, analysis_family, analysis_block) {
  required_files <- c(
    "master_index_grid.parquet",
    "participant.parquet",
    "preprocessed_torque.parquet",
    "preprocessed_nirs.parquet"
  )

  missing <- check_files_exist(run_id, required_files)
  if (length(missing) > 0L) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_file", paste(missing, collapse = ", ")))
  }

  nirs_column_error <- check_required_columns(
    run_id, "preprocessed_nirs.parquet",
    c("time_index", "SEQ", "VC", "VC_count", "sequence_valid", "nirs_hhb_tx1")
  )
  if (!is.null(nirs_column_error)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, "missing_column", nirs_column_error))
  }

  validity_check <- check_validity_file(run_id, "preprocessed_nirs.parquet", "sequence_valid")
  if (!isTRUE(validity_check$ok)) {
    return(make_result(analysis_family, analysis_block, run_id, 0L, validity_check$reason, validity_check$detail))
  }

  return(make_result(analysis_family, analysis_block, run_id, 1L, "ok", ""))
}

# Not-yet-frozen blocks: always recorded as excluded with reason "block_not_frozen".
check_placeholder_block <- function(run_id, analysis_family, analysis_block, detail) {
  return(make_result(analysis_family, analysis_block, run_id, 0L, "block_not_frozen", detail))
}

# Build the full registry: every run x every block, dispatched to the matching checker
# (frozen blocks run their real check; everything else falls through to the placeholder).
registry_rows <- dplyr::bind_rows(
  lapply(run_ids, function(run_id) {
    dplyr::bind_rows(lapply(seq_len(nrow(analysis_block_registry)), function(i) {
      block_row <- analysis_block_registry[i, ]

      if (!isTRUE(block_row$block_frozen[[1]])) {
        return(check_placeholder_block(run_id, block_row$analysis_family[[1]], block_row$analysis_block[[1]], block_row$detail[[1]]))
      }

      if (identical(block_row$analysis_block[[1]], "emg_cohort")) {
        return(check_emg_cohort_block(run_id, block_row$analysis_family[[1]], block_row$analysis_block[[1]]))
      }

      if (block_row$analysis_family[[1]] == "bia") {
        return(check_bia_block(run_id, block_row$analysis_family[[1]], block_row$analysis_block[[1]], block_row$detail[[1]]))
      }

      if (identical(block_row$analysis_block[[1]], "nirs_individual")) {
        return(check_nirs_block(run_id, block_row$analysis_family[[1]], block_row$analysis_block[[1]]))
      }

      check_placeholder_block(run_id, block_row$analysis_family[[1]], block_row$analysis_block[[1]], block_row$detail[[1]])
    }))
  })
)

# ── cohort_final block ────────────────────────────────────────────────────────
# The canonical cohort N used by ALL four cohort notebooks and the multimodal notebook.
# Starts from the four-way intersection (emg_cohort + bia_cohort + nirs_individual + Myoton)
# then applies additional subject-level exclusions below.
#
# EX_STA quality exclusions (STA_EXCLUDE_RUN_IDS, from R/config/cohort_config.R):
#   284ChGe_20251030, 092MaLe_20251031
#   Reason: EX_STA < 25 valid windows after torque corridor [0.60, 0.70] filter.
#   Note: 011BeSa_20251023 was here but re-enters cohort_final (EX_STA parked for M2 memoir).
#
# Signal quality exclusions (COHORT_SIGNAL_EXCLUDE_RUN_IDS, from R/config/cohort_config.R):
#   261PoLe_20251128 — NIRS session-wide probe drift, trajectory unreliable.
#   Excluded from ALL notebooks to maintain a consistent N.
#
# NIRS direction exclusions (NIRS_DIRECTION_EXCLUDE_RUN_IDS, from R/config/cohort_config.R):
#   622PiSt_20251104, 222SoMa_20251113 — EX_DYN ΔHHb inverted (net deoxygenation absent).
#   Mixing inverted-direction runs into cohort mean causes cancellation artefacts.
source(here::here("sources", "R", "config", "cohort_config.R"))  # provides STA_EXCLUDE_RUN_IDS, COHORT_SIGNAL_EXCLUDE_RUN_IDS, NIRS_DIRECTION_EXCLUDE_RUN_IDS

# Runs valid in all three gated blocks (emg_cohort + bia_cohort + nirs_individual): keep a run
# only if it appears as included in exactly 3 of those blocks.
four_way_ids <- registry_rows |>
  dplyr::filter(
    .data$analysis_block %in% c("emg_cohort", "bia_cohort", "nirs_individual"),
    .data$included == 1L
  ) |>
  dplyr::group_by(.data$RUN_ID) |>
  dplyr::filter(dplyr::n() == 3L) |>
  dplyr::ungroup() |>
  dplyr::distinct(.data$RUN_ID) |>
  dplyr::pull(.data$RUN_ID)

# Myoton has no validity block -> include by file scan, then intersect to get the true four-way set.
myoton_ids <- basename(dirname(Sys.glob(
  file.path(export_root, "*", "myoton_checkpoint_feature_table.parquet")
)))
four_way_ids <- intersect(four_way_ids, myoton_ids)

cohort_final_rows <- dplyr::bind_rows(lapply(four_way_ids, function(rid) {
  if (rid %in% STA_EXCLUDE_RUN_IDS) {
    make_result("cohort", "cohort_final", rid, 0L, "ex_sta_quality",
                paste0("EX_STA < ", STA_MIN_VALID_WINDOWS, " valid windows after torque corridor filter"))
  } else if (rid %in% COHORT_SIGNAL_EXCLUDE_RUN_IDS) {
    make_result("cohort", "cohort_final", rid, 0L, "signal_quality",
                "NIRS session-wide probe drift: trajectory unreliable. See R/config/cohort_config.R.")
  } else if (rid %in% NIRS_DIRECTION_EXCLUDE_RUN_IDS) {
    make_result("cohort", "cohort_final", rid, 0L, "nirs_direction",
                "EX_DYN HHb direction inverted: mixing with positive-direction cohort causes cancellation. See R/config/cohort_config.R.")
  } else {
    make_result("cohort", "cohort_final", rid, 1L, "ok", "")
  }
}))

registry_rows <- dplyr::bind_rows(registry_rows, cohort_final_rows)

registry_path_parquet <- file.path(subset_root, "multimodal_subset_registry.parquet")
registry_path_csv <- file.path(subset_root, "multimodal_subset_registry.csv")

arrow::write_parquet(registry_rows, sink = registry_path_parquet)
write.csv(registry_rows, file = registry_path_csv, row.names = FALSE)

summary_table <- registry_rows |>
  dplyr::count(.data$analysis_block, .data$included, .data$reason) |>
  dplyr::arrange(.data$analysis_block, dplyr::desc(.data$included), .data$reason)

print(summary_table)
message("\n[SUBSET REGISTRY] Parquet -> ", normalizePath(registry_path_parquet, winslash = "/", mustWork = FALSE))
message("[SUBSET REGISTRY] CSV -> ", normalizePath(registry_path_csv, winslash = "/", mustWork = FALSE))
