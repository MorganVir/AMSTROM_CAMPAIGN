# Batch runner for emg_individual_pipeline.Rmd
#
# Phase 1: pre-check every run for EMG channel validity and build a confirmed
#           list before touching anything.
# Phase 2: render only the valid runs, one at a time.
#
# Run from the project root (same directory as emg_individual_pipeline.Rmd).

library(arrow)
library(dplyr)
library(rmarkdown)

export_root <- here::here("results", "DATA_EXPORT")
report_dir  <- here::here("results", "QC_EXPORT", "emg_individual_analysis")

# Runs already processed — exclude from this batch.
SKIP_RUN_IDS <- c(
  ""
)

# ── Phase 1: pre-check all runs ───────────────────────────────────────────────

all_run_ids <- setdiff(
  list.dirs(export_root, recursive = FALSE, full.names = FALSE),
  SKIP_RUN_IDS
)

if (length(all_run_ids) == 0L) {
  stop("No run folders found under: ", normalizePath(export_root, mustWork = FALSE))
}

cat(sprintf("Found %d run folder(s). Running pre-flight checks...\n\n",
            length(all_run_ids)))

check_run <- function(run_id) {
  emg_path <- file.path(export_root, run_id, "preprocessed_emg.parquet")

  if (!file.exists(emg_path)) {
    return(data.frame(run_id  = run_id,
                      status  = "missing_emg_file",
                      detail  = NA_character_,
                      stringsAsFactors = FALSE))
  }

  validity <- tryCatch(
    arrow::read_parquet(
      emg_path,
      col_select = c("SEQ",
                     "sequence_valid_emg6",
                     "sequence_valid_emg8",
                     "sequence_valid_emg10")
    ) |>
      dplyr::distinct(SEQ,
                      sequence_valid_emg6,
                      sequence_valid_emg8,
                      sequence_valid_emg10),
    error = function(e) NULL
  )

  if (is.null(validity)) {
    return(data.frame(run_id  = run_id,
                      status  = "read_error",
                      detail  = NA_character_,
                      stringsAsFactors = FALSE))
  }

  critical_seqs <- c("MVC_REF", "SVC_REF", "EX_DYN", "EX_STA", "MVC_RECOV_DYN", "MVC_RECOV_STA")
  validity_crit <- validity |> dplyr::filter(SEQ %in% critical_seqs)

  bad_emg6  <- any(validity_crit$sequence_valid_emg6  == 0L, na.rm = TRUE)
  bad_emg8  <- any(validity_crit$sequence_valid_emg8  == 0L, na.rm = TRUE)
  bad_emg10 <- any(validity_crit$sequence_valid_emg10 == 0L, na.rm = TRUE)  # VL bad -> whole run skipped

  if (bad_emg10) {
    bad_ch <- c(
      if (bad_emg6) "emg6 (RF)",
      if (bad_emg8) "emg8 (VM)",
      "emg10 (VL)"
    )
    return(data.frame(run_id  = run_id,
                      status  = "invalid_vl",
                      detail  = paste(bad_ch, collapse = ", "),
                      stringsAsFactors = FALSE))
  }

  if (bad_emg6 || bad_emg8) {
    bad_ch <- c(
      if (bad_emg6) "emg6 (RF)",
      if (bad_emg8) "emg8 (VM)"
    )
    return(data.frame(run_id  = run_id,
                      status  = "ok_partial",
                      detail  = paste(bad_ch, collapse = ", "),
                      stringsAsFactors = FALSE))
  }

  return(data.frame(run_id  = run_id,
             status  = "ok",
             detail  = NA_character_,
             stringsAsFactors = FALSE))
}

preflight <- dplyr::bind_rows(lapply(all_run_ids, check_run))

cat("Pre-flight results:\n")
print(preflight, row.names = FALSE, na.print = "")
cat("\n")

valid_run_ids <- preflight$run_id[preflight$status %in% c("ok", "ok_partial")]
n_skip        <- nrow(preflight) - length(valid_run_ids)

cat(sprintf("%d / %d run(s) cleared for rendering. %d skipped.\n\n",
            length(valid_run_ids), nrow(preflight), n_skip))

# ── Phase 2: batch render ─────────────────────────────────────────────────────

if (length(valid_run_ids) == 0L) {
  message("Nothing to render.")
} else {

  dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

  render_log        <- character(length(valid_run_ids))
  names(render_log) <- valid_run_ids

  for (run_id in valid_run_ids) {
    cat(sprintf("[%d/%d] %s ... ",
                which(valid_run_ids == run_id),
                length(valid_run_ids),
                run_id))

    render_log[[run_id]] <- tryCatch({
      rmarkdown::render(
        input       = here::here("notebooks", "emg_individual_pipeline.Rmd"),
        params      = list(RUN_ID = run_id),
        output_dir  = report_dir,
        output_file = paste0(run_id, "_report.html"),
        quiet       = TRUE
      )
      "success"
    }, error = function(e) paste0("error: ", conditionMessage(e)))

    cat(render_log[[run_id]], "\n")
  }

  # ── Summary ─────────────────────────────────────────────────────────────────

  cat("\n=== Batch render summary ===\n")

  skipped_df <- preflight |>
    dplyr::filter(!status %in% c("ok", "ok_partial")) |>
    dplyr::transmute(run_id, result = status, detail)

  rendered_df <- data.frame(
    run_id = names(render_log),
    result = unname(render_log),
    detail = NA_character_,
    stringsAsFactors = FALSE
  )

  summary_df <- dplyr::bind_rows(skipped_df, rendered_df) |>
    dplyr::arrange(run_id)

  print(summary_df, row.names = FALSE, na.print = "")

  n_ok  <- sum(render_log == "success")
  n_err <- sum(render_log != "success")
  cat(sprintf("\nRendered: %d ok  |  %d error(s)  |  %d skipped (pre-flight)\n",
              n_ok, n_err, n_skip))
}
