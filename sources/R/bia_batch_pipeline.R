# Batch runner for bia_individual_pipeline.Rmd: render every BIA-eligible run one at a time,
# logging success/error per run. Eligibility comes from the subset registry (no per-file
# pre-check like EMG -- BIA validity is already encoded in the registry). Run from project root.
library(arrow)
library(dplyr)
library(rmarkdown)

subset_registry_path <- here::here("results", "R_ANALYSIS", "subset_registry", "multimodal_subset_registry.parquet"
)
report_dir <- here::here("results", "QC_EXPORT", "bia_individual_analysis")
analysis_block <- "bia_individual"  # registry block that decides which runs are eligible

# Runs to leave out of this batch (already processed / known bad).
SKIP_RUN_IDS <- c(
  ""
)

if (!file.exists(subset_registry_path)) {
  stop(
    "Subset registry not found: ",
    normalizePath(subset_registry_path, winslash = "/", mustWork = FALSE),
    "\nRun multimodal_subset_checker.R first."
  )
}

subset_registry <- arrow::read_parquet(subset_registry_path)

# Eligible = included in this block and not in the skip list.
selected_runs <- subset_registry |>
  dplyr::filter(.data$analysis_block == .env$analysis_block) |>
  dplyr::filter(.data$included == 1L) |>
  dplyr::filter(!.data$RUN_ID %in% .env$SKIP_RUN_IDS) |>
  dplyr::arrange(.data$RUN_ID)

# The complement, printed alongside so you can see why a run was dropped.
excluded_runs <- subset_registry |>
  dplyr::filter(.data$analysis_block == .env$analysis_block) |>
  dplyr::filter(.data$included != 1L) |>
  dplyr::arrange(.data$RUN_ID)

cat("Eligible runs for ", analysis_block, ":\n", sep = "")
print(selected_runs, n = Inf)
cat("\nExcluded runs for ", analysis_block, ":\n", sep = "")
print(excluded_runs, n = Inf)
cat("\n")

valid_run_ids <- selected_runs$RUN_ID

if (length(valid_run_ids) == 0L) {
  stop("No runs cleared for ", analysis_block, ". Check the subset registry.")
}

dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)

render_log <- character(length(valid_run_ids))
names(render_log) <- valid_run_ids

for (run_id in valid_run_ids) {
  cat(sprintf("[%d/%d] %s ... ", which(valid_run_ids == run_id), length(valid_run_ids), run_id))

  render_log[[run_id]] <- tryCatch({
    rmarkdown::render(
      input       = here::here("notebooks", "bia_individual_pipeline.Rmd"),
      params      = list(RUN_ID = run_id),
      output_dir  = report_dir,
      output_file = paste0(run_id, "_bia_report.html"),
      quiet       = TRUE
    )
    "success"
  }, error = function(e) paste0("error: ", conditionMessage(e)))

  cat(render_log[[run_id]], "\n")
}

cat("\n=== BIA batch render summary ===\n")
summary_df <- tibble::tibble(
  RUN_ID = names(render_log),
  result = unname(render_log)
)
print(summary_df, n = Inf)
