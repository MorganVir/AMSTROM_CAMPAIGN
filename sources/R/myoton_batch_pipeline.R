# myoton_batch_pipeline.R
# Render myoton_individual_pipeline.Rmd for all available runs.
# Same pattern as bia_batch_pipeline.R / nirs_batch_pipeline.R.

library(rmarkdown)

# --- Controls ---
SKIP_RUN_IDS    <- ""          # comma-separated RUN_IDs to skip, or ""
FORCE_ALL       <- FALSE       # TRUE = re-render even if output already exists

# --- Discover runs ---
data_export_root <- here::here("results", "DATA_EXPORT")
all_run_dirs     <- list.dirs(data_export_root, full.names = FALSE, recursive = FALSE)

# Keep only runs that actually have a Myoton export (file-scan, not registry-driven).
has_myoton <- sapply(all_run_dirs, function(rid) {
  file.exists(file.path(data_export_root, rid, "preprocessed_myoton.parquet"))
})
candidate_run_ids <- all_run_dirs[has_myoton]

skip_ids <- trimws(strsplit(SKIP_RUN_IDS, ",")[[1]])
skip_ids <- skip_ids[nchar(skip_ids) > 0]
run_ids  <- setdiff(candidate_run_ids, skip_ids)

cat(sprintf("[myoton_batch] %d runs with Myoton data; %d skipped; rendering %d.\n",
            length(candidate_run_ids), length(skip_ids), length(run_ids)))

# --- Render loop ---
results <- data.frame(
  run_id  = run_ids,
  status  = NA_character_,
  message = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_along(run_ids)) {
  rid <- run_ids[i]
  out_path <- file.path(data_export_root, rid, paste0(rid, "_myoton_report.html"))

  if (!FORCE_ALL && file.exists(out_path)) {
    cat(sprintf("[%d/%d] %s — skipped (report exists)\n", i, length(run_ids), rid))
    results$status[i]  <- "skipped"
    results$message[i] <- "report already exists"
    next
  }

  cat(sprintf("[%d/%d] %s — rendering ...\n", i, length(run_ids), rid))

  tryCatch({
    rmarkdown::render(
      input       = here::here("notebooks", "myoton_individual_pipeline.Rmd"),
      output_file = out_path,
      params      = list(RUN_ID = rid),
      quiet       = TRUE,
      envir       = new.env(parent = globalenv())
    )
    results$status[i]  <- "ok"
    results$message[i] <- ""
    cat(sprintf("  -> ok\n"))
  }, error = function(e) {
    results$status[i]  <<- "error"
    results$message[i] <<- conditionMessage(e)
    cat(sprintf("  -> ERROR: %s\n", conditionMessage(e)))
  })
}

# --- Summary ---
cat("\n=== Myoton batch summary ===\n")
table_out <- table(results$status)
print(table_out)

errors <- results[!is.na(results$status) & results$status == "error", ]
if (nrow(errors) > 0) {
  cat("\nFailed runs:\n")
  for (j in seq_len(nrow(errors))) {
    cat(sprintf("  %s: %s\n", errors$run_id[j], errors$message[j]))
  }
}
