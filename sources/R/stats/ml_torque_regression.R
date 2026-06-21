# ml_torque_regression.R
# LOSO torque regression from multimodal features.
# 4 feature sets (EMG, NIRS, BIA, Multimodal) x 4 models x 2 tasks = 32 configs.
#
# Feature engineering rationale (González-Izal 2012):
# rate-of-change features predict fatigue better than static values.
# Three feature groups per modality: raw, delta-from-onset (d_), running slope (s_).

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(rpart)
  library(ranger)
})

set.seed(42L)
cat("Started:", format(Sys.time(), "%H:%M:%S"), "\n\n")

DATA_PATH <- here::here("results/R_ANALYSIS/multimodal_ml_feature_table.parquet")
OUT_DIR   <- here::here("results/R_ANALYSIS/ml_results")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load ────────────────────────────────────────────────────────────────────

tbl <- read_parquet(DATA_PATH)
cat("Loaded:", nrow(tbl), "rows,", n_distinct(tbl$RUN_ID), "runs\n\n")

# Discover BIA columns dynamically — handles 3-freq (old) and 16-freq (BIS) parquets
bia_raw <- grep("^bia_[a-z]+_[0-9]+k$", names(tbl), value = TRUE)

# Exclude Xc and PhA below 24.4 kHz — inductive parasitic artifact confirmed by
# MVC_REF CV check (N=16): CV 3.7-13% at 4-19 kHz vs <3.5% at >=24 kHz.
# R and absZ are clean at all frequencies (CV <1.5%) and kept.
bad_bia <- grep("^bia_(xc|pha)_(4|9|14|19)k$", bia_raw, value = TRUE)
bia_raw <- setdiff(bia_raw, bad_bia)
cat("BIA artifact columns excluded (Xc/PhA < 24.4 kHz):", length(bad_bia), "\n")
cat(paste(bad_bia, collapse = ", "), "\n\n")

# ── Inter-frequency delta features ─────────────────────────────────────────────
# delta = value(high_freq) - value(low_freq) at each rep.
# Captures Cole arc spread shift as fatigue progresses.
# R/absZ anchors: {4k,9k,48k,97k,122k} — all 16 freqs clean at MVC (CV <1.5%).
# Xc/PhA anchors: {24k,48k,97k,122k}   — clean at >=24k (CV <3.5% at MVC_REF).

ifd_pairs_r_absz <- list(
  c("4k","9k"), c("4k","48k"), c("4k","97k"), c("4k","122k"),
  c("9k","48k"), c("9k","97k"), c("9k","122k"),
  c("48k","97k"), c("48k","122k"),
  c("97k","122k")
)
ifd_pairs_xc_pha <- list(
  c("24k","48k"), c("24k","97k"), c("24k","122k"),
  c("48k","97k"), c("48k","122k"),
  c("97k","122k")
)

ifd_cols <- character(0)
for (p in ifd_pairs_r_absz) {
  for (var in c("r", "absz")) {
    lo <- paste0("bia_", var, "_", p[1]); hi <- paste0("bia_", var, "_", p[2])
    nm <- paste0("ifd_", var, "_", p[1], "_", p[2])
    tbl[[nm]] <- tbl[[hi]] - tbl[[lo]]
    ifd_cols  <- c(ifd_cols, nm)
  }
}
for (p in ifd_pairs_xc_pha) {
  for (var in c("xc", "pha")) {
    lo <- paste0("bia_", var, "_", p[1]); hi <- paste0("bia_", var, "_", p[2])
    nm <- paste0("ifd_", var, "_", p[1], "_", p[2])
    tbl[[nm]] <- tbl[[hi]] - tbl[[lo]]
    ifd_cols  <- c(ifd_cols, nm)
  }
}
cat("Inter-freq delta columns added:", length(ifd_cols), "\n\n")

RAW_COLS <- c("nRMS", "nMDF", "hhb_from_seq_rest_umol", bia_raw, ifd_cols)

# ── 2. Feature engineering ─────────────────────────────────────────────────────

# Vectorised OLS slope of y[1..k] ~ seq 1..k, no lm() calls.
# Returns 0 for k <= 2 (insufficient points for meaningful slope).
running_slope <- function(y) {
  n      <- length(y)
  k      <- seq_len(n)
  cum_y  <- cumsum(y)
  cum_iy <- cumsum(k * y)
  cum_i  <- k * (k + 1L) / 2L
  cum_i2 <- k * (k + 1L) * (2L * k + 1L) / 6L
  denom  <- k * cum_i2 - cum_i^2
  out    <- ifelse(denom > 0, (k * cum_iy - cum_i * cum_y) / denom, 0)
  out[seq_len(min(2L, n))] <- 0
  return(out)
}

feat_data <- tbl |>
  group_by(RUN_ID, task) |>
  arrange(time_index, .by_group = TRUE) |>
  mutate(
    across(all_of(RAW_COLS), ~ .x - first(.x),   .names = "d_{.col}"),   # d_ = delta from sequence onset
    across(all_of(RAW_COLS), ~ running_slope(.x), .names = "s_{.col}")   # s_ = running OLS slope up to this rep
  ) |>
  ungroup()

# ── 3. Feature set definitions ─────────────────────────────────────────────────

bia_raw   <- grep("^bia_", RAW_COLS, value = TRUE)
bia_delta <- paste0("d_", bia_raw)
bia_slope <- paste0("s_", bia_raw)
ifd_raw   <- grep("^ifd_", RAW_COLS, value = TRUE)
ifd_delta <- paste0("d_", ifd_raw)
ifd_slope <- paste0("s_", ifd_raw)

FEAT_SETS <- list(
  EMG = c(
    "nRMS", "nMDF",
    "d_nRMS", "d_nMDF",
    "s_nRMS", "s_nMDF"
  ),
  NIRS = c(
    "hhb_from_seq_rest_umol",
    "d_hhb_from_seq_rest_umol",
    "s_hhb_from_seq_rest_umol"
  ),
  BIA = c(bia_raw, bia_delta, bia_slope, ifd_raw, ifd_delta, ifd_slope),
  Multi = c(
    "nRMS", "nMDF",
    "d_nRMS", "d_nMDF",
    "s_nRMS", "s_nMDF",
    "hhb_from_seq_rest_umol",
    "d_hhb_from_seq_rest_umol",
    "s_hhb_from_seq_rest_umol",
    bia_raw, bia_delta, bia_slope,
    ifd_raw, ifd_delta, ifd_slope
  )
)

MODELS <- c("dummy", "lm", "tree", "rf")
TASKS  <- c("EX_DYN")

cat("Feature set sizes:\n")
for (nm in names(FEAT_SETS)) cat(" ", nm, ":", length(FEAT_SETS[[nm]]), "features\n")
cat("\n")

# ── 4. LOSO ────────────────────────────────────────────────────────────────────

run_loso <- function(data, feat_cols, model_type, task_filter) {
  d    <- filter(data, task == task_filter)
  runs <- sort(unique(d$RUN_ID))

  results <- lapply(runs, function(held_out) {
    tr <- filter(d, RUN_ID != held_out)
    te <- filter(d, RUN_ID == held_out)

    y_tr <- tr$torque_norm
    d_tr <- cbind(y = y_tr, select(tr, all_of(feat_cols)))
    d_te <- select(te, all_of(feat_cols))

    if (model_type == "dummy") {
      preds <- rep(mean(y_tr), nrow(te))
      imp   <- NULL

    } else if (model_type == "lm") {
      fit   <- lm(y ~ ., data = d_tr)
      preds <- unname(predict(fit, newdata = d_te))
      imp   <- NULL

    } else if (model_type == "tree") {
      fit   <- rpart(y ~ ., data = d_tr,
                     control = rpart.control(maxdepth = 4, minsplit = 5, cp = 1e-3))
      preds <- unname(predict(fit, newdata = d_te))
      imp   <- NULL

    } else {
      fit   <- ranger(y ~ ., data = d_tr, num.trees = 500, seed = 42L,
                      importance = "permutation")
      preds <- predict(fit, data = d_te)$predictions
      imp   <- fit$variable.importance
    }

    list(
      preds = tibble(
        RUN_ID     = held_out,
        task       = task_filter,
        actual     = te$torque_norm,
        pred       = preds,
        time_index = te$time_index
      ),
      imp = imp
    )
  })

  preds_df <- bind_rows(lapply(results, `[[`, "preds"))

  imp_df <- if (model_type == "rf") {
    imp_mat <- do.call(rbind, lapply(results, `[[`, "imp"))
    tibble(
      feature  = colnames(imp_mat),
      mean_imp = colMeans(imp_mat),
      sd_imp   = apply(imp_mat, 2, sd)
    )
  } else NULL

  return(list(preds = preds_df, importance = imp_df))
}

# ── 5. Run all configurations ──────────────────────────────────────────────────

all_metrics    <- list()
all_preds      <- list()
all_importance <- list()

total <- length(TASKS) * length(MODELS) * length(FEAT_SETS)
i_run <- 0L

for (task_name in TASKS) {
  for (model_name in MODELS) {
    for (feat_name in names(FEAT_SETS)) {
      i_run <- i_run + 1L
      cat(sprintf("[%2d/%d] %-6s | %-5s | %s\n",
                  i_run, total, feat_name, task_name, model_name))

      res <- run_loso(feat_data, FEAT_SETS[[feat_name]], model_name, task_name)

      fold_metrics <- res$preds |>
        group_by(RUN_ID) |>
        summarise(
          rmse = sqrt(mean((actual - pred)^2)),
          mae  = mean(abs(actual - pred)),
          r    = if (sd(pred) > 0) cor(actual, pred) else 0,
          .groups = "drop"
        ) |>
        mutate(task = task_name, feat_set = feat_name, model = model_name)

      all_metrics <- c(all_metrics, list(fold_metrics))
      all_preds   <- c(all_preds,
                       list(mutate(res$preds, feat_set = feat_name, model = model_name)))

      if (!is.null(res$importance)) {
        all_importance <- c(all_importance,
                            list(mutate(res$importance,
                                        feat_set = feat_name, task = task_name)))
      }
    }
  }
}

# ── 6. Export ──────────────────────────────────────────────────────────────────

metrics_tbl    <- bind_rows(all_metrics)
preds_tbl      <- bind_rows(all_preds)
importance_tbl <- bind_rows(all_importance)

write_parquet(metrics_tbl,    file.path(OUT_DIR, "ml_loso_metrics.parquet"))
write_parquet(preds_tbl,      file.path(OUT_DIR, "ml_predictions.parquet"))
write_parquet(importance_tbl, file.path(OUT_DIR, "ml_rf_importance.parquet"))

cat("\nExported to", OUT_DIR, "\n")

# ── 7. Summary ─────────────────────────────────────────────────────────────────

model_order <- c("dummy", "lm", "tree", "rf")

summary_tbl <- metrics_tbl |>
  group_by(task, feat_set, model) |>
  summarise(
    RMSE = sprintf("%.4f +/- %.4f", mean(rmse), sd(rmse)),
    MAE  = sprintf("%.4f +/- %.4f", mean(mae),  sd(mae)),
    r    = sprintf("%.3f +/- %.3f", mean(r),    sd(r)),
    .groups = "drop"
  ) |>
  arrange(task, feat_set, match(model, model_order))

cat(sprintf("\n══ LOSO Summary (mean +/- SD, %d folds) ══\n\n", n_distinct(metrics_tbl$RUN_ID)))
print(summary_tbl, n = Inf)

cat("\n══ Top 10 RF features — Multimodal EX_DYN ══\n")
importance_tbl |>
  filter(feat_set == "Multi", task == "EX_DYN") |>
  arrange(desc(mean_imp)) |>
  slice_head(n = 10) |>
  print()

cat("\n══ Top 10 RF features — Multimodal EX_STA ══\n")
importance_tbl |>
  filter(feat_set == "Multi", task == "EX_STA") |>
  arrange(desc(mean_imp)) |>
  slice_head(n = 10) |>
  print()

# ── 8. Wilcoxon: Multi RF vs EMG RF (EX_DYN, paired LOSO folds) ────────────────

cat("\n══ Test: Multi RF vs EMG RF — EX_DYN LOSO r (Wilcoxon signed-rank, paired) ══\n")

multi_r_vec <- metrics_tbl |>
  dplyr::filter(task == "EX_DYN", feat_set == "Multi", model == "rf") |>
  dplyr::arrange(RUN_ID) |>
  dplyr::pull(r)

emg_r_vec <- metrics_tbl |>
  dplyr::filter(task == "EX_DYN", feat_set == "EMG", model == "rf") |>
  dplyr::arrange(RUN_ID) |>
  dplyr::pull(r)

w_test     <- wilcox.test(multi_r_vec, emg_r_vec, paired = TRUE, alternative = "greater")
w_twosided <- wilcox.test(multi_r_vec, emg_r_vec, paired = TRUE, alternative = "two.sided")
r_eff      <- abs(qnorm(w_twosided$p.value / 2)) / sqrt(length(multi_r_vec))

# Matched-pairs rank-biserial correlation (Kerby 2014): r_rb = 2*V/T - 1, where V is the sum of
# positive signed ranks and T = n(n+1)/2 over the nonzero pairs. This is the effect size reported
# in the article (more standard for the Wilcoxon signed-rank test than the z-based r above).
diff_me <- multi_r_vec - emg_r_vec
diff_me <- diff_me[diff_me != 0]
n_pairs <- length(diff_me)
r_rb    <- 2 * sum(rank(abs(diff_me))[diff_me > 0]) / (n_pairs * (n_pairs + 1) / 2) - 1

cat(sprintf("  Multi RF mean r : %.3f (SD %.3f)\n", mean(multi_r_vec), sd(multi_r_vec)))
cat(sprintf("  EMG   RF mean r : %.3f (SD %.3f)\n", mean(emg_r_vec),   sd(emg_r_vec)))
cat(sprintf("  V = %g,  p (one-sided Multi > EMG) = %s\n",
            w_test$statistic, format.pval(w_test$p.value, digits = 3, eps = 0.001)))
cat(sprintf("  Effect size: rank-biserial r_rb = %.3f (z-based r = %.3f)\n", r_rb, r_eff))

cat("\nDone:", format(Sys.time(), "%H:%M:%S"), "\n")
