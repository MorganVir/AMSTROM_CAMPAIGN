# torque_stats.R - EX_DYN torque decline statistics (cohort, N=16)
# Computes the knee-extension torque values and paired Wilcoxon signed-rank tests reported
# in the article "Torque during EX_DYN" section, with matched-pairs rank-biserial effect sizes:
#   (1) first contraction (rep 1) vs the reference MVIC (torque_norm = 1.0 by construction)
#   (2) sixtieth contraction (rep 60) vs the first contraction (rep 1)
# Reads the per-run EX_DYN feature tables for the cohort_final runs (N=16).
# Run: Rscript --vanilla torque_stats.R

suppressPackageStartupMessages({ library(arrow); library(dplyr); library(purrr) })
source(here::here("sources", "R", "config", "cohort_config.R"))
reg <- read_parquet(here::here("results/R_ANALYSIS/subset_registry/multimodal_subset_registry.parquet"))
run_ids <- reg |> filter(analysis_block == "cohort_final", included == 1L) |> pull(RUN_ID) |> unique()  # the N=16 cohort
cat("N =", length(run_ids), "\n")

pool <- function(ids, root, fname) {
  return(map_dfr(ids, function(x) {
    p <- file.path(root, x, fname)
    if (file.exists(p)) read_parquet(p) |> mutate(RUN_ID = x) else NULL
  }))
}

emg <- pool(run_ids, here::here("results/R_ANALYSIS"), "ex_dyn_mid1s_feature_table_norm.parquet") |>
  filter(channel == "emg10")  # VL only

rep1  <- emg |> filter(VC_count == 1)  |> group_by(RUN_ID) |> summarise(t = mean(torque_norm, na.rm=T), .groups="drop")  # per-run mean nTorque, first contraction
rep60 <- emg |> filter(VC_count == 60) |> group_by(RUN_ID) |> summarise(t = mean(torque_norm, na.rm=T), .groups="drop")  # per-run mean nTorque, 60th contraction

# MVC_REF = 1.0 by construction of torque_norm
mvc_ref_vals <- rep(1.0, nrow(rep1))

# Matched-pairs rank-biserial correlation (Kerby 2014): r_rb = 2*V/T - 1, where V is the
# sum of positive signed ranks and T = n(n+1)/2 over the nonzero pairs (matches wilcox.test's V).
rank_biserial_paired <- function(x, y) {
  d <- (x - y); d <- d[d != 0]; n <- length(d)
  return(2 * sum(rank(abs(d))[d > 0]) / (n * (n + 1) / 2) - 1)
}

cat("\nWilcoxon MVC_REF (1.0) vs rep1:\n")
w1 <- wilcox.test(mvc_ref_vals, rep1$t, paired=TRUE, exact=FALSE)
cat(sprintf("  V = %g, p = %.4f, rank-biserial r_rb = %.3f\n",
            w1$statistic, w1$p.value, rank_biserial_paired(mvc_ref_vals, rep1$t)))

cat("\nWilcoxon rep1 vs rep60:\n")
d <- inner_join(rep1, rep60, by="RUN_ID", suffix=c("_r1","_r60"))  # pair each run's rep1 and rep60
w2 <- wilcox.test(d$t_r1, d$t_r60, paired=TRUE, exact=FALSE)
cat(sprintf("  V = %g, p = %.6f, rank-biserial r_rb = %.3f\n",
            w2$statistic, w2$p.value, rank_biserial_paired(d$t_r1, d$t_r60)))

cat(sprintf("\ntorque_norm rep1:  %.3f +/- %.3f\n", mean(rep1$t), sd(rep1$t)))
cat(sprintf("torque_norm rep60: %.3f +/- %.3f\n", mean(rep60$t), sd(rep60$t)))
cat(sprintf("Drop rep1->rep60: %.1f%%\n", (mean(rep1$t) - mean(rep60$t)) / mean(rep1$t) * 100))
