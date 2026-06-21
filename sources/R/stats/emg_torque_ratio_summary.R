# emg_torque_ratio_summary.R
# Cohort summary of the VL nRMS/torque ratio (neuromuscular efficiency index) over EX_DYN.
# The ratio = rms_norm / torque_norm per contraction (column emg_torque_ratio in the
# per-run ex_dyn_emg_torque_ratio_table.parquet exports). Reported in the memoir as a
# sEMG-in-context fatigue marker: nRMS stays flat while torque collapses, so the ratio rises.
# N=16 cohort_final run list (matches ml_loso_metrics / cohort notebooks). VL = channel emg10.

suppressMessages({library(arrow); library(dplyr); library(purrr)})

runs <- c('011BeSa_20251023','071NaHy_20251031','101SaCl_20251127','151FuCy_20251105',
          '209CeJa_20251105','314DuHu_20251009','374SaSa_20251030','394ZoMo_20251114',
          '444BoAl_20251104','487LePa_20251003','500TeJo_20251114','592RiNa_20251028',
          '743MoGi_20251103','802BlHu_20251028','890PeTh_20251023','923AoMo_20251017')
base <- 'results/R_ANALYSIS'

dat <- map_dfr(runs, ~read_parquet(
  file.path(base, .x, 'ex_dyn_emg_torque_ratio_table.parquet'))) |>
  filter(channel == 'emg10')                       # VL only

# per-run endpoints, OLS slope vs repetition, and within-run Pearson r
ep <- dat |> group_by(RUN_ID) |>
  summarise(r1    = emg_torque_ratio[which.min(VC_count)],    # ratio at the first rep
            r60   = emg_torque_ratio[which.max(VC_count)],    # ratio at the last rep
            slope = coef(lm(emg_torque_ratio ~ VC_count))[2], # OLS rise per rep
            pear  = cor(emg_torque_ratio, VC_count),          # within-run Pearson r vs rep
            .groups = 'drop')

cat(sprintf('VL EMG/torque ratio, N=%d\n', nrow(ep)))
cat(sprintf('  rep1   = %.2f +/- %.2f\n', mean(ep$r1),  sd(ep$r1)))
cat(sprintf('  rep60  = %.2f +/- %.2f\n', mean(ep$r60), sd(ep$r60)))
cat(sprintf('  change = %+.0f%%\n', 100 * (mean(ep$r60) - mean(ep$r1)) / mean(ep$r1)))
cat(sprintf('  per-run slope/rep = %.4f +/- %.4f (rising %d/%d)\n',
            mean(ep$slope), sd(ep$slope), sum(ep$slope > 0), nrow(ep)))
cat(sprintf('  per-run Pearson r vs rep = %.2f +/- %.2f (positive %d/%d)\n',
            mean(ep$pear), sd(ep$pear), sum(ep$pear > 0), nrow(ep)))
