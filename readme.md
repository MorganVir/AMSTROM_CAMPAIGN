# AMSTROM Campaign

Multimodal preprocessing and analysis pipeline for muscular fatigue data collected during a quadriceps fatigue protocol. It synchronizes five signals recorded by independent acquisition systems — surface EMG (sEMG), knee-extension torque, near-infrared spectroscopy (NIRS), electrical impedance myography / bio-impedance (EIM), and Myoton mechanical properties — onto a shared time axis, then extracts fatigue-related features at the individual and cohort level for each modality.

The pipeline has two halves:

1. **Python preprocessing** (`main.ipynb` → `sources/_01_…_09_`): import, synchronize, segment, and export each participant's run to Parquet.
2. **R analysis** (`main.Rmd` → `notebooks/`, `sources/R/`): feature extraction, per-modality cohort analysis, a combined multimodal analysis, and the machine-learning torque regression.

The Parquet files in `results/DATA_EXPORT/` are the contract between the two halves.

Built on top of the [Python minimal environment for reproducible research](https://github.com/DenisMot/Python-minimal-install).

---

## Repository structure

```
AMSTROM_CAMPAIN/
├── readme.md
├── LICENSE                         GPLv3
├── main.Rproj                      RStudio project file
├── main.ipynb                      entry point — Python preprocessing (one participant at a time)
├── main.Rmd                        entry point — R analysis orchestrator
├── data/                           participant info + (local) raw signals — no data shipped
├── results/                        analysis outputs — regenerated, not shipped
├── notebooks/                      per-modality + multimodal analyses (Rmd)
│   ├── emg_individual_pipeline.Rmd      emg_cohort_analysis.Rmd
│   ├── bia_individual_pipeline.Rmd      bia_cohort_analysis.Rmd
│   ├── nirs_individual_pipeline.Rmd     nirs_cohort_analysis.Rmd
│   ├── myoton_individual_pipeline.Rmd   myoton_cohort_analysis.Rmd
│   └── multimodal_cohort_analysis.Rmd
└── sources/
    ├── _01_…_09_*.py               Python preprocessing steps
    ├── utils/                      Python helper scripts
    └── R/                          R support code
        ├── config/                shared plotting style + cohort configuration
        ├── stats/                 scripts computing each reported result
        └── *_batch_pipeline.R  multimodal_subset_checker.R
```

Paths inside the R code are resolved with [`here`](https://here.r-lib.org/) relative to the project root (`main.Rproj`), so notebooks and scripts run correctly regardless of the working directory.

---

## Requirements

**Python** — see the link above for environment setup.

Core packages: `pandas`, `numpy`, `matplotlib`, `pathlib`, `pyarrow`, `openpyxl`

**R** (developed on R 4.5.1)

```r
install.packages(c(
  "here", "rmarkdown", "knitr",
  "arrow", "dplyr", "tidyr", "tibble", "purrr", "stringr",
  "ggplot2", "patchwork", "scales", "signal", "glue"
))
```

---

## Running the pipeline

### 1. Place the data

Copy the acquisition folders into `data/raw_signal/` and the participant Excel files into `data/participants/`:

```
data/raw_signal/emg/<RUN_ID>_DELSYS.csv
data/raw_signal/nirs/<RUN_ID>*.txt
data/raw_signal/bia/<RUN_ID>_BIA*.pkl
data/participants/<RUN_ID[:7]>*.xlsx
```

Raw data files are **not** included in the repository (file size); only sample datasets provided separately are intended for testing.

### 2. Preprocess each participant — `main.ipynb`

Open `main.ipynb`, set the run id in the second cell:

```python
RUN_ID = "your_run_id_here"   # folder name of the participant dataset
```

and run the notebook top to bottom. Most steps are automatic; two require manual input:

- **Sequence picker (`_03a_`):** a Matplotlib widget opens over the torque signal — click to mark the start and end of each protocol sequence, then close the window.
- **VC detection (`_04_`):** review the automatic contraction detection, adjust the threshold knobs if needed, then run the commit cell.

Each step writes a Parquet cache. Already-run steps load from cache; pass `force_recompute=True` to rerun a step. The synchronized exports land in `results/DATA_EXPORT/<RUN_ID>/`.

### 3. Run the R analysis — `main.Rmd`

Open `main.Rproj` in RStudio, then knit `main.Rmd` (or run `Rscript main.Rmd`). It orchestrates the R stages behind switches that are **off by default** — the cohort analyses run off the cached per-run Parquet tables, so the cohort results reproduce without re-processing every run:

| Switch | Stage |
|---|---|
| `RUN_BATCH` | re-extract per-run features for all runs (slow) |
| `REBUILD_REGISTRY` | recompute the `cohort_final` subset registry |
| `KNIT_COHORTS` | render the per-modality + multimodal cohort notebooks |
| `RUN_ML` | leave-one-subject-out torque regression |

Individual notebooks under `notebooks/` can also be opened and knit directly. The per-run individual pipelines take a `RUN_ID` parameter.

The scripts that compute each reported statistic live in `sources/R/stats/` and run standalone from the project root (e.g. `Rscript --vanilla sources/R/stats/torque_stats.R`).


---

## Participant Excel template

Each participant requires one Excel file in `data/participants/`, named starting with the participant key (`RUN_ID[:7]`). A blank template is provided at `data/participants/TEMPLATE_Infos.xlsx`.

Required fields: participant ID, date of birth, sex, height, mass, Delsys init timestamp, and run number. The pipeline fails loudly if any required field is missing or unparseable.

The participant ID format is `XXXNnPp` (three random digits + two-letter abbreviations of surname and first name). The Delsys timestamp must follow `YYYY/MM/DD hh:mm:ss.ms` as recorded by the Trigno software.

---

## What the pipeline produces

| Output | Location | Description |
|---|---|---|
| Synchronized Parquet exports | `results/DATA_EXPORT/<RUN_ID>/` | source of truth for the R analysis |
| QC figures | `results/QC_EXPORT/` | per-participant quality-control plots |
| Per-run feature tables | `results/R_ANALYSIS/<RUN_ID>/` | EMG / BIA / NIRS / Myoton features |
| Cohort tables | `results/R_ANALYSIS/cohort/` | pooled feature tables across participants |
| ML results | `results/R_ANALYSIS/ml_results/` | LOSO metrics, predictions, feature importance |

`results/` is regenerated by the pipeline and is not shipped with the repository.

---

## Notes

- Developed and tested on a standard Windows laptop (8 GB RAM). EMG files are large; the Parquet caching architecture is designed to avoid keeping raw signals in memory across steps.
- The static-hold (`EX_STA`) tables are produced by the pipeline but are exploratory — the reported analysis uses the intermittent fatiguing task.

---

## License

Released under the GNU General Public License v3.0 (see `LICENSE`).
