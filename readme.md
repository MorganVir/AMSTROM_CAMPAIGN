# AMSTROM Campaign

Multimodal preprocessing and (early) analysis pipeline for muscular fatigue data collected during a quadriceps fatigue protocol. Synchronizes sEMG, torque, NIRS, bio-impedance and Myoton recordings from independent acquisition systems onto a shared time axis, then extracts fatigue-related EMG features at the cohort level.

Built on top of the the [Python minimal environment for reproducible research](https://github.com/DenisMot/Python-minimal-install).

---

## Requirements

**Python** (see link above for environment setup)

Core packages: `pandas`, `numpy`, `matplotlib`, `pathlib`, `pyarrow`

**R**

```r
install.packages(c("arrow", "dplyr", "tidyr", "tibble", "ggplot2", "signal", "glue"))
```

---

## Repository structure

```
AMSTROM_CAMPAIN/
├── main.ipynb                  # Dashboard - run the Python pipeline from here
├── main.Rmd                    # Single-run R analysis
├── cohort_analysis.Rmd         # Cohort-level EMG analysis (pools all runs)
├── sources/                    # Python preprocessing scripts (_01_ to _09_)
├── data/
│   ├── raw_signal/             # Raw acquisition files (emg/, nirs/, bia/, myoton/)
│   └── participants/           # One Excel template per participant
└── results/
    ├── DATA_EXPORT/            # Preprocessed parquet exports (source of truth for R)
    ├── QC_EXPORT/              # QC PDF figures
    └── R_ANALYSIS/             # R-derived feature tables, per run and cohort
```

---

## Quick start — running the pipeline on sample data

One sample dataset is provided as a working example (`123DuPi_20251015`). It includes raw signal files and a pre-filled participant Excel template.

**1. Place the sample data**

Copy the provided folders into `data/raw_signal/` and the participant Excel files into `data/participants/`. The expected folder structure is:

```
data/raw_signal/emg/<RUN_ID>_DELSYS.csv
data/raw_signal/nirs/<RUN_ID>*.txt
data/raw_signal/bia/<RUN_ID>_BIA*.pkl
data/participants/<RUN_ID[:7]>*.xlsx
```

**2. Set the RUN_ID**

Open `main.ipynb`. In the second cell, set:

```python
RUN_ID = "your_run_id_here"   # folder name of the participant dataset
```

**3. Run the notebook top to bottom**

Most steps are automatic. Two steps require manual input:

- **Sequence picker (_03a_):** a Matplotlib widget opens over the torque signal. Click to mark the start and end of each protocol sequence, then close the window.
- **VC detection (_04_):** review the automatic contraction detection, adjust threshold knobs if needed, then run the commit cell to save to cache.

Each step writes a Parquet cache file. If a step has already been run, it loads from cache automatically. Set `force_recompute=True` in the call to rerun a step from scratch.

**4. Run the R analysis**

Open `main.Rmd` in RStudio. Set the `RUN_ID` variable in the first chunk to match the participant you just preprocessed, then knit or run chunk by chunk.

To pool results across multiple preprocessed runs, open `cohort_analysis.Rmd` and knit.

---

## Participant Excel template

Each participant requires one Excel file in `data/participants/`, named starting with the participant key (`RUN_ID[:7]`). A blank template is provided at `data/participants/TEMPLATE_Infos.xlsx`. The sample participant file (`123DuPi_20251015_Infos.xlsx`) can serve as a filled example.

Required fields: participant ID, date of birth, sex, height, mass, Delsys init timestamp, and run number. The pipeline will fail loudly if any requirsed field is missing or unparseable.

The participant ID format is `XXXNnPp` (three random digits + two-letter abbreviations of surname and first name). The Delsys timestamp must follow the format `YYYY/MM/DD hh:mm:ss.ms` as recorded by the Trigno software.

---

## What the pipeline produces

| Output | Location | Description |
|---|---|---|
| Synchronized parquet exports | `results/DATA_EXPORT/<RUN_ID>/` | Source of truth for R analysis |
| QC figures | `results/QC_EXPORT/` | Multimodal overlay plots per participant |
| Per-run R feature tables | `results/R_ANALYSIS/<RUN_ID>/` | EMG features, torque, calibration tables |
| Cohort tables | `results/R_ANALYSIS/cohort/` | Pooled feature tables across participants |

---

## Notes

- The pipeline was developed and tested on a standard Windows laptop (8 GB RAM). EMG files are LARGE in a df; the Parquet caching architecture is specifically designed to avoid keeping raw data in memory across steps.
- Raw data files are not included in the repository (file size). Only the two sample datasets provided separately are intended for testing.
- NIRS and BIA preprocessing is implemented and functional. Cohort-level feature analysis for those modalities is not yet complete.
