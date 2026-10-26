# Datasets

## Adult / Census Income (recommended real dataset — real informative missingness)
- Source: https://archive.ics.uci.edu/dataset/2/adult  (download `adult.data`)
  - Or let the loader fetch it from OpenML automatically (no file needed).
- Real informative missingness: workclass / occupation / native-country are '?'
  for a non-random subset whose income distribution differs. Kept as-is.
- Target: income >50K (binary).
- Use:
    python -m src.run_experiment --real adult --path data/raw/adult.data
    # or, auto-download from OpenML:
    python -m src.run_experiment --real adult

## MIMIC-IV Clinical Database Demo (healthcare — open, no credentialing)
- Source: https://physionet.org/content/mimic-iv-demo/
- Extract an admissions-level table with 'hospital_expire_flag' + numeric cols.
- Edit loaders.load_mimic_demo() column names to match your extraction.
- Full MIMIC-IV requires PhysioNet credentialing (data-use agreement + CITI training).

## Give Me Some Credit (finance — optional alternative)
- Original competition page requires rules acceptance; use a mirror instead:
  - https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset
  - or raw CSV: https://github.com/JLZml/Credit-Scoring-Data-Sets (cs-training.csv)
- Place at data/raw/cs-training.csv; target SeriousDlqin2yrs.

## Synthetic (built-in, no download)
- src/benchmark.py generates clinical + credit bases with exact ground truth.
