# Iranian Churn Dataset

## Metadata

- **Source:** UCI Machine Learning Repository, Iranian Churn, dataset id `563`
- **Official Page:** https://archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset
- **Direct Archive URL:** https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip
- **DOI:** `10.24432/C5JW3Z` (https://doi.org/10.24432/C5JW3Z)
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Downloaded / Verified:** 2026-11-27 (via `python -m src.download_data`)
- **File:** `Customer Churn.csv`
- **Rows / Features:** 3,150 rows × 13 predictive features + 1 binary target (`Churn`, 15.71% churn rate)
- **SHA-256:** `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`

## Introductory Paper (Citation)

- **Authors:** Ruholla Jafari-Marandi, Joshua Denton, Adnan Idris, Brian K. Smith, Abbas Keramati
- **Year:** 2020
- **Title:** *Optimum Profit-Driven Churn Decision Making: Innovative Artificial Neural Networks in Telecom Industry*
- **Journal:** *Neural Computing and Applications*, Vol. 32, pp. 14929–14962
- **DOI:** `10.1007/s00521-020-04850-6` (https://doi.org/10.1007/s00521-020-04850-6)

## Notes

- Features aggregate customer activity over months 1–9, and `Churn` is observed at month 12 (3-month planning horizon per UCI metadata), though the CSV lacks per-row timestamps.
- Contains 300 exact duplicate rows and 14 conflicting feature-duplicate groups; `src/compare_datasets.py` uses a leak-free group-aware split by default and never concatenates this dataset with IBM Telco.
