# Raw Datasets Overview & IBM Telco Customer Churn

All raw CSV files are git-ignored (`*.csv`). Run `python -m src.download_data` to download Iranian Churn and Orange Telecom automatically and verify SHA-256 hashes.

## 1. IBM Telco Customer Churn (Manual Download)

- **File:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Source:** IBM Sample Data Sets (IBM Cognos Analytics sample library), popularized on Kaggle by `blastchar`:
  - Kaggle URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **License:** Original data files © IBM / Original Authors (intended for sample demonstrations and educational/research use; IBM's official companion repository `IBM/telco-customer-churn-on-icp4d` is under Apache License 2.0, while the Kaggle mirror does not specify a standalone OSI license).
- **Download Note:** Kaggle requires authentication, so place the CSV manually in `data/raw/` and run `python -m src.download_data --check-only` to verify its hash.
- **Downloaded / Verified:** 2026-11-27 (manual CSV placed in `data/raw/` and verified via `python -m src.download_data --check-only`)
- **Rows / Features:** 7,043 rows × 21 columns (`customerID` + 19 predictive features + `Churn`, ~26.54% churn rate).
- **SHA-256:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

## 2. Benchmark Datasets

- **Iranian Churn (UCI id 563):** stored in `data/raw/iranian-uci/` — see [iranian-uci/README.md](iranian-uci/README.md).
- **Orange Telecom Churn:** stored in `data/raw/orange-telecom/` — see [orange-telecom/README.md](orange-telecom/README.md).
