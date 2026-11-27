# Orange Telecom Churn Dataset

## Metadata

- **Source:** Classic telecom churn benchmark attributed to Orange S.A. / France Télécom (widely circulated via BigML and ML teaching repositories), downloaded from the `mlcourse.ai` mirror.
- **Download URL:** https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/telecom_churn.csv
- **Upstream Mirror Repo:** https://github.com/Yorko/mlcourse.ai
- **License & Provenance Limitation:** The `mlcourse.ai` repository is licensed under `CC BY-NC-SA 4.0`, whereas the underlying `telecom_churn.csv` mirror does not include an explicit standalone upstream dataset license from Orange S.A. / BigML. **Limitation:** exact original collection period and formal dataset redistribution license cannot be independently verified beyond educational/research mirror usage.
- **Downloaded / Verified:** 2026-11-27 (via `python -m src.download_data`)
- **File:** `telecom_churn.csv`
- **Format:** CSV
- **Rows / Features:** 3,333 rows × 19 predictive attributes + 1 binary target (`Churn`: True 483 [14.49%], False 2,850 [85.51%])
- **SHA-256:** `5d0aecc74b6feda887e9d2735363d51ff671b373153863562b7bbfb36464e0a4`

## Feature Overview

- **Customer Account:** `State`, `Account length`, `Area code`, `International plan`, `Voice mail plan`, `Number vmail messages`
- **Day Usage:** `Total day minutes`, `Total day calls`, `Total day charge`
- **Evening Usage:** `Total eve minutes`, `Total eve calls`, `Total eve charge`
- **Night Usage:** `Total night minutes`, `Total night calls`, `Total night charge`
- **International Usage:** `Total intl minutes`, `Total intl calls`, `Total intl charge`
- **Customer Care Activity:** `Customer service calls`

## Role in Research

Serves as the **Intermediate Benchmark (Continuous Usage Dominant)** bridging the gap between IBM Telco (categorical-dominant, noisy) and Iranian Churn (high-separability). Demonstrates the advantage of tree-based models (Random Forest) over linear models on non-linear usage thresholds (e.g., high customer service calls or international plan usage).
