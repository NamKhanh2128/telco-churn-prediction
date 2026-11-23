# Orange Telecom Churn Dataset

## Metadata
- **Source:** Orange S.A. Customer Churn Dataset (KDD / BigML / mlcourse.ai mirror)
- **URL:** https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/telecom_churn.csv
- **Format:** CSV
- **Rows:** 3,333
- **Features:** 19 predictive attributes + 1 binary target (`Churn`)
- **Target Variable:** `Churn` (True: 483 [14.49%], False: 2,850 [85.51%])
- **SHA-256 Hash:** `5d0aecc74b6feda887e9d2735363d51ff671b373153863562b7bbfb36464e0a4`

## Feature Overview
- **Customer Account:** State, Account length, Area code, International plan, Voice mail plan, Number vmail messages
- **Day Usage:** Total day minutes, Total day calls, Total day charge
- **Evening Usage:** Total eve minutes, Total eve calls, Total eve charge
- **Night Usage:** Total night minutes, Total night calls, Total night charge
- **International Usage:** Total intl minutes, Total intl calls, Total intl charge
- **Customer Care Activity:** Customer service calls

## Role in Research
Serves as the **Intermediate Benchmark (Continuous Usage Dominant)** bridging the gap between IBM Telco (Categorical-dominant, noisy) and Iranian Churn (High-separability).
Demonstrates the strong advantage of Tree-based models (Random Forest) over Linear Models when handling quantitative non-linear thresholds (e.g. high customer service calls or international plan usage).
