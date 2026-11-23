# Telco Customer Churn Prediction

> A reproducible machine-learning comparison of two telecom customer churn datasets.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
#    Place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/

# 4. Reproduce every reported result (tests -> IBM 7 models -> 3-dataset
#    benchmark -> ablation A1 -> Iranian audit -> reports/results_snapshot.md)
python -m src.reproduce          # final tuned protocol (~15 min)
python -m src.reproduce --quick  # untuned smoke run
#    or only the IBM training run:
python -m src.train --tune --with-shap

# 5. Run API
uvicorn api.main:app --reload --port 8000

# 6. Open frontend
#    Open frontend/index.html in browser
#    Or visit http://localhost:8000/static/index.html
```

## Current research protocol

The shared comparison protocol uses a stratified 70/15/15
train/validation/test split (group-aware for Iranian, ~71/14/14), in-pipeline
leak-free feature engineering (`TelcoFeatureEngineer`), in-pipeline imputation,
model-specific scaling, calibrated probabilities (`CalibratedClassifierCV`
Platt scaling for LogReg, SVM_Linear, SVM_RBF, and the deployed RF), and
validation-only threshold optimization.

Models compared on IBM Telco: Dummy, LogReg, DecisionTree, **RandomForest
(main method)**, KNN, SVM_Linear and SVM_RBF. The comparison table also has
`RF_Calibrated@0.5` and `RF_Calibrated@<threshold>` rows for the deployed model.

For IBM runs, the default policy drops TotalCharges because it is a redundant
post-tenure aggregate; ablation A1 shows keeping it changes RF ROC-AUC by only
about +0.002.

**All reported numbers live in [`reports/results_snapshot.md`](reports/results_snapshot.md)**,
which is generated from `results/` by `python -m src.export_results` (run
automatically by `src.reproduce`). Do not copy numbers into documents by hand.

The training run writes config, metadata, dataset hash, policy metrics,
training logs, and SHAP explainability charts. See `mustdo.md` for the manual
decisions and evidence checks that remain before submitting a report.

## Multi-Dataset Benchmark (3 Scenarios)

Three distinct telecom churn datasets are evaluated as independent benchmarks:
1. **IBM Telco (Primary — Noisy & High Categorical Diversity):** 7,043 samples, 26.54% churn. Primary benchmark for the 7-model comparison and cost-based threshold optimization. On this mostly-categorical data RF does **not** beat LogReg/SVM_Linear on ROC-AUC/AP (see snapshot §1).
2. **Orange Telecom (Secondary — Quantitative Usage & Tree Advantage):** 3,333 samples, 14.49% churn. Numeric usage features; RF clearly beats LogReg on AP and ROC-AUC (snapshot §3).
3. **Iranian Churn (Control — High Separability & Group-Aware Split):** 3,150 samples, 15.71% churn. Contains the complaint / usage-frequency / tariff features named in the project brief. Evaluated with a group-aware split so duplicate feature vectors never cross train/test; 10-seed stability in snapshot §4.

Run all three datasets with:

    python -m src.compare_datasets --dataset all

Or run a specific dataset:

    python -m src.compare_datasets --dataset orange
    python -m src.compare_datasets --dataset ibm
    python -m src.compare_datasets --dataset iranian

The multi-dataset benchmark table is saved to `results/dataset_runs/dataset_comparison.csv`
and the comparative chart is saved to `figures/dataset_runs/dataset_comparison_metrics.png`.

## Project Structure

```
telco-churn-prediction/
├── data/raw/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # IBM Telco dataset
│   ├── iranian-uci/                           # Iranian Churn dataset (UCI)
│   └── orange-telecom/                        # Orange Telecom Churn dataset
├── models/                    # Trained models + config + metadata
├── notebooks/                 # EDA & experiments (research only)
├── src/                       # Production ML code
│   ├── config.py              # Project configuration & paths
│   ├── data_loader.py         # Multi-dataset loader & validation
│   ├── preprocessing.py       # Leak-free feature engineer & pipeline
│   ├── train.py               # Main training script (calibrated models)
│   ├── compare_datasets.py    # 3-dataset benchmark runner
│   ├── evaluate.py            # Comprehensive evaluation & charts
│   ├── explain.py             # SHAP (aggregated to original features)
│   ├── optimize_threshold.py  # Cost-based threshold search
│   ├── iranian_audit.py       # Iranian duplicate audit + 10-seed stability
│   ├── reproduce.py           # One-command rerun of every reported artifact
│   └── export_results.py      # results/ -> reports/results_snapshot.md
├── reports/                   # Tracked, auto-generated results snapshot
├── api/                       # FastAPI backend (/predict, /health)
├── frontend/                  # Modern interactive UI
├── figures/                   # Generated evaluation & comparison figures
├── results/                   # Metric outputs, comparison CSVs, logs
├── tests/                     # 20 unit tests for pipelines, splits & leak prevention
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Milestones

| # | Milestone | Goal |
|---|-----------|------|
| M1 | Data + MVP Model | RF baseline chạy được |
| M2 | ML Research | Tuning + DT/KNN/SVM comparison |
| M3 | Business Layer | Cost threshold + risk levels |
| M4 | Product | API + Frontend + SHAP |
| M5 | Deploy | Docker + public URL |

## References

1. *Trustworthiness Audit of the Telco Customer Churn Benchmark* (2026)
2. *Comprehensive Analysis of Churn Prediction* — Chen et al. (2025)
3. *Adaptive Ensemble Learning for Churn* (2024)
4. *e-Profits: Business-Aligned Evaluation Metric* (2025)
5. *A Survey on Churn Analysis* (2020)
