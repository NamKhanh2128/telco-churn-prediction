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

# 4. Train model
python -m src.train

# 5. Run API
uvicorn api.main:app --reload --port 8000

# 6. Open frontend
#    Open frontend/index.html in browser
#    Or visit http://localhost:8000/static/index.html
```

## Current research protocol

The shared comparison protocol uses a stratified 70/15/15
train/validation/test split, in-pipeline imputation, model-specific scaling,
calibrated Random Forest probabilities and validation-only threshold selection.
For IBM runs, the default policy drops TotalCharges because it is a redundant
post-tenure aggregate; Iranian Churn has no TotalCharges column.

The training run writes config, metadata, dataset hash, policy metrics and
training logs. See mustdo.md for the manual decisions and evidence checks that
remain before submitting a report or deploying the demo.

## Dataset comparison

IBM Telco and Iranian Churn are evaluated as separate experiments with the same
split, preprocessing, calibration and threshold protocol. They are never
concatenated. Run both with:

    python -m src.compare_datasets --dataset both

The comparison is saved to results/dataset_runs/dataset_comparison.csv. Run
each dataset with --tune only after deciding that the additional compute is
needed for the final comparison. The generated figures are saved under
figures/ and include value labels on bars/important points plus an aligned
metric note below each chart. The two-dataset figure is
figures/dataset_runs/dataset_comparison_metrics.png.

## Project Structure

```
telco-churn-prediction/
├── data/raw/                  # IBM Telco dataset
├── models/                    # Trained models + config
├── notebooks/                 # EDA & experiments (research only)
├── src/                       # Production ML code
│   ├── config.py              # Project configuration
│   ├── data_loader.py         # Load & validate data
│   ├── preprocessing.py       # Leak-free pipeline
│   ├── train.py               # Training script (entry point)
│   ├── evaluate.py            # Metrics & visualization
│   ├── explain.py             # SHAP (aggregated to original features)
│   └── optimize_threshold.py  # Cost-based threshold search
├── api/                       # FastAPI backend
├── frontend/                  # Static frontend
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
