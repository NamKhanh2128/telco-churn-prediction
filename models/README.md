# Trained models will be saved here after running:
#   python -m src.train --tune --with-shap           (deployed IBM model)
#   python -m src.compare_datasets --dataset all --tune   (per-dataset runs, tuned parameters)
#
# Files:
#   rf_best.joblib     - Deployed model: tuned Random Forest pipeline + Platt calibration
#                        (one forest; not in git, regenerate with the commands above)
#   config.json        - Threshold + cost configuration read by the API
#   metadata.json      - Feature columns, calibration settings, data SHA-256, versions
#   tuned_params.json  - Best hyper-parameters of the official tuned run, per dataset run
#                        and model; every analysis script reads them (tracked in git)
#   dataset_runs/      - Calibrated RF of each dataset run (not in git)
