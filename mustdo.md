# MUSTDO

Cap nhat: 2026-10-10

File nay ghi rieng cac viec khong the tu dong quyet dinh mot cach trung thuc:
nguon du lieu, chi phi kinh doanh, ket luan nghien cuu, noi dung bao cao va
quyen truy cap deploy. Cac viec ky thuat co the kiem tra trong repo da duoc
tu dong sua va danh dau o phan tren.

## Da tu dong hoan thien

- [x] Kiem tra schema, target, duplicate ID/row, missing TotalCharges va hash dataset.
- [x] Dung split train/validation/test theo ti le 70/15/15; threshold chi duoc chon tren validation.
- [x] Dat chinh sach primary la bo TotalCharges; co co che ablation reproducible voi --keep-total-charges.
- [x] Dua imputation vao pipeline va chi scale numeric cho model can scale.
- [x] Them RandomizedSearchCV multi-metric khi bat DO_TUNING.
- [x] Them AP/PR-AUC, balanced accuracy, Brier, log-loss, ECE va lift top-10%.
- [x] Them nhan gia tri tren bar/diem quan trong, legend AUC/AP va bang metric
  can le ben duoi tung figure; them metric_summary va figure comparison hai dataset.
- [x] Them calibration, threshold config, model metadata, dataset hash va training log.
- [x] Lam API doc theo metadata, cho phep TotalCharges optional va sua mutable default trong schema.
- [x] Sua Docker port de nhan PORT cua hosting.
- [x] Them test cho scaling, metric calibration/ranking va hai cost mode.

## So sanh hai dataset truoc khi chot

- [x] Da tai Iranian Churn tu UCI vao data/raw/iranian-uci/ va ghi source,
  DOI, license va SHA-256.
- [x] Da tao runner doc lap tai src/compare_datasets.py; IBM va Iranian khong
  bi concatenated va artifact khong ghi de len nhau.
- [x] Da chay quick comparison va luu ket qua tai
  results/dataset_runs/dataset_comparison.csv.
- [x] Da sinh figure co nhan gia tri tai
  figures/dataset_runs/dataset_comparison_metrics.png.

Quick snapshot 2026-10-09, chua tuning:

| Dataset | RF AP | RF ROC-AUC | RF F1 theo policy | Brier | Lift top-10% |
|---|---:|---:|---:|---:|---:|
| IBM Telco | 0.6047 | 0.8194 | 0.5524 | 0.1441 | 2.6986 |
| Iranian Churn | 0.9157 | 0.9858 | 0.8348 | 0.0292 | 5.8586 |

Day la ket qua so bo cua cung mot quick protocol, khong phai ket luan cuoi.
Hai dataset khac nhau ve cohort, feature semantics va ti le churn; can tuning
va xem muc tieu de tai truoc khi chon dataset chinh.

- [ ] Chay lai ca hai voi --tune neu can so sanh final:

  python -m src.compare_datasets --dataset ibm --tune
  python -m src.compare_datasets --dataset iranian --tune

- [ ] Tu quyet dinh dataset chinh sau khi xem AP/PR-AUC, ROC-AUC,
  balanced accuracy, Brier/ECE, lift top-10%, class prevalence va do phu hop
  voi cau hoi nghien cuu. Khong chon chi vi accuracy cao hon.

## Ban phai lam thu cong

### P0 — bat buoc truoc khi nop bao cao

- [ ] Xac nhan voi giang vien/mentor dataset chinh sau khi xem bang so sanh.
  Hien da chay hai dataset rieng; khong gop IBM voi Iranian Telco trong cung
  mot model. Cap nhat README, citation, audit hash va mo ta feature theo dataset
  duoc chon.

- [ ] Xac nhan chi phi that cua business. COST_FP=50 va COST_FN=500 chi la
  placeholder. Chon mot trong hai cach dien giai:
  classification cost = FP + FN; hoac action cost = chi phi moi khach hang
  bi flag + chi phi bo sot. Ghi ly do va don vi tien trong bao cao.

- [ ] Sau khi chon dataset, chay protocol final va doc ket qua, khong chi copy
  metric. Neu chon IBM, dung:

  python -m src.train --tune

  Neu chon Iranian, dung:

  python -m src.compare_datasets --dataset iranian --tune

  API va frontend hien tai dang nhap schema IBM. Neu chon Iranian lam model
  chinh, phai cap nhat schema/request fields va model path truoc khi deploy;
  khong lay artifact Iranian gan vao API IBM ma khong kiem tra.

  Kiem tra cac file models/config.json, models/metadata.json,
  results/test_metrics.json, results/policy_metrics.json,
  results/training_log.json va cac hinh trong figures/. Chu y cac hinh
  metric_summary.png, cost_threshold.png va dataset_runs/dataset_comparison_metrics.png;
  cac gia tri quan trong da duoc ghi ngay tren bar/diem va trong note ben duoi.

- [ ] Neu chon IBM, chay ablation A1 an toan trong thu muc rieng:

  python -m src.compare_datasets --dataset ibm --keep-total-charges

  Ket qua nam tai results/dataset_runs/ibm_with_total_charges va khong ghi de
  model IBM root.

  Bao cao phai ghi ro day la ablation, khong dung ket qua A1 thay cho primary
  neu chua co ly do phuong phap hoc.

- [ ] Neu can chay nhanh de debug, bo --tune va ghi ro day la baseline. Ban
  khong duoc goi ket qua quick run la benchmark toi uu khi chua tune.

- [ ] Dien bang ket qua that vao bao cao: ROC-AUC, AP/PR-AUC, balanced accuracy,
  recall churn, precision churn, F1, Brier/ECE, lift top-10%, threshold va
  chi phi validation/test. Khong tu dien so lieu khi chua co artifact.

- [ ] Kiem tra giao dien va API bang mot mau khach hang:

  uvicorn api.main:app --reload --port 8000

  Mo /docs, goi /health va /predict; xac nhan probability, threshold,
  risk level, top factors va recommendation phu hop voi model da train.

- [ ] Neu can hinh SHAP cho phan explainability, chay them:

  python -m src.train --with-shap

  SHAP la global/aggregated explanation trong prototype; khong viet thanh
  causal explanation va khong thay the validation cua business intervention.

### P1 — hoan thien chat luong nghien cuu va demo

- [ ] Kiem tra citation, license, provenance va ngay truy cap dataset; phan biet
  bai peer-reviewed voi preprint. Khong giu cac claim audit hoac savings neu
  khong truy ve duoc nguon va phep tinh.

- [ ] Lap lai primary voi it nhat 3 random seeds hoac repeated stratified CV,
  bao cao mean va do lech chuan; neu khong lam duoc, ghi limitation.

- [ ] Viet EDA va limitation tu ket qua that: class imbalance, missingness,
  calibration, concept drift, bias, khong co intervention/outcome retention va
  khong duoc dien giai causal.

- [ ] Chon kenh deploy va tu tao credential/account neu can: Render, Streamlit
  Community Cloud hoac Hugging Face Spaces. Khong commit secret. Kiem tra
  start command, PORT, model artifact va public URL tren hosting.

- [ ] Rehearse defense: giai thich leakage, split, calibration, threshold,
  classification-vs-action cost, SHAP/global-vs-local explanation va vi sao
  TotalCharges duoc drop trong primary.

## Acceptance checklist

- [ ] Co mot protocol primary duoc mentor phe duyet.
- [ ] Co A1 va neu co the A2 calibration/seed ablation.
- [ ] So lieu trong bao cao khop artifact trong results/.
- [ ] API doc va demo chay duoc tu model artifact cung version metadata.
- [ ] README va citation khop dataset/ket qua thuc te.
