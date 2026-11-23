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
- [x] Da chay lai ca hai dataset voi --tune; artifact tuned nam rieng tai
  results/dataset_runs/ibm/ va results/dataset_runs/iranian/. Khi doc ket qua,
  uu tien cac artifact per-dataset va file dataset_comparison hien tai; mot
  stdout tong hop cu co the hien IBM baseline neu bi ghi tu lan chay truoc.

### Audit Iranian va do on dinh

- [x] Da audit schema, missingness, direct target-copy va feature association
  cua Iranian tai src/iranian_audit.py.
- [x] Da phat hien 314 dong co feature vector trung, 162 nhom feature trung
  va 14 nhom trung nhung khac label. Trong split row-wise seed 42, 86/630
  test rows trung feature voi train. Day la rui ro duplicate leakage; khong
  dung benchmark row-wise nay lam ket luan cuoi neu chua co giai phap.
- [x] Da chay 10 seeds cho ca row-wise va group-aware split. Bao cao nam tai
  results/dataset_runs/iranian/audit_report.json,
  feature_audit.csv, stability_runs.csv, stability_summary.csv va
  stability_report.json.
- [x] Da sinh figures/dataset_runs/iranian_stability_metrics.png voi mean,
  +/- SD, error bars va note placeholder cost; khong viet savings thanh
  business claim cuoi.
- [x] Da kiem tra group-aware: khong co test row nao trung feature voi train;
  AP/AUC van cao, nhung threshold va policy metrics dao dong giua seed.

- [x] Da thiet ke Feature Engineering cho IBM Telco (charges_per_tenure, total_additional_services, bundle, high_risk_contract) leak-free.
- [x] Da calibration xac suat toan dien (Platt scaling) cho LogReg (ECE giam tu 0.150 xuong 0.018) va SVM_Linear (xuat predict_proba, Brier va ECE day du).
- [x] Da chot Group-aware split mac dinh cho Iranian Churn de loai bo 100% duplicate leakage, dong nhat thang chi phi voi IBM Telco.
- [x] Da bo sung va tich hop Dataset thu 3: Orange Telecom Churn (3.333 mau, 20 dac trung, churn 14.49%) tai data/raw/orange-telecom/.
- [x] Da tich hop runner benchmark 3 datasets tai src/compare_datasets.py (--dataset all), xuat dataset_comparison.csv va figures/dataset_runs/dataset_comparison_metrics.png.

- [x] Sua split ve dung 70/15/15 (truoc do config la TEST_SIZE=0.20 -> thuc te 65/15/20);
  group-aware split cua Iranian chon so fold theo ti le muc tieu (~71/14/14).
- [x] Them SVM_RBF (SVC kernel RBF, Platt-calibrated, co tuning) vao so sanh IBM.
- [x] Them dong RF_Calibrated@0.5 va RF_Calibrated@<nguong> vao comparison_table de
  khong tron metric RF chua calibrate voi model deploy.
- [x] Them src/reproduce.py (chay lai toan bo) va src/export_results.py
  (results/ -> reports/results_snapshot.md, file nay duoc commit).

So lieu hien hanh: xem reports/results_snapshot.md (khong chep tay so vao day).

## Ban phai lam thu cong

### P0 — bat buoc truoc khi nop bao cao

- [ ] Xac nhan provenance va thoi diem co san cua cac feature Iranian co rui ro
  proxy/temporal semantics: Complains, Status va Customer Value. Nguon UCI mo
  ta feature duoc tong hop trong 9 thang dau va nhan churn o cuoi thang 12,
  nhung file CSV khong co timestamp cho tung dong. Phai xac nhan cac feature
  nay thuc su co san tai thoi diem scoring va khong duoc tinh sau churn.

- [ ] Xu ly/bao ve duplicate leakage Iranian. Can hoi source owner ve 300
  exact duplicate rows va 14 nhom feature trung nhung khac label. Neu khong
  co customer_id/time de group, dung group-aware protocol lam primary va ghi
  limitation; khong tu y xoa cac nhom conflicting-label.

- [ ] Xac nhan voi giang vien/mentor dataset chinh sau audit. Iranian co
  group-aware AP va ROC-AUC rat cao qua 10 seeds (xem reports/results_snapshot.md §4),
  va la dataset DUY NHAT co dung cac feature de tai nhac (khieu nai, tan suat su dung,
  goi cuoc). Neu chot Iranian, khong gop IBM voi Iranian trong cung mot model; cap
  nhat README, citation, audit hash, API/frontend va mo ta feature theo schema Iranian.

- [ ] Xac nhan chi phi that cua business. COST_FP=50 va COST_FN=500 chi la
  placeholder. Chon mot trong hai cach dien giai:
  classification cost = FP + FN; hoac action cost = chi phi moi khach hang
  bi flag + chi phi bo sot. Ghi ly do, don vi tien, va cach uoc tinh trong
  bao cao. Khong dung validation savings voi $50/$500 lam business claim.

- [x] Chay lai protocol final bang MOT lenh (2026-11-24, sau khi sua split 70/15/15
  va them SVM_RBF):

  python -m src.reproduce

  Lenh nay chay pytest -> src.train --tune --with-shap -> compare_datasets --dataset all
  --tune -> ablation A1 -> iranian_audit, ghi results/reproduce_log.json va sinh
  reports/results_snapshot.md. MOI so lieu trong bao cao phai lay tu snapshot nay.

  Doc ket qua (khong chi copy metric):
  - IBM Telco: RF KHONG thang ro LogReg/SVM_Linear ve ROC-AUC/AP (cac model tuyen tinh
    ngang hoac nhinh hon). DecisionTree va RF co recall/F1 churn cao nhat o nguong 0.5
    nho class_weight. Bao cao phai noi thang dieu nay, khong viet "RF tot nhat" tren IBM.
  - Orange Telecom va Iranian: RF vuot ro LogReg ve AP va ROC-AUC -> day la bang chung
    chinh cho lua chon RF (bien dinh luong, quan he phi tuyen).
  - Nguong toi uu chi phi tren IBM ~0.06 (recall ~0.98, precision ~0.36) la HE QUA cua
    ti le placeholder FN:FP = 10:1, chua phai chinh sach that.

- [x] Ablation A1 (giu TotalCharges) chay trong thu muc rieng
  results/dataset_runs/ibm_with_total_charges/: thay doi ROC-AUC chi khoang +0.002
  -> giu quyet dinh drop TotalCharges o primary.

- [x] Kiem tra API bang TestClient voi model moi: /health 200, threshold 0.0633;
  /predict 200 cho ca khach hang rui ro cao va thap.
  LUU Y DEMO: voi nguong 0.0633, mot khach hang hop dong 2 nam, tenure 60 thang
  (p = 0.085) van bi gan nhan "Churn". Can chot chi phi that (P0) hoac dung chinh
  sach top-k/ngan sach cham soc truoc khi demo, neu khong hoi dong se hoi.

- [x] Da chay SHAP explainability va sinh figure day du:

  python -m src.train --with-shap

  Da tao xong figures/shap_summary.png va figures/feature_importance_aggregated.png
  (SHAP values duoc tong hop tu one-hot ve dac trung goc nhu Contract, tenure, MonthlyCharges).

### P1 — hoan thien chat luong nghien cuu va demo

- [ ] Kiem tra citation, license, provenance va ngay truy cap dataset; phan biet
  bai peer-reviewed voi preprint. Khong giu cac claim audit hoac savings neu
  khong truy ve duoc nguon va phep tinh.

- [x] Lap lai Iranian voi 10 seeds theo ca row-wise va group-aware split,
  bao cao mean, do lech chuan, min/max va overlap. Ket qua cho thay ranking
  on dinh hon policy threshold; can chot group-aware lam protocol primary neu
  khong co customer_id/time.

- [ ] Sau khi xac nhan business cost, lap lai threshold optimization voi
  COST_FP/COST_FN that va bao cao do nhay cua threshold/savings theo seed.

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
