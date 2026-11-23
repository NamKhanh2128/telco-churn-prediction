# Plan — Lê Hải Anh · Trưởng nhóm · Hệ thống & triển khai

- **Branch:** `haianh/system-deploy`
- **Vùng file phụ trách:**
  - Hệ thống: `api/`, `frontend/`, `Dockerfile`, `docker-compose.yml`, `requirements*.txt`,
    `src/download_data.py` (mới), `tests/test_api.py` (mới), `.github/`
  - Tài liệu: `README.md`, `.gitignore`, `PLAN.md`, `mustdo.md`
  - Báo cáo: `docs/report/00-outline.md`, `01-gioi-thieu.md`, `06-he-thong.md`, `07-1-ket-luan.md`
    và bản ráp cuối cùng
- **Mục tiêu:**
  1. Chốt các quyết định với giảng viên để cả nhóm không bị chặn.
  2. Môi trường chạy giống nhau cho mọi người: phiên bản thư viện, CI, dữ liệu.
  3. Demo công khai chạy đúng model của tag `v1.0-results`.
  4. Ráp báo cáo cuối cùng.

## Đọc trước

- `PLAN.md` (mục 3, 6–7), `mustdo.md` (mục P0), `reports/results_snapshot.md`
- `api/main.py`, `api/predict.py`, `api/schemas.py`, `frontend/index.html`, `frontend/app.js`
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- `tests/test_pipeline.py` (để biết test nào cần dữ liệu thật)

## Việc cần làm

### Tuần 1 (24–29/11) — chốt giao thức, môi trường chung

- [ ] **Chuẩn bị 1 trang tóm tắt cho giảng viên**, gồm:
  - 3 dataset; kết quả snapshot §1 và §3.
  - Lý do Iranian khớp mô tả đề tài (khiếu nại, tần suất sử dụng, gói cước…).
  - Rủi ro của dữ liệu Iranian: dòng trùng lặp; `Complains`/`Status`/`Customer Value` có thể chưa
    có tại thời điểm dự đoán.
  - Đề xuất chi phí của Nhật Linh.

  Quỳnh Anh và Nhật Linh gửi phần của họ trước 27/11.
- [ ] **Gặp hoặc gửi email giảng viên, chốt D1, D2, D3, D6** (bảng mục 3 trong `PLAN.md`). Điền cột
  "Kết quả" bằng một PR nhỏ vào `main` **trước 29/11**, rồi báo cả nhóm.
- [ ] Viết `docs/report/00-outline.md`: dàn ý chương/mục, số trang dự kiến, danh sách hình/bảng và
  người phụ trách, quy ước trình bày (đánh số hình, trích dẫn, thuật ngữ Việt–Anh).
- [ ] **Ghim phiên bản:** `requirements.txt` hiện chỉ có `>=`, nhưng file model `.joblib` phụ thuộc
  đúng phiên bản `scikit-learn`. Máy train hiện tại: Python 3.13, scikit-learn 1.8.0, pandas 2.3.3.
  Còn `Dockerfile` đang dùng `python:3.11-slim`.
  → Tạo `requirements-lock.txt` (`pip freeze`) hoặc ghim các gói chính, và đồng bộ phiên bản Python
  trong Docker với môi trường train.
- [ ] **CI (GitHub Actions):** `.github/workflows/ci.yml` chạy `python -m pytest -q` cho mọi PR.
  Các test đọc CSV thật sẽ fail trên CI vì dữ liệu không có trong git → thêm
  `pytest.mark.skipif(not path.exists(), ...)` cho các test đó (PR nhỏ, làm sớm; báo Nam Khánh vì
  `tests/test_pipeline.py` thuộc vùng của Khánh).
- [ ] **`src/download_data.py`:** tải Iranian (UCI) và Orange (mirror GitHub), kiểm tra SHA-256 theo
  bảng trong `PLAN.md`; IBM ghi hướng dẫn tải từ Kaggle. Giúp các bạn tự lấy dữ liệu đúng phiên bản.

### Tuần 2 (30/11–06/12) — API

- [ ] **Kích thước model deploy:** nén bằng `joblib.dump(..., compress=3)`, kết hợp phương án
  calibration mà Nam Khánh chốt. Mục tiêu < 25 MB; so metric trước/sau qua snapshot.
- [ ] **Nếu D1 = Iranian:** đổi API và giao diện sang schema Iranian:
  - `api/schemas.py`: 13 trường, có ràng buộc miền giá trị theo `docs/data_dictionary.md` của Quỳnh Anh.
  - Form trong `frontend/index.html` và `frontend/app.js`.
  - Hàm top factors trong `api/predict.py`.

  Hoặc thiết kế API chọn dataset theo `models/metadata.json`. Bắt đầu tuần 2, xong tuần 3.
- [ ] **Áp dụng chính sách D4 vào API** (ngưỡng / top-k / mức rủi ro theo đề xuất của Nhật Linh).
  Hiện `risk_level` dùng cứng 0.7 và ngưỡng chi phí.
  (Tùy chọn) top factors theo SHAP từng khách, hàm do Nam Khánh cung cấp.
- [ ] **Review PR** về API, giao diện, CI và tài liệu.

### Tuần 3 (07–13/12) — deploy và viết

- [ ] **Deploy (D5):** Render (Docker) hoặc Hugging Face Spaces (Docker SDK). Vì `*.joblib` bị ignore,
  chọn một cách đưa model của tag `v1.0-results` lên:
  - (a) upload model làm asset của GitHub Release gắn tag `v1.0-results`, Dockerfile tải về khi build;
  - (b) đẩy model lên Hugging Face Hub;
  - (c) nếu model sau khi nén < 50 MB, thêm ngoại lệ `!models/deploy/*.joblib` vào `.gitignore`.

  **Không commit token/secret** — dùng biến môi trường của nền tảng.
- [ ] Kiểm tra `/health`, `/predict`, giao diện trên URL công khai; ghi URL vào `README.md`.
- [ ] **`tests/test_api.py`** dùng `fastapi.testclient.TestClient`: `/health` 200; `/predict` 200 với
  1 khách rủi ro cao và 1 khách rủi ro thấp; 422 khi thiếu trường bắt buộc. Skip nếu chưa có model.
- [ ] **Chương 1 – Giới thiệu:** bối cảnh churn trong viễn thông, chi phí giữ chân so với tìm khách
  mới, phát biểu bài toán phân loại nhị phân, mục tiêu, phạm vi, đóng góp.
- [ ] **Chương 6 – Hệ thống demo:**
  - Sơ đồ kiến trúc: pipeline huấn luyện → model artifact + `metadata.json` → FastAPI → giao diện.
  - Đặc tả API (request/response mẫu), ảnh chụp giao diện, hướng dẫn chạy local/Docker, URL công khai.
- [ ] Làm 2–3 slide phần hệ thống/demo theo khung của Nhật Linh.

### Tuần 4 (14–20/12) — ráp báo cáo và bảo vệ

- [ ] **Mục 7.1 – Kết luận & hướng phát triển.**
- [ ] Ráp toàn bộ `docs/report/*.md` sang định dạng nộp (D6); thống nhất văn phong, đánh số
  hình/bảng, tài liệu tham khảo.
- [ ] Sau khi Nhật Linh đối chiếu số liệu (18/12): sửa các chỗ lệch còn lại khi ráp; tick `mustdo.md`
  và checklist mục 8 trong `PLAN.md`.
- [ ] `README.md` cuối cùng (dataset chính, link snapshot, URL demo, cách tái lập) và **demo offline
  dự phòng**: chạy local bằng `uvicorn` hoặc `docker compose up` phòng khi mạng hỏng; quay video demo ngắn.
- [ ] Review chéo mục 4.3, 5.1, 5.4 (Thùy Linh).
- [ ] Viết câu trả lời phần mình trong `docs/defense_qa.md`. Tổ chức 2 buổi tổng duyệt.

## Phụ thuộc

- **Cần từ:**
  - Quỳnh Anh: tóm tắt dữ liệu (27/11); data dictionary (cho form nhập liệu nếu đổi sang Iranian).
  - Nhật Linh: đề xuất chi phí D3 (27/11), chính sách D4; kết quả đối chiếu số liệu (18/12).
  - Nam Khánh: quyền truy cập và dữ liệu (25/11); model cuối và phương án calibration (06/12);
    (tùy chọn) hàm SHAP cho API.
- **Cung cấp cho:**
  - Cả nhóm: kết quả D1–D6 (29/11), CI, URL demo (13/12), báo cáo bản cuối.
  - Nam Khánh: D1 (dataset chính cho phân tích RF).

## Câu hỏi bảo vệ cần chuẩn bị

1. Bài toán churn có ý nghĩa kinh doanh gì? Vì sao dùng học máy thay vì luật cố định?
2. Kiến trúc hệ thống thế nào: từ model đã train tới API và giao diện?
3. Khi người dùng không nhập `TotalCharges` thì sao? Dữ liệu nhập sai kiểu được xử lý ra sao (lỗi 422)?
4. Vì sao phải ghim phiên bản thư viện khi deploy model `.joblib`?
5. Model được nạp và cập nhật ra sao khi train lại? Bí mật/credential được quản lý thế nào?
6. Nếu đưa vào dùng thật, hệ thống demo còn thiếu gì (giám sát, cập nhật model khi dữ liệu trôi)?

## Lệnh hay dùng

```bash
uvicorn api.main:app --reload --port 8000  # API + giao diện: http://localhost:8000
docker compose up --build
python -m pytest -q
python -m src.reproduce --quick
git fetch origin && git merge origin/main
```
