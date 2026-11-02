# MinSight Banking AI — Hệ Thống Hỗ Trợ Đầu Tư Cổ Phiếu Ngân Hàng Bằng Học Sâu Lai Ghép

> **Khóa luận tốt nghiệp:** Nghiên cứu mô hình học sâu lai ghép (CNN-BiLSTM-Attention) đa nhiệm dự báo biến động giá cổ phiếu và xây dựng hệ thống hỗ trợ đầu tư cho nhóm ngân hàng niêm yết Việt Nam.  
> **Sinh viên thực hiện:** Mai Quang Minh  
> **Repository:** [https://github.com/maiquangminh-work/MinSight_App](https://github.com/maiquangminh-work/MinSight_App)

---

## 📌 1. Giới Thiệu & Bối Cảnh Nghiên Cứu

Ngành ngân hàng đóng vai trò huyết mạch trong nền kinh tế Việt Nam, chiếm tỷ trọng vốn hóa và thanh khoản vượt trội trên thị trường chứng khoán (HOSE). Tuy nhiên, chuỗi thời gian giá cổ phiếu ngân hàng mang tính phi tuyến tính cao, bất đối xứng, biến đổi chế độ (regime-shift) và dễ chịu ảnh hưởng bởi các cú sốc thanh khoản, chính sách tiền tệ.

**MinSight Banking AI** là giải pháp kết hợp giữa **Nghiên cứu Định Lượng Học Thuật (Quantitative Deep Learning)** và **Hệ Thống Web Đầu Tư Thực Tiễn (Institutional Financial Terminal)** nhằm:
1. Dự báo đồng thời quỹ đạo giá đa bước ($H=1 \dots 5$ ngày) và xác suất xu hướng (Directional Outperform/Underperform/Neutral).
2. Tối ưu hóa kỳ vọng lợi nhuận ròng (Net Expectancy) và Sharpe Ratio sau khi trừ triệt để chi phí giao dịch (phí 0.15% + thuế).
3. Minh bạch hóa cơ chế dự báo qua khả năng diễn giải nhân quả (**Explainable AI - XAI**) của cơ chế Attention đa không gian con (Multi-Head Temporal Attention).
4. Cung cấp nền tảng Web Terminal chuẩn TradingView với các chỉ báo kỹ thuật chuyên sâu và mô hình xếp hạng cơ bản ngân hàng (CAMEL).

---

## 🏛️ 2. Phạm Vi Nghiên Cứu & Triển Khai

* **Trọng tâm nghiên cứu thực nghiệm học thuật (Big-3 Ngân hàng Nhà nước):**
  * **VCB** (Vietcombank) — Đầu ngành vốn hóa và tỷ lệ CASA.
  * **BID** (BIDV) — Ngân hàng có quy mô tổng tài sản lớn nhất hệ thống.
  * **CTG** (VietinBank) — Ngân hàng trụ cột tài trợ công nghiệp và khách hàng doanh nghiệp.
* **Mở rộng kiểm thử hệ thống (10 Mã Ngân Hàng Niêm Yết):**
  * Nhóm TMCP tư nhân hàng đầu: **MBB, TCB, VPB, ACB, HDB, SHB, VIB**.
  * Chứng minh tính tổng quát hóa và khả năng mở rộng không gian dự báo sang các ngân hàng bán lẻ và ngân hàng số.

---

## 🔬 3. Kiến Trúc Mô Hình & Đóng Góp Học Thuật

### 3.1. Kiến Trúc Lai Ghép Đa Nhiệm: CNN - BiLSTM - Multi-Head Attention (MHTA)
Mô hình giải quyết bài toán kép thông qua kiến trúc nhánh kép (**Multi-Task Learning**):
* **Nhánh 1 (Magnitude Regression):** Dự báo mức biến động giá liên tục theo tỷ lệ lợi nhuận $r_{t+h}$ ($h \in \{1, 2, 3, 4, 5\}$).
* **Nhánh 2 (Direction Classification):** Dự báo xác suất xu hướng phân loại 3 lớp qua phân phối softmax hiệu chỉnh nhiệt độ (*Outperform*, *Neutral*, *Underperform*).

```text
[Chuỗi Thời Gian OHLCV + 30 Features (T=30)]
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
 [1D-CNN Vi Mô]            [1D-CNN Vĩ Mô]
 (Kernel Size=3)           (Kernel Size=5)
       └────────────┬────────────┘
                    ▼
     [Residual Projection + Dropout]
                    ▼
      [Bidirectional LSTM (Bi-LSTM)]  <── Nắm bắt ngữ cảnh đa chiều (quá khứ & tương lai cục bộ)
                    ▼
  [Multi-Head Temporal Attention (MHTA)]
   ├── Head 1: Momentum Breakout (Động lượng)
   ├── Head 2: Liquidity Shock (Thanh khoản & Dòng tiền)
   ├── Head 3: Volatility Regime (Chế độ rủi ro Parkinson/GK)
   └── Head 4: Mean-Reversion (Hồi quy về giá trị thực)
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
 [Nhánh Hồi Quy Biên Độ]   [Nhánh Phân Loại Xu Hướng]
    Dense(64) -> Dense(5)     Dense(32) -> Dense(3, Softmax)
    Dự báo H=1..5 ngày        Xác suất Outperform/Neutral/Underperform
```

### 3.2. Hàm Mất Mát Đa Nhiệm Cải Tiến (MultiTask-VarianceMatching Loss)
Tránh hiện tượng mô hình học sâu tài chính bị hội tụ về giá trị trung bình phẳng (Mean-Collapse):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VM-MSE}} + \lambda_{\text{dir}} \mathcal{L}_{\text{Direction}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$
Trong đó:
$$\mathcal{L}_{\text{VM-MSE}} = \text{MSE}(\hat{y}, y) + \beta \left| \sigma(\hat{y}) - \sigma(y) \right| + \gamma \frac{1}{H} \sum_{h=1}^H \max(0, -\text{sign}(\hat{y}_h) \cdot y_h)$$

### 3.3. Bộ 30 Đặc Trưng Định Lượng Chuyên Sâu Ngành Ngân Hàng
Hệ thống không chỉ dùng giá đóng cửa mà chuẩn hóa 30 chỉ báo theo bản chất tài chính ngân hàng:
1. **Chế độ Biến động (Volatility Regime):** Parkinson Volatility, Garman-Klass Volatility, ATR(14), Bollinger Bandwidth.
2. **Thanh khoản & Áp lực Bán (Flow & Liquidity):** Khối lượng so với MA20, Volume Shock Ratio, Chaikin Money Flow (CMF), Tỷ lệ khối lượng mua chủ động.
3. **Động lượng & Sức mạnh Tương đối (Momentum & RS):** RSI(14), MACD Histogram, ROC(5), ROC(20), Beta tương đối so với VN-Index.
4. **Cơ bản & Rủi ro Tín dụng (CAMEL Proxy):** Tỷ lệ nợ xấu NPL, Tỷ lệ bao phủ nợ xấu LLR, Biên lãi thuần NIM, P/E trailing, P/B, Tỷ lệ sở hữu khối ngoại (Foreign Room).
5. **Tín hiệu Tin tức & Tâm lý (News Sentiment Score):** Tích hợp điểm phân tích cảm xúc từ RSS vĩ mô ngành ngân hàng.

---

## 📊 4. Kết Quả Thực Nghiệm & Báo Cáo Ablation Study

Nghiên cứu thực hiện quy trình kiểm định **Walk-Forward Validation nghiêm ngặt** (Train 756 ngày, Validation 126 ngày, Test 63 ngày, trượt tịnh tiến +63 ngày, embargo 5 ngày tránh rò rỉ dữ liệu):

| Cấu Hình Thử Nghiệm (Ablation) | MAE (Nghìn VNĐ) | Directional Accuracy (%) | Expected Return (Net) | Sharpe Ratio | Max Drawdown |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline: LSTM Đơn Lẻ** | 1.42 | 52.4% | +0.41% | 0.88 | -18.4% |
| **CNN-LSTM Tiêu Chuẩn** | 1.28 | 54.3% | +0.72% | 1.15 | -14.6% |
| **CNN-LSTM-Attention (Seed Đơn)** | 1.16 | 57.1% | +0.95% | 1.34 | -12.1% |
| **CNN-LSTM-Attention Ensemble 5-Seed** | 1.05 | 59.8% | +1.18% | 1.58 | -9.8% |
| **MinSight: MHTA Multi-Task + 30 Features (Hiện tại)** | **0.94** | **62.8%** | **+1.42%** | **1.84** | **-7.2%** |

*Toàn bộ kết quả đối sánh chi tiết lưu tại file `figures/ablation_study_report.md` và `figures/ablation_study_results.json`.*

---

## 💻 5. Tính Năng Hệ Thống Web MinSight Banking AI

Sản phẩm được thiết kế theo phong cách Dark Obsidian Financial Terminal của các nền tảng định lượng hiện đại (TradingView, Bloomberg):

### 5.1. Biểu Đồ Kỹ Thuật Tương Tác (TradingView Lightweight Charts v5)
* **Zooming Khung Thời Gian Chuẩn Xác (`1M`, `3M`, `6M`, `1Y`, `All`):** Giữ trọn vẹn dữ liệu lịch sử trong bộ nhớ, tự động dãn nến phủ kín toàn màn hình, hỗ trợ lăn chuột zoom in/out và kéo sang trái để xem quá khứ mà không bị cắt xén dữ liệu.
* **Chế độ `⛶ Toàn màn hình` / `🗗 Thu nhỏ`:** Cho phép ẩn/hiện cột khuyến nghị định lượng để biểu đồ chiếm 100% độ rộng.
* **Đường Quỹ Đạo Dự Báo AI ($H=1 \dots 5$):** Vẽ đường đứt nét dự báo kèm các mốc mục tiêu giá $T+1$, $T+3$, $T+5$ và Dải biên độ bất định (Uncertainty Confidence Band).

### 5.2. Bàn Phân Tích Định Lượng Chi Tiết 5 Chuyên Mục (Bottom Analytics Dock)
* **Tab 1: 📊 Lịch Sử Tín Hiệu & Hit-Rate:** 4 thẻ KPI Walk-Forward (Hit-Rate 74%, Kỳ vọng ròng +1.38%, Profit Factor 1.85, 90 phiên), thanh trượt lọc độ tin cậy ($\ge 40\%$ đến $\ge 85\%$), bảng nhật ký khớp lệnh giá vào/giá ra.
* **Tab 2: 🧠 Giải Trình Attention (XAI):** Khám phá 4 Attention Heads, biểu đồ nhiệt 30 cột tương tác kèm thông tin phiên, 3 thẻ Top-3 phiên lịch sử dẫn dắt dự báo kèm lý do kinh tế.
* **Tab 3: 🏢 Hồ Sơ & Chỉ Số Cơ Bản (CAMEL):** Vốn hóa sàn, vốn điều lệ, P/E, P/B, ROE (21.6%), ROA (1.85%), NIM (3.18%), NPL nợ xấu (1.16%), kiểm toán Big 4.
* **Tab 4: 📰 Tin Tức & Dòng Tiền:** Tin tức tức thời lọc theo từng mã cổ phiếu, phân tích nhãn sắc thái (Tích cực / Thận trọng / Trung lập).
* **Tab 5: 📋 Bảng Giá Lịch Sử OHLCV:** Toàn bộ dữ liệu giá lịch sử kèm chức năng **Xuất File CSV**.

### 5.3. Cột Khuyến Nghị Định Lượng & Quản Trị Rủi Ro (Quant Decision Panel)
* **Khuyến Nghị Rõ Ràng:** MUA (Khả quan) / BÁN (Kém khả quan) / NẮM GIỮ (Trung lập).
* **Độ Tin Cậy Calibrated:** Điểm số qua Temperature Scaling và Hybrid Confidence Gate.
* **Classic Pivot Points:** Tính toán tức thời các mức cản kỹ thuật R2, R1, Điểm xoay PP, Hỗ trợ S1, S2.
* **Quản Trị Rủi Ro ATR(14):** Đề xuất mức Cắt lỗ (Stop Loss) động và Mục tiêu chốt lời (Take Profit) theo tỷ lệ Risk/Reward.

---

## 🛠️ 6. Công Nghệ Sử Dụng

* **Deep Learning Engine:** Python 3.10+, TensorFlow 2.15 / Keras, Scikit-learn, NumPy, Pandas.
* **Dữ Liệu Tài Chính:** vnstock, tích hợp crawler SQLite và pipeline trích xuất 30 chỉ báo.
* **Backend API:** FastAPI, Uvicorn, SQLAlchemy, Pydantic v2.
* **Frontend Web:** React 18, Vite, React Router v6, Lightweight Charts v5.1.0, Lucide Icons.

---

## 🚀 7. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### Yêu cầu tiên quyết
* Python 3.10 trở lên
* Node.js 18+ và npm

### Bước 1: Khởi động Backend API
```bash
# Di chuyển vào thư mục dự án
cd MinSight

# Tạo và kích hoạt môi trường ảo Python
python -m venv venv
venv\Scripts\activate       # Trên Windows
# source venv/bin/activate  # Trên macOS/Linux

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Khởi chạy server FastAPI (mặc định tại http://localhost:8000)
uvicorn api:app --reload --port 8000
```

### Bước 2: Khởi động Frontend Web
Mở một cửa sổ Terminal mới:
```bash
# Di chuyển vào thư mục frontend
cd MinSight/frontend

# Cài đặt gói phụ thuộc
npm install

# Khởi chạy dev server (mặc định tại http://localhost:5173)
npm run dev
```

Mở trình duyệt truy cập: **`http://localhost:5173/chart/VCB`** để bắt đầu trải nghiệm.

---

## 📈 8. Tái Lập Thực Nghiệm Học Thuật

### 1. Chạy Walk-Forward Validation Ensemble 5-Seed:
```bash
python src/model/walk_forward.py --tickers VCB BID CTG --seeds 42 123 456 789 2024
```

### 2. Kiểm Định Thống Kê (Diebold-Mariano & Pesaran-Timmermann):
```bash
python src/model/statistical_tests.py
```

### 3. Xuất Báo Cáo Phụ Lục Khóa Luận (Thesis Summary):
```bash
python src/model/thesis_summary.py
```

---

## 📄 9. Bản Quyền & Giấy Phép
Dự án được xây dựng và phục vụ mục đích nghiên cứu học thuật trong khuôn khổ Khóa luận tốt nghiệp chuyên ngành Hệ thống Thông tin / Khoa học Dữ liệu.
Mọi phân tích dự báo và khuyến nghị kỹ thuật là kết quả từ mô hình định lượng máy học, không cấu thành lời mời gọi đầu tư tài chính bắt buộc.
