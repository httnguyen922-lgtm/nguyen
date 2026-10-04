# 📈 Quantitative Trading Strategy Web App: SMA + OBV Backtester

Ứng dụng web kiểm định định lượng (Backtesting & Quantitative Analysis) chiến lược giao dịch kết hợp giữa **Đường trung bình động (SMA)** và **Khối lượng cân bằng (OBV)** trên nền tảng **Streamlit**, ứng dụng phương pháp phân chia **Train / Test (In-Sample / Out-of-Sample)** và thuật toán tối ưu hóa tự động **Hyperopt (TPE)**.

---

## 🌟 Các Tính Năng Nổi Bật

1. **Phân chia Train / Test chuẩn định lượng**:
   - Giai đoạn **TRAIN (In-Sample: 2014 - 2019)**: Tối ưu hóa bộ tham số SMA và OBV nhằm tránh hiện tượng khớp quá mức (Overfitting).
   - Giai đoạn **TEST (Out-of-Sample: 2020 - 2023)**: Giữ nguyên tham số tối ưu từ Train để kiểm thử tính hiệu quả trong điều kiện thị trường thực tế mới.

2. **So sánh 4 chiến lược trực quan**:
   - **SMA**: Chiến lược giao cắt đường trung bình động (Golden Cross / Death Cross).
   - **OBV**: Chiến lược giao cắt giữa OBV và đường trung bình động của OBV.
   - **SMA + OBV (AND)**: Vào/ra lệnh khi cả SMA và OBV cùng xuất hiện tín hiệu giao cắt trong phiên.
   - **SMA + OBV (OR)**: Vào/ra lệnh linh hoạt khi một trong hai chỉ báo phát tín hiệu.

3. **Tối ưu hóa tham số tự động (Hyperopt)**:
   - Tích hợp thuật toán Bayesian Optimization (Tree-structured Parzen Estimator - TPE) để tìm bộ tham số tốt nhất theo các hàm mục tiêu: *Lợi nhuận (Return %)*, *Tỷ số Sharpe (Sharpe Ratio)* hoặc *Mức sụt giảm tối đa (Max Drawdown)*.

4. **Biểu đồ tương tác chuyên nghiệp (Plotly)**:
   - Biểu đồ nến kỹ thuật tích hợp đường SMA và điểm đánh dấu Mua (Buy) / Bán (Sell).
   - Biểu đồ khối lượng cân bằng OBV kèm đường trung bình động OBV MA.
   - Biểu đồ so sánh đường cong tăng trưởng vốn (Equity Curves) đối chiếu với chiến lược Mua & Nắm giữ (Buy & Hold Benchmark).

5. **Nhật ký giao dịch & Xuất báo cáo (Trade Log & Export)**:
   - Chi tiết từng lệnh: Ngày mở/đóng vị thế, giá vào/ra, lợi nhuận (%) và thời gian nắm giữ.
   - Hỗ trợ tải dữ liệu kết quả dưới dạng file CSV.

---

## 📁 Cấu Trúc Thư Mục

```text
├── ACB.csv               # Dữ liệu giá cổ phiếu ACB mẫu (2014 - 2023)
├── SMA_+_OBV.ipynb       # Jupyter Notebook phân tích và thử nghiệm gốc
├── app.py                # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt      # Danh sách các thư viện phụ thuộc
└── README.md             # Hướng dẫn chi tiết sử dụng và deploy
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cá Nhân (Local)

### 1. Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python phiên bản **3.9**, **3.10** hoặc **3.11**.

Mở Terminal / PowerShell và tạo môi trường ảo:
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở địa chỉ `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đưa Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một tài khoản trên [GitHub](https://github.com) (nếu chưa có).
2. Tạo một repository mới (ví dụ: `sma-obv-trading-strategy`), chọn chế độ **Public**.
3. Khởi tạo Git và đẩy toàn bộ thư mục lên GitHub:
   ```bash
   git init
   git add app.py requirements.txt README.md ACB.csv
   git commit -m "Initial commit: Streamlit trading strategy app"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/sma-obv-trading-strategy.git
   git push -u origin main
   ```

### Bước 2: Triển khai (Deploy) trên Streamlit Community Cloud
1. Truy cập vào [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **New app**.
3. Điền các thông tin triển khai:
   - **Repository**: Chọn repository bạn vừa tải lên (ví dụ: `<tai-khoan-cua-ban>/sma-obv-trading-strategy`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Nhấn nút **Deploy!**.
5. Đợi 1-2 phút để Streamlit Cloud tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy ứng dụng. Sau khi hoàn tất, bạn sẽ nhận được một đường link chia sẻ công khai (ví dụ: `https://sma-obv-strategy.streamlit.app`).

---

## 📊 Kết Quả Kiểm Định Mẫu (Cổ Phiếu ACB 2014 - 2023)

Bộ tham số tối ưu tìm được từ Hyperopt trên tập TRAIN (2014-2019):
- **SMA Ngắn (`ma_short`)**: 90
- **SMA Dài (`ma_long`)**: 240
- **Chu kỳ OBV (`obv_window`)**: 35

### 1. Kết quả trên tập TRAIN (In-Sample: 2014 - 2019)
| Chiến lược | Lợi nhuận (Return) | Sharpe Ratio | Max Drawdown | Số lệnh (Trades) |
| :--- | :---: | :---: | :---: | :---: |
| **SMA** | +196.41% | 0.94 | -41.45% | 2 |
| **OBV** | +248.51% | 1.13 | -24.96% | 31 |
| **SMA + OBV (AND)** | 0.00% | 0.00 | 0.00% | 0 |
| **SMA + OBV (OR)** | **+288.26%** | **1.23** | **-18.53%** | 31 |

### 2. Kết quả trên tập TEST (Out-of-Sample: 2020 - 2023)
| Chiến lược | Lợi nhuận (Return) | Sharpe Ratio | Max Drawdown | Số lệnh (Trades) |
| :--- | :---: | :---: | :---: | :---: |
| **SMA** | +17.24% | 0.58 | -13.51% | 0 |
| **OBV** | +53.63% | 0.54 | -53.14% | 43 |
| **SMA + OBV (AND)** | 0.00% | 0.00 | 0.00% | 0 |
| **SMA + OBV (OR)** | **+72.58%** | **0.66** | **-47.41%** | 43 |

> **Nhận định Định lượng**:
> - Chiến lược **SMA + OBV (OR)** thể hiện sự vượt trội cả về tỷ suất sinh lời và tỷ số Sharpe so với các chiến lược đơn lẻ trên cả hai giai đoạn thị trường.
> - Điều kiện **AND** đòi hỏi 2 chỉ báo cùng lúc xuất hiện điểm cắt trong cùng một phiên giao dịch là quá khắt khe, dẫn đến không phát sinh lệnh giao dịch nào.

---

## 📜 Giấy Phép & Tuyên Bố Miễn Trừ Trách Nhiệm

- Ứng dụng được xây dựng phục vụ mục đích nghiên cứu học thuật và kiểm định định lượng.
- Các kết quả backtest trong quá khứ không bảo đảm hiệu suất sinh lời trong tương lai. Người dùng cần tự chịu trách nhiệm đối với các quyết định giao dịch thực tế trên thị trường tài chính.
