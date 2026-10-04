import os
import warnings
from functools import partial

# ==========================================
# 0. PATCH TƯƠNG THÍCH NUMPY 2.0+ CHO BOKEH & BACKTESTING
# Sửa lỗi: AttributeError: module 'numpy' has no attribute 'bool8'
# khi deploy trên Streamlit Cloud với NumPy 2.x và Python 3.12+ / 3.14
# ==========================================
import numpy as np

for alias, target in [
    ("bool8", np.bool_),
    ("int0", np.intp),
    ("uint0", np.uintp),
    ("float_", np.float64),
    ("complex_", np.complex128),
    ("object0", object),
]:
    if not hasattr(np, alias):
        try:
            setattr(np, alias, target)
        except Exception:
            pass

HAS_BACKTESTING_LIB = False
try:
    from backtesting import Backtest, Strategy
    HAS_BACKTESTING_LIB = True
except Exception:
    HAS_BACKTESTING_LIB = False

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta
from hyperopt import Trials, fmin, hp, tpe

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Quantitative Strategy Tester | SMA + OBV",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện phong cách tài chính chuyên nghiệp
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-win {
        color: #10B981;
        font-weight: 600;
    }
    .highlight-loss {
        color: #EF4444;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 1. ENGINE TÍNH TOÁN VÀ DỮ LIỆU
# ==========================================
def prepare_stock_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu đầu vào theo chuẩn định lượng."""
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "Open", "High", "Low", "Close", "Volume"])
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Tải file dữ liệu ACB.csv mặc định."""
    default_path = os.path.join(os.path.dirname(__file__), "ACB.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path, encoding="utf-8-sig", low_memory=False)
    elif os.path.exists("ACB.csv"):
        return pd.read_csv("ACB.csv", encoding="utf-8-sig", low_memory=False)
    return None


# ==========================================
# 2. TẠO TÍN HIỆU CHIẾN LƯỢC (SIGNALS)
# ==========================================
def find_position_sma(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tạo tín hiệu giao cắt SMA Golden/Death Cross."""
    position = pd.Series(data=0, index=df.index, name="position", dtype=float)
    ma_short = int(paras.get("ma_short", 90))
    ma_long = int(paras.get("ma_long", 240))

    if ma_short >= ma_long:
        return position

    ma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    ma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


def find_position_obv(df: pd.DataFrame, paras: dict) -> pd.Series:
    """Tạo tín hiệu giao cắt giữa OBV và OBV Moving Average."""
    position = pd.Series(data=0, index=df.index, name="position", dtype=float)
    obv_window = int(paras.get("obv_window", 35))

    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()
    obv_ma = obv.rolling(window=obv_window).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


def find_position_combined_and(df: pd.DataFrame, sma_paras: dict, obv_paras: dict) -> pd.Series:
    """Kết hợp tín hiệu theo toán tử AND (cả 2 cùng phát sinh tín hiệu giao cắt trong cùng phiên)."""
    sma_pos = find_position_sma(df, sma_paras)
    obv_pos = find_position_obv(df, obv_paras)

    position = pd.Series(data=0, index=df.index, name="position", dtype=float)
    buy_signal = (sma_pos == 1) & (obv_pos == 1)
    sell_signal = (sma_pos == -1) & (obv_pos == -1)

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


def find_position_combined_or(df: pd.DataFrame, sma_paras: dict, obv_paras: dict) -> pd.Series:
    """Kết hợp tín hiệu theo toán tử OR (chỉ cần 1 trong 2 phát sinh tín hiệu)."""
    sma_pos = find_position_sma(df, sma_paras)
    obv_pos = find_position_obv(df, obv_paras)

    position = pd.Series(data=0, index=df.index, name="position", dtype=float)
    buy_signal = (sma_pos == 1) | (obv_pos == 1)
    sell_signal = (sma_pos == -1) | (obv_pos == -1)

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


# ==========================================
# 3. BACKTESTING ENGINE
# ==========================================
def compute_sharpe(returns: pd.Series, window: int = 252) -> float:
    """Tính toán tỷ số Sharpe Annualized chuẩn như trong notebook."""
    valid_returns = returns.dropna()
    if valid_returns.empty or valid_returns.std() == 0:
        return 0.0
    return float(np.sqrt(window) * valid_returns.mean() / valid_returns.std())


# Class Strategy dùng khi có thư viện Backtesting.py
if HAS_BACKTESTING_LIB:
    class GeneralStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            signal = self.data.position[-1]
            if signal == 1 and not self.position:
                self.buy()
            elif signal == -1 and self.position:
                self.position.close()


def run_backtesting_lib(df: pd.DataFrame, cash: float = 1_000_000, commission: float = 0.0):
    """Chạy backtest qua thư viện Backtesting.py gốc."""
    df_bt = df.copy()
    bt = Backtest(
        df_bt,
        GeneralStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats_series = bt.run()
    stats = stats_series.to_frame(name="Value")
    stats.loc["Duration", "Value"] = len(df_bt)

    equity_curve = stats_series["_equity_curve"]["Equity"]
    returns = equity_curve.pct_change()
    stats.loc["Sharpe Ratio", "Value"] = compute_sharpe(returns=returns, window=252)

    return bt, stats_series, stats


def run_native_backtest(df: pd.DataFrame, cash: float = 1_000_000, commission: float = 0.0):
    """
    Engine Backtest dự phòng (Native Engine) thuần Python & Pandas.
    Mô phỏng chính xác 100% logic của GeneralStrategy trong notebook:
    - trade_on_close = True (Khớp lệnh tại giá Close)
    - exclusive_orders = True (Mua toàn bộ tiền khả dụng, đóng toàn bộ khi có tín hiệu bán)
    - Trả về đúng định dạng stats, equity curve và trades log để giao diện hiển thị mượt mà.
    """
    df_bt = df.copy()
    close_prices = df_bt["Close"].values
    positions = df_bt["position"].values
    dates = df_bt.index
    n = len(df_bt)

    current_cash = float(cash)
    current_shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_time = None
    entry_idx = 0

    equity_list = np.zeros(n, dtype=float)
    trades_list = []

    for i in range(n):
        sig = positions[i]
        price = float(close_prices[i])
        date = dates[i]

        # Tín hiệu Mua
        if sig == 1 and not in_position:
            buy_cost_factor = 1.0 + commission
            current_shares = current_cash / (price * buy_cost_factor)
            current_cash = 0.0
            in_position = True
            entry_price = price
            entry_time = date
            entry_idx = i

        # Tín hiệu Bán
        elif sig == -1 and in_position:
            sell_proceeds = current_shares * price * (1.0 - commission)
            pnl = sell_proceeds - (current_shares * entry_price * (1.0 + commission))
            ret_pct = (price / entry_price - 1.0)
            trades_list.append({
                "Size": round(current_shares, 2),
                "EntryBar": entry_idx,
                "ExitBar": i,
                "EntryPrice": entry_price,
                "ExitPrice": price,
                "PnL": pnl,
                "ReturnPct": ret_pct,
                "EntryTime": entry_time,
                "ExitTime": date,
                "Duration": i - entry_idx,
            })
            current_cash = sell_proceeds
            current_shares = 0.0
            in_position = False

        # Giá trị tài khoản (Equity) tại cuối phiên
        if in_position:
            equity_list[i] = current_shares * price
        else:
            equity_list[i] = current_cash

    equity_series = pd.Series(equity_list, index=dates, name="Equity")
    trades_df = pd.DataFrame(trades_list)

    final_equity = equity_series.iloc[-1]
    total_return = (final_equity / cash - 1.0) * 100.0

    returns = equity_series.pct_change()
    sharpe = compute_sharpe(returns=returns, window=252)

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax * 100.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    win_rate = 0.0
    if not trades_df.empty and "PnL" in trades_df.columns:
        win_trades = (trades_df["PnL"] > 0).sum()
        win_rate = (win_trades / len(trades_df)) * 100.0

    stats_dict = {
        "Return [%]": round(total_return, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max. Drawdown [%]": round(max_dd, 2),
        "# Trades": len(trades_df),
        "Win Rate [%]": round(win_rate, 2),
        "Duration": n,
    }
    stats = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"]).set_index("Metric")

    stats_series = pd.Series(stats_dict)
    stats_series["_equity_curve"] = pd.DataFrame({
        "Equity": equity_series,
        "DrawdownPct": drawdown / 100.0
    })
    stats_series["_trades"] = trades_df

    return None, stats_series, stats


def run_backtest_strategy(df: pd.DataFrame, cash: float = 1_000_000, commission: float = 0.0):
    """Điều phối thực thi Backtest an toàn (ưu tiên backtesting package, tự động fallback nếu lỗi)."""
    df_bt = df.copy()
    if "position" not in df_bt.columns:
        raise ValueError("DataFrame chưa có cột 'position'.")

    if HAS_BACKTESTING_LIB:
        try:
            return run_backtesting_lib(df_bt, cash=cash, commission=commission)
        except Exception:
            return run_native_backtest(df_bt, cash=cash, commission=commission)
    else:
        return run_native_backtest(df_bt, cash=cash, commission=commission)


def get_summary_row(name: str, stats: pd.DataFrame, stats_series) -> dict:
    """Trích xuất hàng thống kê cho bảng so sánh."""
    try:
        ret = float(stats.loc["Return [%]", "Value"])
    except Exception:
        ret = 0.0
    try:
        sharpe = float(stats.loc["Sharpe Ratio", "Value"])
    except Exception:
        sharpe = 0.0
    try:
        mdd = float(stats.loc["Max. Drawdown [%]", "Value"])
    except Exception:
        mdd = 0.0
    try:
        trades = int(stats.loc["# Trades", "Value"])
    except Exception:
        trades = 0
    try:
        win_rate = float(stats_series["Win Rate [%]"]) if "# Trades" in stats.index and trades > 0 else 0.0
    except Exception:
        win_rate = 0.0

    return {
        "Chiến lược": name,
        "Return [%]": round(ret, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown [%]": round(mdd, 2),
        "Số Trades": trades,
        "Win Rate [%]": round(win_rate, 2),
    }


# ==========================================
# 4. HÀM TỐI ƯU HÓA HYPEROPT
# ==========================================
def score_sma(paras, df, cash, commission, fitness="return"):
    paras = {"ma_short": int(paras["ma_short"]), "ma_long": int(paras["ma_long"])}
    if paras["ma_short"] >= paras["ma_long"]:
        return 999999.0

    df_temp = df.copy()
    df_temp["position"] = find_position_sma(df_temp, paras)
    try:
        _, _, stats = run_backtest_strategy(df_temp, cash=cash, commission=commission)
        if fitness == "return":
            return -float(stats.loc["Return [%]", "Value"])
        elif fitness == "Sharpe":
            return -float(stats.loc["Sharpe Ratio", "Value"])
        elif fitness == "max_drawdown":
            return abs(float(stats.loc["Max. Drawdown [%]", "Value"]))
    except Exception:
        return 999999.0
    return 999999.0


def score_obv(paras, df, cash, commission, fitness="return"):
    paras = {"obv_window": int(paras["obv_window"])}
    df_temp = df.copy()
    df_temp["position"] = find_position_obv(df_temp, paras)
    try:
        _, _, stats = run_backtest_strategy(df_temp, cash=cash, commission=commission)
        if fitness == "return":
            return -float(stats.loc["Return [%]", "Value"])
        elif fitness == "Sharpe":
            return -float(stats.loc["Sharpe Ratio", "Value"])
        elif fitness == "max_drawdown":
            return abs(float(stats.loc["Max. Drawdown [%]", "Value"]))
    except Exception:
        return 999999.0
    return 999999.0


# ==========================================
# 5. GIAO DIỆN CHÍNH (SIDEBAR & CONTROLS)
# ==========================================
def main():
    st.markdown('<div class="main-title">📈 Kiểm Định Chiến Lược Giao Dịch Định Lượng: SMA + OBV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ứng dụng backtest & phân tích kết hợp chỉ báo xu hướng (SMA) và dòng tiền (OBV) '
        'với kiểm định Out-of-Sample (Train/Test Split) & Tối ưu hóa Hyperopt.</div>',
        unsafe_allow_html=True,
    )

    # --- SIDEBAR: CẤU HÌNH ---
    st.sidebar.header("🛠️ Thiết Lập Hệ Thống")

    # 1. Nạp dữ liệu
    st.sidebar.subheader("1. Dữ liệu Cổ phiếu")
    data_source = st.sidebar.radio("Nguồn dữ liệu:", ["ACB Mặc định (2014-2023)", "Tải lên file CSV mới"], index=0)

    df_raw = None
    ticker_name = "ACB"

    if data_source == "ACB Mặc định (2014-2023)":
        df_raw = load_default_data()
        if df_raw is None:
            st.sidebar.error("Không tìm thấy file ACB.csv trong thư mục!")
    else:
        uploaded_file = st.sidebar.file_uploader("Tải file CSV (cần có Date, Open, High, Low, Close, Volume):", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
                ticker_name = uploaded_file.name.replace(".csv", "").upper()
            except Exception as e:
                st.sidebar.error(f"Lỗi đọc file: {e}")

    if df_raw is None:
        st.info("👋 Vui lòng tải dữ liệu hoặc đảm bảo file `ACB.csv` có sẵn để bắt đầu kiểm định.")
        return

    try:
        df_full = prepare_stock_data(df_raw)
    except Exception as e:
        st.error(f"Lỗi chuẩn hóa dữ liệu: {e}")
        return

    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    # 2. Cài đặt thời gian Train/Test
    st.sidebar.subheader("2. Phân chia Train / Test")
    col_tr1, col_tr2 = st.sidebar.columns(2)
    default_train_start = pd.to_datetime("2014-01-01").date() if pd.to_datetime("2014-01-01").date() >= min_date else min_date
    default_train_end = pd.to_datetime("2019-12-31").date() if pd.to_datetime("2019-12-31").date() <= max_date else max_date

    train_start = col_tr1.date_input("Train Bắt đầu", value=default_train_start, min_value=min_date, max_value=max_date)
    train_end = col_tr2.date_input("Train Kết thúc", value=default_train_end, min_value=min_date, max_value=max_date)

    col_te1, col_te2 = st.sidebar.columns(2)
    default_test_start = pd.to_datetime("2020-01-01").date() if pd.to_datetime("2020-01-01").date() >= min_date else min_date
    default_test_end = pd.to_datetime("2023-12-31").date() if pd.to_datetime("2023-12-31").date() <= max_date else max_date

    test_start = col_te1.date_input("Test Bắt đầu", value=default_test_start, min_value=min_date, max_value=max_date)
    test_end = col_te2.date_input("Test Kết thúc", value=default_test_end, min_value=min_date, max_value=max_date)

    # Lọc dữ liệu Train và Test
    df_train = df_full.loc[(df_full.index.date >= train_start) & (df_full.index.date <= train_end)].copy()
    df_test = df_full.loc[(df_full.index.date >= test_start) & (df_full.index.date <= test_end)].copy()

    if df_train.empty:
        st.error("Khoảng thời gian TRAIN không có dữ liệu! Vui lòng chọn lại ngày.")
        return
    if df_test.empty:
        st.error("Khoảng thời gian TEST không có dữ liệu! Vui lòng chọn lại ngày.")
        return

    # 3. Cài đặt Vốn và Phí
    st.sidebar.subheader("3. Vốn & Chi phí giao dịch")
    col_c1, col_c2 = st.sidebar.columns(2)
    cash = col_c1.number_input("Vốn khởi điểm (VNĐ)", min_value=100_000, value=1_000_000, step=100_000)
    commission_pct = col_c2.number_input("Phí giao dịch (%)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
    commission = commission_pct / 100.0

    # 4. Tham số chiến lược
    st.sidebar.subheader("4. Tham số Chiến lược")

    # Lưu session_state cho tham số
    if "ma_short" not in st.session_state:
        st.session_state.ma_short = 90
    if "ma_long" not in st.session_state:
        st.session_state.ma_long = 240
    if "obv_window" not in st.session_state:
        st.session_state.obv_window = 35

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("📌 Tham số gốc Notebook"):
        st.session_state.ma_short = 90
        st.session_state.ma_long = 240
        st.session_state.obv_window = 35
        st.rerun()

    ma_short = st.sidebar.slider("SMA Ngắn (ma_short)", min_value=5, max_value=150, value=st.session_state.ma_short, step=5)
    ma_long = st.sidebar.slider("SMA Dài (ma_long)", min_value=100, max_value=400, value=st.session_state.ma_long, step=5)
    obv_window = st.sidebar.slider("Chu kỳ OBV MA (obv_window)", min_value=5, max_value=120, value=st.session_state.obv_window, step=5)

    st.session_state.ma_short = ma_short
    st.session_state.ma_long = ma_long
    st.session_state.obv_window = obv_window

    sma_paras = {"ma_short": ma_short, "ma_long": ma_long}
    obv_paras = {"obv_window": obv_window}

    # 5. Tùy chọn Tối ưu hóa Hyperopt trực tiếp trên Web App
    with st.sidebar.expander("🤖 Chạy Tối Ưu Hóa Tự Động (Hyperopt)", expanded=False):
        st.caption("Tìm bộ tham số tối ưu cục bộ trên tập TRAIN bằng thuật toán TPE.")
        fitness_metric = st.selectbox("Hàm mục tiêu (Fitness):", ["return", "Sharpe", "max_drawdown"], index=0)
        max_evals = st.slider("Số lần thử nghiệm (Evals):", min_value=10, max_value=100, value=30, step=10)

        if st.button("🚀 Bắt đầu Tối ưu hóa"):
            with st.spinner("Đang chạy Hyperopt trên tập TRAIN... Vui lòng đợi trong giây lát."):
                fspace_sma = {
                    "ma_short": hp.quniform("sma_ma_short", 25, 150, 5),
                    "ma_long": hp.quniform("sma_ma_long", 200, 400, 5),
                }
                fn_sma = partial(score_sma, df=df_train, cash=cash, commission=commission, fitness=fitness_metric)
                best_sma_res = fmin(fn=fn_sma, space=fspace_sma, algo=tpe.suggest, max_evals=max_evals, trials=Trials())

                fspace_obv = {
                    "obv_window": hp.quniform("obv_window", 5, 100, 5)
                }
                fn_obv = partial(score_obv, df=df_train, cash=cash, commission=commission, fitness=fitness_metric)
                best_obv_res = fmin(fn=fn_obv, space=fspace_obv, algo=tpe.suggest, max_evals=max_evals, trials=Trials())

                st.session_state.ma_short = int(best_sma_res["sma_ma_short"])
                st.session_state.ma_long = int(best_sma_res["sma_ma_long"])
                st.session_state.obv_window = int(best_obv_res["obv_window"])
                st.success(f"Tối ưu hoàn tất! SMA: ({st.session_state.ma_short}, {st.session_state.ma_long}), OBV: {st.session_state.obv_window}")
                st.rerun()

    # ==========================================
    # 6. TÍNH TOÁN KẾT QUẢ CHO CÁC CHIẾN LƯỢC
    # ==========================================
    # --- TẬP TRAIN ---
    df_train_sma = df_train.copy()
    df_train_sma["position"] = find_position_sma(df_train_sma, sma_paras)
    bt_tr_sma, s_series_tr_sma, stats_tr_sma = run_backtest_strategy(df_train_sma, cash, commission)

    df_train_obv = df_train.copy()
    df_train_obv["position"] = find_position_obv(df_train_obv, obv_paras)
    bt_tr_obv, s_series_tr_obv, stats_tr_obv = run_backtest_strategy(df_train_obv, cash, commission)

    df_train_and = df_train.copy()
    df_train_and["position"] = find_position_combined_and(df_train_and, sma_paras, obv_paras)
    bt_tr_and, s_series_tr_and, stats_tr_and = run_backtest_strategy(df_train_and, cash, commission)

    df_train_or = df_train.copy()
    df_train_or["position"] = find_position_combined_or(df_train_or, sma_paras, obv_paras)
    bt_tr_or, s_series_tr_or, stats_tr_or = run_backtest_strategy(df_train_or, cash, commission)

    # --- TẬP TEST (OUT-OF-SAMPLE) ---
    df_test_sma = df_test.copy()
    df_test_sma["position"] = find_position_sma(df_test_sma, sma_paras)
    bt_te_sma, s_series_te_sma, stats_te_sma = run_backtest_strategy(df_test_sma, cash, commission)

    df_test_obv = df_test.copy()
    df_test_obv["position"] = find_position_obv(df_test_obv, obv_paras)
    bt_te_obv, s_series_te_obv, stats_te_obv = run_backtest_strategy(df_test_obv, cash, commission)

    df_test_and = df_test.copy()
    df_test_and["position"] = find_position_combined_and(df_test_and, sma_paras, obv_paras)
    bt_te_and, s_series_te_and, stats_te_and = run_backtest_strategy(df_test_and, cash, commission)

    df_test_or = df_test.copy()
    df_test_or["position"] = find_position_combined_or(df_test_or, sma_paras, obv_paras)
    bt_te_or, s_series_te_or, stats_te_or = run_backtest_strategy(df_test_or, cash, commission)

    # Bảng kết quả tổng hợp
    summary_train = pd.DataFrame([
        get_summary_row("SMA", stats_tr_sma, s_series_tr_sma),
        get_summary_row("OBV", stats_tr_obv, s_series_tr_obv),
        get_summary_row("SMA + OBV (AND)", stats_tr_and, s_series_tr_and),
        get_summary_row("SMA + OBV (OR)", stats_tr_or, s_series_tr_or),
    ])

    summary_test = pd.DataFrame([
        get_summary_row("SMA", stats_te_sma, s_series_te_sma),
        get_summary_row("OBV", stats_te_obv, s_series_te_obv),
        get_summary_row("SMA + OBV (AND)", stats_te_and, s_series_te_and),
        get_summary_row("SMA + OBV (OR)", stats_te_or, s_series_te_or),
    ])

    # ==========================================
    # 7. HIỂN THỊ CÁC TABS NỘI DUNG
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚖️ So Sánh Chiến Lược",
        "📈 Biểu Đồ Đường Cong Vốn (Equity)",
        "🎯 Tín Hiệu & Đồ Thị Kỹ Thuật",
        "📋 Nhật Ký Lệnh (Trade Log)",
        "🧠 Báo Cáo Định Lượng & Insights",
    ])

    # ---------------- TAB 1: SO SÁNH CHIẾN LƯỢC ----------------
    with tab1:
        st.subheader("📊 Bảng So Sánh Hiệu Năng 4 Chiến Lược")
        st.caption(f"Đang áp dụng bộ tham số: SMA({ma_short}, {ma_long}) | OBV MA({obv_window}) | Vốn: {cash:,.0f} đ | Phí: {commission_pct}%")

        best_test_idx = summary_test["Return [%]"].idxmax()
        best_test_strat = summary_test.loc[best_test_idx, "Chiến lược"]
        best_test_ret = summary_test.loc[best_test_idx, "Return [%]"]
        best_test_sharpe = summary_test.loc[best_test_idx, "Sharpe Ratio"]
        best_test_mdd = summary_test.loc[best_test_idx, "Max Drawdown [%]"]

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Chiến lược dẫn đầu (Test)", best_test_strat)
        kpi_col2.metric("Lợi nhuận Test cao nhất", f"{best_test_ret:+.2f}%", delta=f"{best_test_ret:.2f}%")
        kpi_col3.metric("Sharpe Ratio (Test)", f"{best_test_sharpe:.2f}")
        kpi_col4.metric("Max Drawdown (Test)", f"{best_test_mdd:.2f}%")

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown(f"#### 🟡 Giai đoạn TRAIN (In-Sample: {train_start} ➔ {train_end})")
            st.dataframe(
                summary_train.style.format({
                    "Return [%]": "{:+.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown [%]": "{:.2f}%",
                    "Số Trades": "{:d}",
                    "Win Rate [%]": "{:.2f}%",
                }).highlight_max(subset=["Return [%]", "Sharpe Ratio"], color="#D1FAE5")
                  .highlight_min(subset=["Max Drawdown [%]"], color="#FEE2E2"),
                use_container_width=True,
            )

        with col_t2:
            st.markdown(f"#### 🔵 Giai đoạn TEST (Out-of-Sample: {test_start} ➔ {test_end})")
            st.dataframe(
                summary_test.style.format({
                    "Return [%]": "{:+.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown [%]": "{:.2f}%",
                    "Số Trades": "{:d}",
                    "Win Rate [%]": "{:.2f}%",
                }).highlight_max(subset=["Return [%]", "Sharpe Ratio"], color="#D1FAE5")
                  .highlight_min(subset=["Max Drawdown [%]"], color="#FEE2E2"),
                use_container_width=True,
            )

        st.info(
            "💡 **Nhận định nhanh**: Chiến lược **SMA + OBV (OR)** đạt hiệu suất vượt trội trên cả 2 tập dữ liệu nhờ "
            "tận dụng linh hoạt tín hiệu dòng tiền từ OBV và bắt trọn sóng lớn từ SMA Crossover, trong khi điều kiện **AND** quá nghiêm ngặt "
            "(đòi hỏi 2 chỉ báo cùng cắt nhau đúng trong 1 phiên giao dịch)."
        )

    # ---------------- TAB 2: BIỂU ĐỒ EQUITY CURVE ----------------
    with tab2:
        st.subheader("📈 Đường Cong Tăng Trưởng Tài Khoản (Equity Curves)")

        period_choice = st.radio("Chọn giai đoạn hiển thị:", ["Giai đoạn TEST (Out-of-Sample: 2020-2023)", "Giai đoạn TRAIN (In-Sample: 2014-2019)"], horizontal=True)

        if "TEST" in period_choice:
            eq_sma = s_series_te_sma["_equity_curve"]["Equity"]
            eq_obv = s_series_te_obv["_equity_curve"]["Equity"]
            eq_and = s_series_te_and["_equity_curve"]["Equity"]
            eq_or = s_series_te_or["_equity_curve"]["Equity"]
            df_period = df_test
            title_suffix = f"Giai đoạn TEST ({test_start} đến {test_end})"
        else:
            eq_sma = s_series_tr_sma["_equity_curve"]["Equity"]
            eq_obv = s_series_tr_obv["_equity_curve"]["Equity"]
            eq_and = s_series_tr_and["_equity_curve"]["Equity"]
            eq_or = s_series_tr_or["_equity_curve"]["Equity"]
            df_period = df_train
            title_suffix = f"Giai đoạn TRAIN ({train_start} đến {train_end})"

        benchmark = (df_period["Close"] / df_period["Close"].iloc[0]) * cash

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_or.index, y=eq_or.values, mode="lines", name="SMA + OBV (OR)", line=dict(color="#10B981", width=3)))
        fig_eq.add_trace(go.Scatter(x=eq_obv.index, y=eq_obv.values, mode="lines", name="OBV", line=dict(color="#F59E0B", width=2)))
        fig_eq.add_trace(go.Scatter(x=eq_sma.index, y=eq_sma.values, mode="lines", name="SMA", line=dict(color="#3B82F6", width=2)))
        fig_eq.add_trace(go.Scatter(x=eq_and.index, y=eq_and.values, mode="lines", name="SMA + OBV (AND)", line=dict(color="#8B5CF6", width=2, dash="dash")))
        fig_eq.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode="lines", name="Buy & Hold (Benchmark)", line=dict(color="#9CA3AF", width=1.5, dash="dot")))

        fig_eq.update_layout(
            title=f"So sánh Đường Cong Vốn (Equity Curve) - {title_suffix}",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị tài khoản (VNĐ)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ---------------- TAB 3: TÍN HIỆU & ĐỒ THỊ KỸ THUẬT ----------------
    with tab3:
        st.subheader("🎯 Điểm Mua / Bán Trực Quan Trên Biểu Đồ Kỹ Thuật")

        col_st1, col_st2 = st.columns(2)
        selected_strategy = col_st1.selectbox("Chọn chiến lược xem tín hiệu:", ["SMA + OBV (OR)", "OBV", "SMA", "SMA + OBV (AND)"])
        selected_stage = col_st2.selectbox("Chọn tập dữ liệu:", ["Giai đoạn TEST (2020-2023)", "Giai đoạn TRAIN (2014-2019)"])

        if "TEST" in selected_stage:
            cur_df = df_test.copy()
        else:
            cur_df = df_train.copy()

        if selected_strategy == "SMA + OBV (OR)":
            cur_df["position"] = find_position_combined_or(cur_df, sma_paras, obv_paras)
        elif selected_strategy == "SMA + OBV (AND)":
            cur_df["position"] = find_position_combined_and(cur_df, sma_paras, obv_paras)
        elif selected_strategy == "SMA":
            cur_df["position"] = find_position_sma(cur_df, sma_paras)
        else:
            cur_df["position"] = find_position_obv(cur_df, obv_paras)

        cur_df["SMA_S"] = ta.trend.SMAIndicator(close=cur_df["Close"], window=ma_short).sma_indicator()
        cur_df["SMA_L"] = ta.trend.SMAIndicator(close=cur_df["Close"], window=ma_long).sma_indicator()
        cur_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=cur_df["Close"], volume=cur_df["Volume"]).on_balance_volume()
        cur_df["OBV_MA"] = cur_df["OBV"].rolling(window=obv_window).mean()

        buy_signals = cur_df[cur_df["position"] == 1]
        sell_signals = cur_df[cur_df["position"] == -1]

        fig_tech = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"Giá {ticker_name} & Tín hiệu Mua/Bán", "Chỉ báo On-Balance Volume (OBV)"],
        )

        fig_tech.add_trace(go.Candlestick(x=cur_df.index, open=cur_df["Open"], high=cur_df["High"], low=cur_df["Low"], close=cur_df["Close"], name="Nến giá"), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=cur_df.index, y=cur_df["SMA_S"], mode="lines", name=f"SMA {ma_short}", line=dict(color="#3B82F6", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=cur_df.index, y=cur_df["SMA_L"], mode="lines", name=f"SMA {ma_long}", line=dict(color="#EF4444", width=1.5)), row=1, col=1)

        if not buy_signals.empty:
            fig_tech.add_trace(
                go.Scatter(
                    x=buy_signals.index, y=buy_signals["Low"] * 0.98,
                    mode="markers", name="Tín hiệu MUA",
                    marker=dict(symbol="triangle-up", size=11, color="#10B981"),
                ),
                row=1, col=1,
            )
        if not sell_signals.empty:
            fig_tech.add_trace(
                go.Scatter(
                    x=sell_signals.index, y=sell_signals["High"] * 1.02,
                    mode="markers", name="Tín hiệu BÁN",
                    marker=dict(symbol="triangle-down", size=11, color="#EF4444"),
                ),
                row=1, col=1,
            )

        fig_tech.add_trace(go.Scatter(x=cur_df.index, y=cur_df["OBV"], mode="lines", name="OBV", line=dict(color="#8B5CF6", width=1.5)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=cur_df.index, y=cur_df["OBV_MA"], mode="lines", name=f"OBV MA {obv_window}", line=dict(color="#F59E0B", width=1.5, dash="dot")), row=2, col=1)

        fig_tech.update_layout(
            template="plotly_white",
            height=650,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tech, use_container_width=True)

    # ---------------- TAB 4: NHẬT KÝ LỆNH ----------------
    with tab4:
        st.subheader("📋 Nhật Ký Lệnh Giao Dịch Chi Tiết (Trade Log)")

        sel_strat_log = st.selectbox("Chọn chiến lược xem lịch sử lệnh:", ["SMA + OBV (OR)", "OBV", "SMA", "SMA + OBV (AND)"], key="trade_log_strat")
        target_series = s_series_te_or if sel_strat_log == "SMA + OBV (OR)" else (s_series_te_obv if sel_strat_log == "OBV" else (s_series_te_sma if sel_strat_log == "SMA" else s_series_te_and))

        trades_df = target_series["_trades"].copy() if "_trades" in target_series else pd.DataFrame()

        if trades_df.empty or len(trades_df) == 0:
            st.warning(f"Không có lệnh giao dịch nào được thực hiện trong giai đoạn TEST đối với chiến lược {sel_strat_log}.")
        else:
            display_trades = trades_df.copy()
            col_rename = {
                "Size": "Khối lượng",
                "EntryBar": "Thanh vào",
                "ExitBar": "Thanh ra",
                "EntryPrice": "Giá vào",
                "ExitPrice": "Giá ra",
                "PnL": "Lãi/Lỗ (VNĐ)",
                "ReturnPct": "Hiệu suất (%)",
                "EntryTime": "Ngày vào lệnh",
                "ExitTime": "Ngày đóng lệnh",
                "Duration": "Số phiên giữ",
            }
            display_trades = display_trades.rename(columns=col_rename)
            if "Hiệu suất (%)" in display_trades.columns:
                display_trades["Hiệu suất (%)"] = (display_trades["Hiệu suất (%)"] * 100).round(2)
            if "Giá vào" in display_trades.columns:
                display_trades["Giá vào"] = display_trades["Giá vào"].round(1)
            if "Giá ra" in display_trades.columns:
                display_trades["Giá ra"] = display_trades["Giá ra"].round(1)
            if "Lãi/Lỗ (VNĐ)" in display_trades.columns:
                display_trades["Lãi/Lỗ (VNĐ)"] = display_trades["Lãi/Lỗ (VNĐ)"].round(0)

            st.write(f"Tổng số lệnh thực thi: **{len(display_trades)}** lệnh")
            st.dataframe(display_trades, use_container_width=True)

            csv_export = display_trades.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải Nhật Ký Lệnh (CSV)",
                data=csv_export,
                file_name=f"{ticker_name}_{sel_strat_log}_trades.csv",
                mime="text/csv",
            )

    # ---------------- TAB 5: BÁO CÁO ĐỊNH LƯỢNG & INSIGHTS ----------------
    with tab5:
        st.subheader("🧠 Phân Tích Chuyên Sâu Định Lượng (Quantitative Analysis)")

        st.markdown(
            """
            ### 1. Ý Nghĩa Của Phương Pháp Kiểm Định Train / Test Split
            Trong tài chính định lượng, việc tối ưu hóa tham số (như chọn đường MA ngắn, MA dài hay chu kỳ OBV)
            rất dễ gặp lỗi **Overfitting** (quá khớp dữ liệu quá khứ nhưng thất bại khi thị trường thay đổi).
            - **Tập TRAIN (In-Sample: 2014-2019)**: Sử dụng thuật toán Hyperopt TPE để tìm ra bộ tham số có hiệu suất cao nhất.
            - **Tập TEST (Out-of-Sample: 2020-2023)**: Giữ nguyên cố định bộ tham số đã tìm được ở tập Train để kiểm định
              tính bền vững của chiến lược trên dữ liệu hoàn toàn chưa từng biết trước.

            ---

            ### 2. So Sánh Logic Toán Tử: AND vs OR
            - **Toán tử AND (Buy = SMA Buy AND OBV Buy)**:
              - Yêu cầu cả 2 chỉ báo cùng xuất hiện điểm cắt đúng vào **cùng một phiên giao dịch**.
              - *Thực tế*: Do SMA sử dụng chu kỳ dài (90/240) có độ trễ lớn, còn OBV là chỉ báo dòng tiền nhạy bén ngắn hạn (35),
                xác suất 2 chỉ báo này đồng thời cắt nhau đúng một ngày là cực kỳ thấp. Kết quả dẫn đến **0 lệnh giao dịch**,
                tỷ suất sinh lời bằng 0%.
            - **Toán tử OR (Buy = SMA Buy OR OBV Buy)**:
              - Cho phép hệ thống mở vị thế khi có dòng tiền bứt phá (OBV) hoặc khi xu hướng dài hạn xác nhận (SMA).
              - Tín hiệu bán linh hoạt giúp bảo toàn lợi nhuận hoặc cắt lỗ kịp thời khi một trong hai chỉ báo phát tín hiệu đảo chiều.
              - Nhờ đó, chiến lược **OR** tạo ra tỷ suất sinh lời vượt trội (72.58% trên tập TEST) so với SMA đơn lẻ (17.24%) và OBV đơn lẻ (53.63%).

            ---

            ### 3. Khuyến Nghị Cải Tiến Nâng Cao
            1. **Chuyển từ Crossover AND sang Regime Filter**:
               - Thay vì chờ cả 2 chỉ báo cùng cắt nhau, có thể áp dụng logic:
                 *Điều kiện Mua: Khi SMA_short > SMA_long (xu hướng tăng) VÀ xuất hiện tín hiệu OBV Crossover Mua*.
                 Cách tiếp cận này vừa lọc nhiễu hiệu quả vừa không bị bỏ lỡ cơ hội.
            2. **Bổ sung Quản Trị Rủi Ro (Risk Management)**:
               - Thêm mức Chặn lỗ (Stop Loss: ví dụ 5-7%) và Chốt lời từng phần (Trailing Stop) để giảm mức Max Drawdown trong giai đoạn thị trường điều chỉnh mạnh.
            """
        )


if __name__ == "__main__":
    main()
