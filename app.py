import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Solomon MVP", page_icon="📈", layout="wide")

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi", "macd", "macd_signal", "macd_hist",
    "atr_pct", "vol_z", "ema_fast_dist", "ema_slow_dist"
]

@st.cache_data(ttl=300)
def get_binance_klines(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["time","open","high","low","close","volume"]]

def rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_features(df, horizon=6, threshold=0.002):
    x = df.copy()
    x["ret_1"] = x.close.pct_change(1)
    x["ret_3"] = x.close.pct_change(3)
    x["ret_6"] = x.close.pct_change(6)
    x["ret_12"] = x.close.pct_change(12)

    ema12 = x.close.ewm(span=12, adjust=False).mean()
    ema26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x.macd - x.macd_signal
    x["rsi"] = rsi(x.close)

    prev_close = x.close.shift(1)
    tr = pd.concat([
        x.high - x.low,
        (x.high - prev_close).abs(),
        (x.low - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / x.close

    vol_mean = x.volume.rolling(30).mean()
    vol_std = x.volume.rolling(30).std()
    x["vol_z"] = (x.volume - vol_mean) / vol_std.replace(0, np.nan)

    ema20 = x.close.ewm(span=20, adjust=False).mean()
    ema50 = x.close.ewm(span=50, adjust=False).mean()
    x["ema_fast_dist"] = x.close / ema20 - 1
    x["ema_slow_dist"] = x.close / ema50 - 1

    future_ret = x.close.shift(-horizon) / x.close - 1
    x["target"] = np.select(
        [future_ret > threshold, future_ret < -threshold],
        [1, -1],
        default=0
    )
    return x.dropna(subset=FEATURES + ["target"]).copy()

def train_and_predict(df, test_size=0.2):
    n = len(df)
    split = int(n * (1 - test_size))
    train = df.iloc[:split]
    test = df.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[FEATURES], train.target)

    pred = model.predict(test[FEATURES])
    acc = accuracy_score(test.target, pred)

    latest = df.iloc[[-1]][FEATURES]
    proba = model.predict_proba(latest)[0]
    classes = model.classes_
    probabilities = {int(c): float(p) for c, p in zip(classes, proba)}
    signal = max(probabilities, key=probabilities.get)

    return model, test, pred, acc, signal, probabilities

def signal_name(s):
    return {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}[int(s)]

def backtest(test, pred, fee=0.0004):
    bt = test.copy()
    bt["pred"] = pred
    bt["market_ret"] = bt.close.pct_change().fillna(0)
    bt["strategy_ret"] = bt["pred"].shift(1).fillna(0) * bt["market_ret"]

    changed = bt.pred.ne(bt.pred.shift(1)).fillna(False)
    bt.loc[changed, "strategy_ret"] -= fee

    bt["equity"] = (1 + bt.strategy_ret).cumprod()
    peak = bt.equity.cummax()
    bt["drawdown"] = bt.equity / peak - 1

    trades = bt.loc[bt.pred != 0, "strategy_ret"]
    win_rate = float((trades > 0).mean()) if len(trades) else 0.0
    total_return = float(bt.equity.iloc[-1] - 1)
    max_dd = float(bt.drawdown.min())
    return bt, total_return, max_dd, win_rate

st.title("🧠 Solomon MVP")
st.caption("Experimental ML market-analysis prototype — not financial advice.")

with st.sidebar:
    st.header("Parameters")
    symbol = st.text_input("Symbol", "BTCUSDT").upper().strip()
    interval = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.slider("Candles", 300, 1000, 1000)
    horizon = st.slider("Prediction horizon (candles)", 1, 24, 6)
    threshold = st.slider("Movement threshold", 0.001, 0.02, 0.002, 0.001)
    run = st.button("Run Solomon", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Downloading market data and training model..."):
            raw = get_binance_klines(symbol, interval, limit)
            data = make_features(raw, horizon=horizon, threshold=threshold)
            model, test, pred, acc, signal, probabilities = train_and_predict(data)
            bt, total_return, max_dd, win_rate = backtest(test, pred)

        p_short = probabilities.get(-1, 0)
        p_neutral = probabilities.get(0, 0)
        p_long = probabilities.get(1, 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", signal_name(signal))
        c2.metric("LONG probability", f"{p_long:.1%}")
        c3.metric("NEUTRAL probability", f"{p_neutral:.1%}")
        c4.metric("SHORT probability", f"{p_short:.1%}")

        st.subheader("Market")
        fig = go.Figure(go.Candlestick(
            x=raw.time, open=raw.open, high=raw.high,
            low=raw.low, close=raw.close
        ))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model validation")
        a, b, c, d = st.columns(4)
        a.metric("Holdout accuracy", f"{acc:.1%}")
        b.metric("Backtest return", f"{total_return:.1%}")
        c.metric("Max drawdown", f"{max_dd:.1%}")
        d.metric("Win rate", f"{win_rate:.1%}")

        st.info(
            "Important: accuracy alone is not a trading edge. "
            "This prototype is deliberately simple and does not yet model slippage, "
            "liquidity, news, order book, funding, regime changes, or realistic execution."
        )

        st.subheader("Backtest equity")
        eq = go.Figure()
        eq.add_trace(go.Scatter(x=bt.time, y=bt.equity, mode="lines", name="Strategy"))
        eq.update_layout(height=350, yaxis_title="Equity")
        st.plotly_chart(eq, use_container_width=True)

        st.subheader("Latest model inputs")
        latest = data.iloc[-1]
        cols = st.columns(4)
        metrics = [
            ("RSI", latest.rsi),
            ("MACD histogram", latest.macd_hist),
            ("Volume Z-score", latest.vol_z),
            ("ATR %", latest.atr_pct),
        ]
        for col, (name, value) in zip(cols, metrics):
            col.metric(name, f"{value:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("Set parameters on the left and press **Run Solomon**.")
    st.markdown("""
    ### What this MVP does
    - Downloads public BTC/USDT market data from Binance.
    - Builds technical and volume features.
    - Trains a Random Forest classifier.
    - Predicts LONG / NEUTRAL / SHORT probabilities.
    - Runs a simple out-of-sample backtest.
    - Displays the result in a web dashboard.

    ### What it does NOT do yet
    News, social sentiment, smart money, order book, on-chain data,
    macro events, crowd psychology, multi-asset relationships,
    reinforcement learning, or autonomous retraining.
    """)

