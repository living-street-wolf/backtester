import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from datetime import timedelta

# =========================================
# PAGE
# =========================================
st.set_page_config(layout="wide")

st.title("📈 REALISTIC MACD + EMA Strategy Backtester")
st.write("Full state-machine simulation of your live trading bot")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("Backtest Settings")

symbol = st.sidebar.selectbox(
    "Symbol",
    ["SPY", "AAPL", "MSFT", "NVDA"]
)

period = st.sidebar.selectbox(
    "Period",
    ["1d", "5d", "7d", "30d"]
)

starting_equity = st.sidebar.number_input(
    "Starting Equity",
    value=10000
)

risk_per_trade = st.sidebar.slider(
    "Risk Per Trade %",
    0.001,
    0.05,
    0.01,
    0.001
)

max_daily_drawdown = st.sidebar.slider(
    "Max Daily Drawdown %",
    0.01,
    0.20,
    0.03,
    0.01
)

cooldown_bars = st.sidebar.slider(
    "Cooldown Bars",
    1,
    20,
    5
)

ema_length = st.sidebar.slider(
    "EMA Length",
    20,
    200,
    50
)

macd_fast = st.sidebar.slider(
    "MACD Fast",
    5,
    20,
    12
)

macd_slow = st.sidebar.slider(
    "MACD Slow",
    10,
    40,
    26
)

macd_signal = st.sidebar.slider(
    "MACD Signal",
    5,
    20,
    9
)

stop_pct = st.sidebar.slider(
    "Stop Loss %",
    0.1,
    2.0,
    0.5,
    0.1
)

tp_pct = st.sidebar.slider(
    "Take Profit %",
    0.1,
    3.0,
    0.9,
    0.1
)

use_zero_filter = st.sidebar.checkbox(
    "Use MACD Zero-Line Filter",
    value=True
)

use_single_trigger = st.sidebar.checkbox(
    "Use Single Trigger System",
    value=True
)

trigger_window = st.sidebar.slider(
    "Trigger Window (bars)",
    1,
    10,
    4
)

run_backtest = st.sidebar.button("Run Backtest")

# =========================================
# DATA
# =========================================
@st.cache_data
def load_data(symbol, period):

    df = yf.download(
        symbol,
        period=period,
        interval="1m",
        auto_adjust=True
    )

    df = df.dropna()

    return df
df = yf.download(symbol, period=period, interval="1m")

df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df.dropna()

# =========================================
# INDICATORS
# =========================================
def ema(series, length):
    return series.ewm(span=length).mean()


def add_indicators(df):

    df["EMA"] = ema(df["Close"], ema_length)

    fast = ema(df["Close"], macd_fast)
    slow = ema(df["Close"], macd_slow)

    df["MACD"] = fast - slow
    df["SIGNAL"] = ema(df["MACD"], macd_signal)

    return df


# =========================================
# BACKTEST ENGINE
# =========================================
def backtest(df):

    equity = starting_equity

    equity_curve = [equity]

    trades = []

    trade_log = []

    daily_pnl = 0

    cooldown = 0

    position = None

    setup_armed = False
    setup_dir = None
    setup_index = None

    for i in range(len(df)):

        if i < 60:
            continue

        row = df.iloc[i]

        price = float(row["Close"])

        ema_val = float(row["EMA"])

        macd_now = float(row["MACD"])
        signal_now = float(row["SIGNAL"])

        macd_prev = float(df["MACD"].iloc[i - 1])
        signal_prev = float(df["SIGNAL"].iloc[i - 1])

        timestamp = df.index[i]

        cooldown = max(0, cooldown - 1)

        # =========================================
        # DAILY DRAWDOWN LOCK
        # =========================================
        if daily_pnl <= -(equity * max_daily_drawdown):
            equity_curve.append(equity)
            continue

        # =========================================
        # MANAGE POSITION
        # =========================================
        if position is not None:

            side = position["side"]
            entry = position["entry"]
            qty = position["qty"]

            pnl = (
                (price - entry) * qty
                if side == "buy"
                else (entry - price) * qty
            )

            # =========================================
            # TRAILING
            # =========================================
            if side == "buy":

                position["best"] = max(position["best"], price)

                if pnl > 0:
                    position["stop"] = max(
                        position["stop"],
                        position["best"] * 0.997
                    )

            else:

                position["best"] = min(position["best"], price)

                if pnl > 0:
                    position["stop"] = min(
                        position["stop"],
                        position["best"] * 1.003
                    )

            exit_trade = False

            # LONG
            if side == "buy":

                if price <= position["stop"]:
                    exit_trade = True

                if price >= position["tp"]:
                    exit_trade = True

            # SHORT
            else:

                if price >= position["stop"]:
                    exit_trade = True

                if price <= position["tp"]:
                    exit_trade = True

            if exit_trade:

                equity += pnl
                daily_pnl += pnl

                trades.append(pnl)

                trade_log.append({
                    "Time": timestamp,
                    "Side": side,
                    "Entry": round(entry, 2),
                    "Exit": round(price, 2),
                    "PnL": round(pnl, 2),
                    "Equity": round(equity, 2)
                })

                position = None

                cooldown = cooldown_bars

            equity_curve.append(equity)

            continue

        # =========================================
        # NO POSITION
        # =========================================
        if cooldown > 0:
            equity_curve.append(equity)
            continue

        # =========================================
        # SIGNALS
        # =========================================
        bullish_cross = (
            macd_prev < signal_prev and
            macd_now > signal_now
        )

        bearish_cross = (
            macd_prev > signal_prev and
            macd_now < signal_now
        )

        # =========================================
        # ZERO FILTER
        # =========================================
        if use_zero_filter:

            bullish_cross = bullish_cross and macd_now < 0
            bearish_cross = bearish_cross and macd_now > 0

        # =========================================
        # EMA FILTER
        # =========================================
        bullish_confirm = price > ema_val
        bearish_confirm = price < ema_val

        # =========================================
        # SINGLE TRIGGER ARMING
        # =========================================
        if use_single_trigger:

            if bullish_cross:
                setup_armed = True
                setup_dir = "buy"
                setup_index = i

            elif bearish_cross:
                setup_armed = True
                setup_dir = "sell"
                setup_index = i

            # ACTIVE WINDOW
            if setup_armed:

                valid_window = (
                    i - setup_index <= trigger_window
                )

                if not valid_window:
                    setup_armed = False

                else:

                    if (
                        setup_dir == "buy" and
                        bullish_confirm
                    ):

                        stop = price * (1 - stop_pct / 100)

                        risk_per_share = abs(price - stop)

                        risk_amount = equity * risk_per_trade

                        qty = max(
                            1,
                            int(risk_amount / risk_per_share)
                        )

                        position = {
                            "side": "buy",
                            "entry": price,
                            "qty": qty,
                            "stop": stop,
                            "tp": price * (1 + tp_pct / 100),
                            "best": price
                        }

                        setup_armed = False

                    elif (
                        setup_dir == "sell" and
                        bearish_confirm
                    ):

                        stop = price * (1 + stop_pct / 100)

                        risk_per_share = abs(price - stop)

                        risk_amount = equity * risk_per_trade

                        qty = max(
                            1,
                            int(risk_amount / risk_per_share)
                        )

                        position = {
                            "side": "sell",
                            "entry": price,
                            "qty": qty,
                            "stop": stop,
                            "tp": price * (1 - tp_pct / 100),
                            "best": price
                        }

                        setup_armed = False

        # =========================================
        # SIMPLE MODE
        # =========================================
        else:

            if bullish_cross and bullish_confirm:

                stop = price * (1 - stop_pct / 100)

                risk_per_share = abs(price - stop)

                risk_amount = equity * risk_per_trade

                qty = max(
                    1,
                    int(risk_amount / risk_per_share)
                )

                position = {
                    "side": "buy",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "tp": price * (1 + tp_pct / 100),
                    "best": price
                }

            elif bearish_cross and bearish_confirm:

                stop = price * (1 + stop_pct / 100)

                risk_per_share = abs(price - stop)

                risk_amount = equity * risk_per_trade

                qty = max(
                    1,
                    int(risk_amount / risk_per_share)
                )

                position = {
                    "side": "sell",
                    "entry": price,
                    "qty": qty,
                    "stop": stop,
                    "tp": price * (1 - tp_pct / 100),
                    "best": price
                }

        equity_curve.append(equity)

    return trades, equity_curve, trade_log


# =========================================
# RUN
# =========================================
if run_backtest:

    with st.spinner("Running realistic backtest..."):

        df = load_data(symbol, period)

        df = add_indicators(df)

        trades, equity_curve, trade_log = backtest(df)

    # =========================================
    # STATS
    # =========================================
    st.subheader("📊 Results")

    total = len(trades)

    wins = len([x for x in trades if x > 0])

    losses = len([x for x in trades if x <= 0])

    winrate = (wins / total * 100) if total else 0

    total_pnl = sum(trades)

    final_equity = starting_equity + total_pnl

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Trades", total)
    col2.metric("Win Rate", f"{winrate:.2f}%")
    col3.metric("PnL", f"${total_pnl:,.2f}")
    col4.metric("Wins", wins)
    col5.metric("Losses", losses)

    st.metric("Final Equity", f"${final_equity:,.2f}")

    # =========================================
    # EQUITY CURVE
    # =========================================
    st.subheader("📈 Equity Curve")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(equity_curve)

    ax.set_title("Equity Curve")

    ax.set_xlabel("Trades / Time")
    ax.set_ylabel("Equity")

    st.pyplot(fig)

    # =========================================
    # TRADE DISTRIBUTION
    # =========================================
    st.subheader("📉 Trade Distribution")

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.hist(trades, bins=30)

    ax2.set_title("PnL Distribution")

    st.pyplot(fig2)

    # =========================================
    # TRADE LOG
    # =========================================
    st.subheader("📋 Trade Log")

    if len(trade_log) > 0:
        st.dataframe(pd.DataFrame(trade_log))
    else:
        st.warning("No trades found")
