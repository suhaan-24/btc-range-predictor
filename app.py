"""
Part B + C — Live BTC Prediction Dashboard with Persistence.
Streamlit app that shows live predictions and stores history.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from model import fetch_btc_data, predict_range


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Range Predictor",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Next-Hour Range Predictor")
st.caption("GBM Monte Carlo model — BTCUSDT 1h candles via Binance")


# ─── Load Backtest Metrics ───────────────────────────────────────────────────

METRICS_FILE = "backtest_metrics.json"
HISTORY_FILE = "prediction_history.jsonl"

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
else:
    metrics = {"coverage_95": 0.0, "avg_width_95": 0.0, "mean_winkler_95": 0.0}

# ─── Backtest Metrics Row ────────────────────────────────────────────────────

st.subheader("Backtest Performance (30-day)")
col1, col2, col3 = st.columns(3)
col1.metric("Coverage (95%)", f"{metrics['coverage_95']:.2%}")
col2.metric("Avg Width", f"${metrics['avg_width_95']:,.2f}")
col3.metric("Mean Winkler", f"${metrics['mean_winkler_95']:,.2f}")

st.divider()


# ─── Live Prediction ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # cache for 5 min
def get_live_prediction():
    """Fetch latest data and run prediction."""
    prices = fetch_btc_data(limit=500)
    low_95, high_95, current_price, sigma_fig = predict_range(prices, n_sims=10000)
    return prices, low_95, high_95, current_price, sigma_fig


with st.spinner("Fetching live data & running 10,000 simulations..."):
    prices, low_95, high_95, current_price, sigma_fig = get_live_prediction()

# Keep UTC timestamp for history file, then convert index to IST for display
_utc_last_bar = prices.index[-1]
prices.index = prices.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

# Display current prediction
st.subheader("Live Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Current BTC Price", f"${current_price:,.2f}")
c2.metric("Predicted Low (2.5%)", f"${low_95:,.2f}")
c3.metric("Predicted High (97.5%)", f"${high_95:,.2f}")

width = high_95 - low_95
st.info(
    f"**95% Prediction Range:** ${low_95:,.2f} — ${high_95:,.2f}  "
    f"(width: ${width:,.2f})"
)


# ─── Part C: Persistence ────────────────────────────────────────────────────

def save_prediction(timestamp, current_price, low_95, high_95):
    """Append prediction to history file."""
    entry = {
        "timestamp": str(timestamp),
        "current_price": float(current_price),
        "low_95": float(low_95),
        "high_95": float(high_95),
        "predicted_at": datetime.utcnow().isoformat()
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_history():
    """Load prediction history and fill in actuals where possible."""
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    records = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# Save current prediction (store UTC timestamp in history file)
save_prediction(_utc_last_bar, current_price, low_95, high_95)


# ─── Chart: Last 50 bars + Prediction Ribbon ────────────────────────────────

st.subheader("Price Chart with Prediction Range")

last_50 = prices.tail(50)

# Build historical ribbon: for each of last 50 bars at time t,
# show the predicted band based on price and FIGARCH sigma at t-1.
n_ribbon = min(50, len(sigma_fig))
ribbon_times = prices.index[-n_ribbon:]
prev_prices = prices.iloc[-n_ribbon - 1:-1].values
ribbon_sigma = sigma_fig.iloc[-n_ribbon:].values
ribbon_low = prev_prices * np.exp(-1.96 * ribbon_sigma)
ribbon_high = prev_prices * np.exp(1.96 * ribbon_sigma)

fig = go.Figure()

# Historical ribbon (upper bound, then lower filled)
fig.add_trace(go.Scatter(
    x=ribbon_times,
    y=ribbon_high,
    mode='lines',
    line=dict(width=0),
    name='95% Band',
    showlegend=True,
    hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=ribbon_times,
    y=ribbon_low,
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 165, 0, 0.18)',
    line=dict(width=0),
    name='95% Band',
    showlegend=False,
    hoverinfo='skip',
))

# Price line (drawn on top of ribbon)
fig.add_trace(go.Scatter(
    x=last_50.index,
    y=last_50.values,
    mode='lines+markers',
    name='BTC Close',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=3)
))

# Next-hour prediction as a shaded box extending one bar into the future
next_bar_time = last_50.index[-1] + pd.Timedelta(hours=1)

fig.add_shape(
    type="rect",
    x0=last_50.index[-1], x1=next_bar_time,
    y0=low_95, y1=high_95,
    fillcolor="rgba(50, 200, 50, 0.2)",
    line=dict(color="rgba(50, 200, 50, 0.6)", width=1),
)

fig.add_trace(go.Scatter(
    x=[next_bar_time, next_bar_time],
    y=[low_95, high_95],
    mode='markers+text',
    marker=dict(size=8, color='green', symbol='diamond'),
    text=[f'${low_95:,.0f}', f'${high_95:,.0f}'],
    textposition=['bottom center', 'top center'],
    name='Next-Hour Range',
    showlegend=True
))

fig.update_layout(
    xaxis_title="Time (IST)",
    yaxis_title="Price (USD)",
    height=500,
    template="plotly_white",
    hovermode="x unified",
    yaxis=dict(tickprefix="$", tickformat=",.0f"),
)

st.plotly_chart(fig, use_container_width=True)


# ─── Prediction History (Part C) ────────────────────────────────────────────

history = load_history()
if not history.empty:
    st.subheader("Prediction History")
    st.caption("Predictions are saved on each visit. Actuals fill in as candles close.")

    # Convert history timestamps from UTC to IST for display and matching
    history["timestamp"] = (
        pd.to_datetime(history["timestamp"])
        .dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert('Asia/Kolkata')
    )
    history = history.drop_duplicates(subset=["timestamp"], keep="last")
    history = history.sort_values("timestamp", ascending=False)

    # Match actuals (both prices.index and history timestamps are now IST)
    price_dict = prices.to_dict()
    history["actual"] = history["timestamp"].apply(
        lambda t: price_dict.get(t, None)
    )
    history["hit"] = history.apply(
        lambda r: "✅" if r["actual"] and r["low_95"] <= r["actual"] <= r["high_95"]
        else ("❌" if r["actual"] else "⏳"),
        axis=1
    )

    display_cols = ["timestamp", "current_price", "low_95", "high_95", "actual", "hit"]
    st.dataframe(
        history[display_cols].head(50),
        use_container_width=True,
        column_config={
            "timestamp": "Predicted For (IST)",
            "current_price": st.column_config.NumberColumn("Price at Prediction", format="$%.2f"),
            "low_95": st.column_config.NumberColumn("Low (2.5%)", format="$%.2f"),
            "high_95": st.column_config.NumberColumn("High (97.5%)", format="$%.2f"),
            "actual": st.column_config.NumberColumn("Actual Close", format="$%.2f"),
            "hit": "Result"
        }
    )

st.divider()
st.caption("Built by Suhaan Raqeeb Khavas | AlphaI × Polaris Build Challenge")
