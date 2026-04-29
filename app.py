"""
Part B + C — Live BTC Prediction Dashboard with Persistence.
Streamlit app that shows live predictions and stores history.
"""

import json
import os
import requests
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

CURRENCIES = {
    "USD": ("$",   "US Dollar"),
    "INR": ("₹",   "Indian Rupee"),
    "EUR": ("€",   "Euro"),
    "GBP": ("£",   "British Pound"),
    "JPY": ("¥",   "Japanese Yen"),
    "AUD": ("A$",  "Australian Dollar"),
    "CAD": ("C$",  "Canadian Dollar"),
    "SGD": ("S$",  "Singapore Dollar"),
    "AED": ("د.إ", "UAE Dirham"),
    "CHF": ("Fr",  "Swiss Franc"),
    "KRW": ("₩",   "South Korean Won"),
    "BRL": ("R$",  "Brazilian Real"),
}

TIMEZONES = {
    "IST":  ("Asia/Kolkata",        "India Standard Time"),
    "UTC":  ("UTC",                 "UTC"),
    "EST":  ("America/New_York",    "US Eastern"),
    "CST":  ("America/Chicago",     "US Central"),
    "PST":  ("America/Los_Angeles", "US Pacific"),
    "GMT":  ("Europe/London",       "London / GMT"),
    "CET":  ("Europe/Paris",        "Central European"),
    "GST":  ("Asia/Dubai",          "Gulf Standard Time"),
    "SGT":  ("Asia/Singapore",      "Singapore"),
    "JST":  ("Asia/Tokyo",          "Japan Standard Time"),
    "HKT":  ("Asia/Hong_Kong",      "Hong Kong"),
    "AEST": ("Australia/Sydney",    "Australia Eastern"),
}


@st.cache_data(ttl=3600)
def fetch_exchange_rates():
    """Fetch USD-based exchange rates from open.er-api.com (free, no key)."""
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        r.raise_for_status()
        return r.json()["rates"]
    except Exception:
        return {"USD": 1.0}


rates = fetch_exchange_rates()

# ─── Title row with dropdowns top-right ──────────────────────────────────────

_title_col, _curr_col, _tz_col = st.columns([3, 1, 1])
with _title_col:
    st.title("₿ Bitcoin Next-Hour Range Predictor")
    st.caption("GBM Monte Carlo model — BTCUSDT 1h candles via Binance")
with _curr_col:
    st.markdown("<br>", unsafe_allow_html=True)
    selected_currency = st.selectbox(
        "Currency",
        options=list(CURRENCIES.keys()),
        format_func=lambda c: f"{c} — {CURRENCIES[c][1]}",
        index=0,
    )
with _tz_col:
    st.markdown("<br>", unsafe_allow_html=True)
    selected_tz_key = st.selectbox(
        "Timezone",
        options=list(TIMEZONES.keys()),
        format_func=lambda z: f"{z} — {TIMEZONES[z][1]}",
        index=0,
    )

curr_symbol, curr_name = CURRENCIES[selected_currency]
fx_rate   = rates.get(selected_currency, 1.0)
tz_name, tz_label = TIMEZONES[selected_tz_key]


def fmt(usd_value):
    """Format a USD value in the selected currency."""
    return f"{curr_symbol}{usd_value * fx_rate:,.2f}"


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
col2.metric("Avg Width", fmt(metrics['avg_width_95']))
col3.metric("Mean Winkler", fmt(metrics['mean_winkler_95']))

st.divider()


# ─── Live Prediction ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # cache for 5 min
def get_live_prediction():
    """Fetch latest data and run prediction."""
    prices = fetch_btc_data(limit=500)
    low_95, high_95, current_price, sigma_fig, finals = predict_range(prices, n_sims=10000)
    return prices, low_95, high_95, current_price, sigma_fig, finals


with st.spinner("Fetching live data & running 10,000 simulations..."):
    prices, low_95, high_95, current_price, sigma_fig, finals = get_live_prediction()

# Keep UTC timestamp for history file, then convert index to selected tz
_utc_last_bar = prices.index[-1]
prices.index = prices.index.tz_localize('UTC').tz_convert(tz_name)

# ─── Sidebar: Price Alert ────────────────────────────────────────────────────

st.sidebar.header("Price Alert")
_alert_step = float(max(1.0, round(current_price * fx_rate / 200)))
alert_input = st.sidebar.number_input(
    f"Alert Price Level ({selected_currency})",
    min_value=0.0,
    value=0.0,
    step=_alert_step,
    format="%.2f",
    help="Enter a price to check whether it falls inside the predicted range.",
)

# ─── Volatility confidence computations ──────────────────────────────────────

log_ret = np.log(prices / prices.shift(1)).dropna()
recent_vol     = float(log_ret.tail(24).std())
full_roll_std  = log_ret.rolling(24).std().dropna()
p30 = float(np.percentile(full_roll_std, 30))
p70 = float(np.percentile(full_roll_std, 70))

if recent_vol < p30:
    conf_badge   = "🟢 High Confidence"
    conf_msg     = "Recent volatility is low — model predictions are more reliable."
    conf_fn      = st.success
    regime_label = "Calm"
elif recent_vol < p70:
    conf_badge   = "🟡 Medium Confidence"
    conf_msg     = "Recent volatility is moderate — predictions carry typical uncertainty."
    conf_fn      = st.warning
    regime_label = "Normal"
else:
    conf_badge   = "🔴 Low Confidence"
    conf_msg     = "Recent volatility is high — prediction intervals are wider and less reliable."
    conf_fn      = st.error
    regime_label = "Volatile"

# ─── Live Prediction Metrics ─────────────────────────────────────────────────

st.subheader("Live Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Current BTC Price", fmt(current_price))
c2.metric("Predicted Low (5%)", fmt(low_95))
c3.metric("Predicted High (95%)", fmt(high_95))

width = high_95 - low_95
st.info(
    f"**90% Prediction Range ({curr_name}):** {fmt(low_95)} — {fmt(high_95)}  "
    f"(width: {fmt(width)})"
)

# ─── Model Confidence Indicator ──────────────────────────────────────────────

conf_fn(f"**Model Confidence: {conf_badge}** — {conf_msg}")

# ─── Alert Threshold Display ─────────────────────────────────────────────────

if alert_input > 0:
    alert_usd = alert_input / fx_rate
    alert_str = f"{curr_symbol}{alert_input:,.2f}"
    if alert_usd < low_95:
        st.warning(
            f"⚠️ **{alert_str}** is **BELOW** your predicted range "
            f"({fmt(low_95)} – {fmt(high_95)}) — less than 5% chance of reaching this level."
        )
    elif alert_usd > high_95:
        st.warning(
            f"⚠️ **{alert_str}** is **ABOVE** your predicted range "
            f"({fmt(low_95)} – {fmt(high_95)}) — less than 5% chance of reaching this level."
        )
    else:
        st.success(
            f"✅ **{alert_str}** falls **WITHIN** your predicted range "
            f"({fmt(low_95)} – {fmt(high_95)})."
        )


# ─── Part C: Persistence ─────────────────────────────────────────────────────

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


# Save current prediction (UTC timestamp stored in file)
save_prediction(_utc_last_bar, current_price, low_95, high_95)


# ─── Chart: Last 50 bars + Prediction Ribbon ─────────────────────────────────

st.subheader("Price Chart with Prediction Range")

last_50 = prices.tail(50)

# Historical ribbon: for each bar t, band is [prev_price * exp(±1.96 * sigma_t)]
n_ribbon       = min(50, len(sigma_fig))
ribbon_times   = prices.index[-n_ribbon:]
prev_prices_arr = prices.iloc[-n_ribbon - 1:-1].values
ribbon_sigma   = sigma_fig.iloc[-n_ribbon:].values
ribbon_low     = prev_prices_arr * np.exp(-1.96 * ribbon_sigma) * fx_rate
ribbon_high    = prev_prices_arr * np.exp( 1.96 * ribbon_sigma) * fx_rate

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ribbon_times, y=ribbon_high,
    mode='lines', line=dict(width=0),
    name='95% Band', showlegend=True, hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=ribbon_times, y=ribbon_low,
    mode='lines', fill='tonexty',
    fillcolor='rgba(255, 165, 0, 0.18)',
    line=dict(width=0), name='95% Band', showlegend=False, hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=last_50.index, y=last_50.values * fx_rate,
    mode='lines+markers', name='BTC Close',
    line=dict(color='#1f77b4', width=2), marker=dict(size=3),
))

next_bar_time = last_50.index[-1] + pd.Timedelta(hours=1)
fig.add_shape(
    type="rect",
    x0=last_50.index[-1], x1=next_bar_time,
    y0=low_95 * fx_rate, y1=high_95 * fx_rate,
    fillcolor="rgba(50, 200, 50, 0.2)",
    line=dict(color="rgba(50, 200, 50, 0.6)", width=1),
)
fig.add_trace(go.Scatter(
    x=[next_bar_time, next_bar_time],
    y=[low_95 * fx_rate, high_95 * fx_rate],
    mode='markers+text',
    marker=dict(size=8, color='green', symbol='diamond'),
    text=[fmt(low_95), fmt(high_95)],
    textposition=['bottom center', 'top center'],
    name='Next-Hour Range', showlegend=True,
))

fig.update_layout(
    xaxis_title=f"Time ({selected_tz_key})",
    yaxis_title=f"Price ({selected_currency})",
    height=500, template="plotly_white", hovermode="x unified",
    yaxis=dict(tickprefix=curr_symbol, tickformat=",.0f"),
)
st.plotly_chart(fig, use_container_width=True)


# ─── Distribution Plot ────────────────────────────────────────────────────────

st.subheader("Simulated Next-Hour Price Distribution")

finals_conv = finals * fx_rate
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=finals_conv, nbinsx=100, name="Simulated prices",
    marker_color='rgba(31, 119, 180, 0.5)',
    marker_line=dict(color='rgba(31, 119, 180, 0.8)', width=0.5),
))
fig_dist.add_vline(
    x=low_95 * fx_rate, line_dash="dash", line_color="red", line_width=2,
    annotation_text=f"5th pct: {fmt(low_95)}", annotation_position="top right",
)
fig_dist.add_vline(
    x=high_95 * fx_rate, line_dash="dash", line_color="red", line_width=2,
    annotation_text=f"95th pct: {fmt(high_95)}", annotation_position="top left",
)
fig_dist.add_vline(
    x=current_price * fx_rate, line_dash="solid", line_color="black", line_width=2,
    annotation_text=f"Current: {fmt(current_price)}", annotation_position="top right",
)
fig_dist.update_layout(
    xaxis_title=f"Simulated Close Price ({selected_currency})",
    yaxis_title="Frequency",
    height=350, template="plotly_white", showlegend=False,
    xaxis=dict(tickprefix=curr_symbol, tickformat=",.0f"),
)
st.plotly_chart(fig_dist, use_container_width=True)


# ─── Volatility Regime ────────────────────────────────────────────────────────

st.subheader(f"Volatility Regime — Current: **{regime_label}**")

roll_vol_pct = log_ret.rolling(24).std().tail(50) * 100
p30_pct = p30 * 100
p70_pct = p70 * 100

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=roll_vol_pct.index, y=roll_vol_pct.values,
    mode='lines', name='24h Rolling Vol (%)',
    line=dict(color='#e377c2', width=2),
    fill='tozeroy', fillcolor='rgba(227, 119, 194, 0.12)',
))
fig_vol.add_hline(
    y=p30_pct, line_dash="dot", line_color="green", line_width=1.5,
    annotation_text="30th pct — Calm threshold", annotation_position="bottom right",
)
fig_vol.add_hline(
    y=p70_pct, line_dash="dot", line_color="red", line_width=1.5,
    annotation_text="70th pct — Volatile threshold", annotation_position="top right",
)
fig_vol.update_layout(
    xaxis_title=f"Time ({selected_tz_key})",
    yaxis_title="Hourly Volatility (%)",
    height=300, template="plotly_white",
    yaxis=dict(ticksuffix="%"),
)
st.plotly_chart(fig_vol, use_container_width=True)


# ─── Prediction History (Part C) ─────────────────────────────────────────────

history = load_history()
if not history.empty:
    st.subheader("Prediction History")
    st.caption("Predictions are saved on each visit. Actuals fill in as candles close.")

    # Convert timestamps to selected tz for display and matching
    history["timestamp"] = (
        pd.to_datetime(history["timestamp"])
        .dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert(tz_name)
    )
    history = history.drop_duplicates(subset=["timestamp"], keep="last")
    history = history.sort_values("timestamp", ascending=False)

    price_dict = prices.to_dict()
    history["actual"] = history["timestamp"].apply(lambda t: price_dict.get(t, None))
    history["hit"] = history.apply(
        lambda r: "✅" if r["actual"] and r["low_95"] <= r["actual"] <= r["high_95"]
        else ("❌" if r["actual"] else "⏳"),
        axis=1
    )

    # ─── Accuracy Tracker ────────────────────────────────────────────────────

    resolved = history[history["actual"].notna()]
    if not resolved.empty:
        n_total = len(resolved)
        n_hits  = int((resolved["hit"] == "✅").sum())
        acc_pct = n_hits / n_total * 100
        a1, a2, a3 = st.columns(3)
        a1.metric("Predictions with Actuals", n_total)
        a2.metric("Hits", f"{n_hits} / {n_total}")
        a3.metric("Live Accuracy", f"{acc_pct:.1f}%")

    # ─── History Table ────────────────────────────────────────────────────────

    hist_display = history[["timestamp", "current_price", "low_95", "high_95", "actual", "hit"]].copy()
    for col in ["current_price", "low_95", "high_95", "actual"]:
        hist_display[col] = pd.to_numeric(hist_display[col], errors='coerce') * fx_rate

    price_fmt = f"{curr_symbol}%.2f"
    st.dataframe(
        hist_display.head(50),
        use_container_width=True,
        column_config={
            "timestamp": f"Predicted For ({selected_tz_key})",
            "current_price": st.column_config.NumberColumn(f"Price at Prediction ({selected_currency})", format=price_fmt),
            "low_95": st.column_config.NumberColumn(f"Low 5% ({selected_currency})", format=price_fmt),
            "high_95": st.column_config.NumberColumn(f"High 95% ({selected_currency})", format=price_fmt),
            "actual": st.column_config.NumberColumn(f"Actual Close ({selected_currency})", format=price_fmt),
            "hit": "Result"
        }
    )

st.divider()
st.caption("Built by Suhaan Raqeeb Khavas | AlphaI × Polaris Build Challenge")
