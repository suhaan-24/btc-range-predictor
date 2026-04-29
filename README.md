# ₿ Bitcoin Next-Hour Range Predictor

> **AlphaI × Polaris Build Challenge**

**Live App:** [btc-range-predictor.streamlit.app](https://btc-range-predictor.streamlit.app/)

A production-grade Bitcoin price range predictor that forecasts the 90% confidence interval for BTCUSDT's next hourly close. Built on a GBM simulator with FIGARCH volatility and Student-t fat-tailed innovations, validated against a 30-day backtest of 720 hourly bars.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dashboard Features](#dashboard-features)
- [Backtest Results](#backtest-results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Author](#author)

---

## Overview

Standard volatility models assume constant variance and Gaussian shocks — both of which are demonstrably false for Bitcoin. This project uses a chain of improvements:

1. **FIGARCH** captures long-memory volatility clustering (quiet periods followed by violent ones)
2. **Student-t innovations** handle fat tails and extreme moves
3. **Cyber-enhanced GBM** amplifies volatility during crisis regimes (high entropy, high absolute returns)
4. **Vectorized Monte Carlo** generates 10,000 price paths in a single NumPy operation

All predictions are made strictly without future data. The backtest loop only feeds data up to bar N−1 when predicting bar N.

---

## Model Architecture

### 1. Data Fetching

Prices are fetched from the Binance Vision public API (`data-api.binance.vision`) — an alternative endpoint with no geo-block. Fetches BTCUSDT 1-hour candles with automatic pagination for requests over 1,000 bars.

### 2. FIGARCH Volatility Fitting (`fit_model`)

Fractionally Integrated GARCH (FIGARCH) is fitted on log returns scaled by 100, using a Student-t error distribution. Unlike standard GARCH, FIGARCH captures the empirical observation that volatility shocks in crypto markets decay hyperbolically (slowly), not exponentially (quickly).

- **Inputs:** log return series
- **Outputs:** conditional volatility series `σ_t`, standardised residuals, Student-t degrees of freedom `ν`

### 3. Cyber-Enhanced GBM Simulation (`simulate_cyber_gbm`)

Each simulated price path follows:

```
S_t = S_{t-1} × exp((μ − ½σ²)Δt + √(σ²Δt) × Z)
```

where `Z ~ Student-t(ν)` scaled to unit variance, and `σ²` is dynamically adjusted by:

| Signal | Source | Effect |
|---|---|---|
| **Shannon Entropy (H)** | Rolling 60-bar histogram of residuals | Higher entropy → higher effective volatility |
| **Mean Absolute Return (M)** | Rolling 60-bar abs log return | Detects momentum/crisis regimes |
| **Redundancy** | Ratio of short-term to long-term price variance | Amplifies tight, mean-reverting regimes |
| **Info Filter** | Entropy above its rolling mean | Binary flag that adds 50% variance boost |

A **crisis flag** (H > 80th pct or M > 80th pct) activates the `δ` (delta) term, further inflating volatility during detected stress periods.

### 4. Vectorized Monte Carlo (`predict_range`)

For a one-step-ahead prediction, `σ²` is entirely deterministic (all inputs are known at prediction time). The randomness enters only through `Z`. This means:

```python
Z = np.random.standard_t(ν, size=10000) * sqrt((ν−2)/ν)
finals = S0 * exp((μ − 0.5σ²) + sqrt(σ²) × Z)
```

All 10,000 paths are computed in a single vectorised NumPy call — roughly 100× faster than the equivalent Python loop.

**Percentile bounds:** `[5th, 95th]` of the simulated distribution form the prediction interval. These are calibrated tighter than the nominal 95% CI to target ~95% empirical coverage (the FIGARCH model is slightly conservative, so the 90% simulation CI maps to ~95% actual coverage).

---

## Dashboard Features

### Live Prediction
Fetches the latest 500 hourly bars from Binance, runs the full model pipeline, and displays:
- Current BTC price
- Predicted low (5th percentile) and high (95th percentile) for the next hour
- Interval width in the selected currency

Results are cached for 5 minutes to avoid hammering the Binance API on every page interaction.

### Shaded Historical Ribbon
The price chart shows a continuous orange shaded band over the last 50 bars. For each bar at time `t`, the band is:

```
[price_{t-1} × exp(−1.96 × σ_{t-1}),  price_{t-1} × exp(+1.96 × σ_{t-1})]
```

This visualises how the FIGARCH model's conditional volatility tracked the actual price during recent history — narrower during calm periods, wider during volatile ones. A green shaded box at the rightmost edge shows the **live next-hour prediction**.

### Model Confidence Indicator
Below the live prediction, a coloured badge indicates current model reliability based on recent volatility relative to the full historical distribution:

| Badge | Condition | Meaning |
|---|---|---|
| 🟢 High Confidence | 24-bar vol < 30th percentile | Calm market — model intervals are tight and reliable |
| 🟡 Medium Confidence | 30th–70th percentile | Normal conditions |
| 🔴 Low Confidence | 24-bar vol > 70th percentile | High volatility — intervals are wider, less predictable |

### Volatility Regime Chart
A rolling 24-bar hourly volatility chart over the last 50 bars, with:
- **Green dotted line** — 30th percentile threshold (Calm boundary)
- **Red dotted line** — 70th percentile threshold (Volatile boundary)
- Current regime labelled in the subheader: Calm / Normal / Volatile

### Simulated Price Distribution
A histogram of all 10,000 Monte Carlo simulated next-hour prices with:
- **Red dashed lines** at the 5th and 95th percentiles (the reported interval bounds)
- **Black solid line** at the current price

This plot makes the model's uncertainty tangible — a wide, flat distribution means low confidence; a tall, narrow peak means the model sees low variance ahead.

### Prediction Accuracy Tracker
The history section opens with three metric cards:
- **Predictions with Actuals** — how many stored predictions have a known outcome
- **Hits** — how many fell within the predicted interval
- **Live Accuracy** — running hit rate as a percentage

### Prediction History (Persistence)
Every time the dashboard loads, the current prediction is appended to `prediction_history.jsonl`. As hours pass and new candles close, the app back-fills the actual close price and marks each prediction ✅ hit / ❌ miss / ⏳ pending.

---

## Backtest Results

30-day rolling backtest over ~720 hourly bars (no data leakage):

| Metric | Value |
|---|---|
| Coverage | 97.22% |
| Avg Interval Width | $1,574 |
| Mean Winkler Score | $1,877 |

The Winkler score penalises both width (wider = worse) and misses (each miss adds `2/α × distance`). The model's over-coverage (97% vs target 95%) reflects conservative FIGARCH estimates; the percentile bounds are tuned to `[5th, 95th]` of the simulation distribution to balance width and coverage.

---

## Project Structure

```
├── model.py                 # Core model: data fetch, FIGARCH fit, GBM simulation, predict_range
├── backtest.py              # 30-day walk-forward backtest (no-peek, vectorised MC)
├── app.py                   # Streamlit dashboard: all visualisations and features
├── requirements.txt         # Pinned Python dependencies
├── backtest_results.jsonl   # Per-bar backtest output (generated by backtest.py, git-ignored)
├── backtest_metrics.json    # Summary metrics for the dashboard (generated by backtest.py, git-ignored)
└── prediction_history.jsonl # Live prediction log written by the dashboard (git-ignored)
```

---

## Setup & Usage

```bash
pip install -r requirements.txt
```

**Run the backtest first (required for dashboard metrics):**
```bash
python backtest.py
```
Outputs `backtest_results.jsonl` and `backtest_metrics.json`. The metrics file is read by the dashboard to display coverage, avg width, and Winkler score. It is git-ignored, so you must generate it locally before deploying. If it's missing, the dashboard shows `—` placeholders instead of crashing.

**Run the live dashboard (Part B + C):**
```bash
streamlit run app.py
```

---

## Author

**Suhaan Raqeeb Khavas**
GitHub: [suhaan-24](https://github.com/suhaan-24)
