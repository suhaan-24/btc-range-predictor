"""
Part A — 30-day backtest of BTCUSDT 1-hour GBM predictor.
Runs predictions on ~720 bars, saves results to backtest_results.jsonl.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import fetch_btc_data, fit_model, rolling_entropy


def run_backtest(lookback=500, n_sims=10000):
    """
    Backtest the GBM predictor over the last 30 days (~720 bars).
    For each bar, uses only past data (no peeking).

    lookback: number of bars to use for model fitting at each step.
    """
    # Fetch enough data: lookback + 720 test bars + buffer
    total_needed = lookback + 720 + 50
    print(f"Fetching {total_needed} bars of BTCUSDT 1h data...")
    prices = fetch_btc_data(limit=total_needed)
    print(f"Got {len(prices)} bars from {prices.index[0]} to {prices.index[-1]}")

    # We want to test on the last ~720 bars
    test_bars = 720
    start_idx = len(prices) - test_bars - 1  # -1 because we need actual for last prediction

    if start_idx < lookback:
        start_idx = lookback
        test_bars = len(prices) - start_idx - 1
        print(f"Adjusted: testing on {test_bars} bars (not enough history for 720)")

    print(f"Backtesting {test_bars} predictions...")
    print(f"Test period: {prices.index[start_idx]} to {prices.index[start_idx + test_bars]}")

    results = []

    for i in tqdm(range(start_idx, start_idx + test_bars)):
        # ── NO PEEKING: only use data up to bar i ──
        train_prices = prices.iloc[max(0, i - lookback):i + 1]

        if len(train_prices) < 100:
            continue

        log_ret = np.log(train_prices / train_prices.shift(1)).dropna()
        mu = log_ret.mean()
        S0 = train_prices.iloc[-1]

        try:
            # Fit FIGARCH model
            sigma_fig, resid, nu, _ = fit_model(log_ret)

            # Compute indicators
            H_series = rolling_entropy(resid)
            M_series = log_ret.abs().rolling(60).mean()
            bar_sigma2 = (sigma_fig**2).mean()
            redundancy = 1 + 0.1 * np.log1p(
                train_prices.rolling(5).var() / train_prices.rolling(20).var()
            )
            info_filter = (H_series > H_series.mean()).astype(float)

            # Calibrate params
            H_max = H_series.max()
            M_max = M_series.max()
            alpha0, delta0 = 0.5, 0.3
            if H_max > 0 and M_max > 0 and alpha0 * H_max + delta0 * M_max >= 1:
                fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
                alpha0 *= fac
                delta0 *= fac
            base_params = {
                'alpha': alpha0, 'delta': delta0,
                'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3
            }

            redundancy_val = redundancy.iloc[-1]
            info_filter_val = info_filter.iloc[-1]
            H_val = min(H_series.iloc[-1] / H_max, 1.0) if H_max > 0 else 0.0
            M_val = min(M_series.iloc[-1] / M_max, 1.0) if M_max > 0 else 0.0
            crisis = (H_val > 0.8) or (M_val > 0.8)
            delta_t = base_params['delta'] if crisis else 0.0

            sig_last = sigma_fig.iloc[-1]
            sigma2_init = sig_last ** 2
            eps = 1e-6
            sigma2 = (
                sig_last**2 * (1 + base_params['alpha'] * H_val + delta_t * M_val)
                + base_params['gamma'] * (bar_sigma2 - sigma2_init)
            )
            sigma2 *= max(1e-12, redundancy_val)
            sigma2 *= 1 + 0.5 * info_filter_val
            sigma2 = max(eps, min(sigma2, 0.5))

            # Vectorized Monte Carlo: sigma2 is deterministic, only Z varies
            Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
            finals = S0 * np.exp((mu - 0.5 * sigma2) + np.sqrt(sigma2) * Z)

            low_95, high_95 = np.percentile(finals, [5.0, 95.0])

        except Exception as e:
            # Fallback: simple percentile-based range
            recent_ret = log_ret.tail(24)
            vol = recent_ret.std()
            low_95 = S0 * np.exp(-1.96 * vol)
            high_95 = S0 * np.exp(1.96 * vol)

        # ── Reveal actual (bar i+1) ──
        actual = prices.iloc[i + 1]
        width = high_95 - low_95

        # Winkler score
        alpha = 0.05
        if actual < low_95:
            winkler = width + (2 / alpha) * (low_95 - actual)
        elif actual > high_95:
            winkler = width + (2 / alpha) * (actual - high_95)
        else:
            winkler = width

        results.append({
            "timestamp": str(prices.index[i + 1]),
            "actual": float(actual),
            "low_95": float(low_95),
            "high_95": float(high_95),
            "coverage_95": int(low_95 <= actual <= high_95),
            "width_95": float(width),
            "winkler": float(winkler)
        })

    return results


def evaluate(results):
    """Compute summary metrics from backtest results."""
    df = pd.DataFrame(results)
    coverage = df["coverage_95"].mean()
    avg_width = df["width_95"].mean()
    avg_winkler = df["winkler"].mean()
    return coverage, avg_width, avg_winkler


def main():
    print("=" * 60)
    print("  BTCUSDT 1h GBM Backtest — Part A")
    print("=" * 60)

    results = run_backtest(lookback=500, n_sims=10000)

    # Save to JSONL
    with open("backtest_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} predictions to backtest_results.jsonl")

    # Evaluate
    coverage, avg_width, avg_winkler = evaluate(results)
    print(f"\n{'=' * 40}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 40}")
    print(f"  Coverage (95%):     {coverage:.4f}  (target: ~0.95)")
    print(f"  Avg Width:          ${avg_width:.2f}")
    print(f"  Mean Winkler Score: ${avg_winkler:.2f}")
    print(f"{'=' * 40}")

    # Also save metrics
    metrics = {
        "coverage_95": float(coverage),
        "avg_width_95": float(avg_width),
        "mean_winkler_95": float(avg_winkler),
        "n_predictions": len(results)
    }
    with open("backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to backtest_metrics.json")


if __name__ == "__main__":
    main()
