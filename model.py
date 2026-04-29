"""
GBM-based Bitcoin 1-hour price range predictor.
Adapted from AlphaI starter notebook for BTCUSDT hourly candles.
"""

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_btc_data(limit=1000):
    """
    Fetch BTCUSDT 1-hour candles from Binance public API.
    Uses data-api.binance.vision (no geo-block in India).
    Returns a Series of close prices indexed by datetime.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data = []

    if limit <= 1000:
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        all_data = response.json()
    else:
        # Binance caps at 1000 per request, so paginate
        end_time = None
        remaining = limit
        while remaining > 0:
            batch = min(remaining, 1000)
            params = {"symbol": "BTCUSDT", "interval": "1h", "limit": batch}
            if end_time:
                params["endTime"] = end_time
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data = data + all_data
            end_time = data[0][0] - 1  # 1ms before earliest candle
            remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df["close"].sort_index()


# ─── Volatility & Model Fitting ─────────────────────────────────────────────

def rolling_entropy(x, window=60, bins=20):
    """Compute rolling Shannon entropy of residuals."""
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)


def fit_model(log_ret):
    """
    Fit FIGARCH model with Student-t distribution on log returns.
    Returns sigma_fig, residuals, and degrees of freedom (nu).
    """
    am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    sigma_fig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    return sigma_fig, resid, nu, res


# ─── GBM Simulation ─────────────────────────────────────────────────────────

def simulate_cyber_gbm(S0, mu, sigma_fig, H, M, params, bar_sigma2,
                       redundancy_val, info_filter_val, nu,
                       n_steps=1, dt=1, eps=1e-6):
    """
    Single-path GBM simulation with cyber-enhanced volatility.
    Uses Student-t innovations for fat tails.
    """
    S = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = sigma_fig.iloc[-1] ** 2
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0

    for t in range(1, n_steps + 1):
        H_val = min(H.iloc[-1] / H_max, 1.0)
        M_val = min(M.iloc[-1] / M_max, 1.0)
        crisis = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        sigma2 = (
            sigma_fig.iloc[-1]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, redundancy_val)
        sigma2 *= 1 + 0.5 * info_filter_val
        sigma2 = max(eps, min(sigma2, 0.5))
        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)

    return S


def predict_range(prices, n_sims=10000):
    """
    Given a Series of close prices (at least ~100 bars),
    predict the 95% confidence interval for the next bar's close.
    Returns (low_95, high_95, current_price).
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu = log_ret.mean()
    S0 = prices.iloc[-1]

    # Fit volatility model
    sigma_fig, resid, nu, _ = fit_model(log_ret)

    # Compute indicators
    H_series = rolling_entropy(resid)
    M_series = log_ret.abs().rolling(60).mean()
    bar_sigma2 = (sigma_fig**2).mean()
    redundancy = 1 + 0.1 * np.log1p(
        prices.rolling(5).var() / prices.rolling(20).var()
    )
    info_filter = (H_series > H_series.mean()).astype(float)

    # Calibrate params
    H_max = H_series.max()
    M_max = M_series.max()
    alpha0, delta0 = 0.5, 0.3
    if alpha0 * H_max + delta0 * M_max >= 1:
        fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
        alpha0 *= fac
        delta0 *= fac
    base_params = {
        'alpha': alpha0, 'delta': delta0,
        'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3
    }

    # Get latest indicator values
    redundancy_val = redundancy.iloc[-1]
    info_filter_val = info_filter.iloc[-1]

    # Monte Carlo
    finals = np.zeros(n_sims)
    for i in range(n_sims):
        path = simulate_cyber_gbm(
            S0, mu, sigma_fig, H_series, M_series,
            base_params.copy(), bar_sigma2,
            redundancy_val, info_filter_val, nu,
            n_steps=1, dt=1
        )
        finals[i] = path[1]

    low_95, high_95 = np.percentile(finals, [2.5, 97.5])
    return low_95, high_95, S0
