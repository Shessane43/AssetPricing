import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import norm
from scipy.optimize import newton, brentq
from scipy.interpolate import UnivariateSpline, griddata

import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ----------------------------
# Black-Scholes helpers
# ----------------------------
def bs_price(S, K, T, r, q, sigma, option_type: str) -> float:
    option_type = option_type.lower()
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan

    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt
    d2 = d1 - vol_sqrt

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def implied_volatility(S, K, T, r, q, market_price, option_type, sigma0=0.2, tol=1e-8, max_iter=100):
    option_type = option_type.lower()

    if S <= 0 or K <= 0 or T <= 0 or market_price is None:
        return np.nan
    if not np.isfinite(market_price) or market_price <= 0:
        return np.nan

    sigma0 = float(np.clip(sigma0, 1e-4, 5.0))

    def f(sig):
        return bs_price(S, K, T, r, q, sig, option_type) - market_price

    # Newton
    try:
        vol = newton(
            func=f,
            x0=sigma0,
            fprime=lambda sig: bs_vega(S, K, T, r, q, max(sig, 1e-8)),
            tol=tol,
            maxiter=max_iter,
        )
        if np.isfinite(vol) and (1e-6 < vol <= 5.0):
            return float(vol)
    except Exception:
        pass

    # Brent (robuste)
    lo, hi = 1e-6, 5.0
    try:
        f_lo, f_hi = f(lo), f(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return np.nan
        if f_lo * f_hi > 0:
            hi2 = 10.0
            f_hi2 = f(hi2)
            if np.isfinite(f_hi2) and f_lo * f_hi2 <= 0:
                hi = hi2
            else:
                return np.nan

        vol = brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
        if np.isfinite(vol) and (1e-6 < vol <= 10.0):
            return float(vol)
    except Exception:
        return np.nan

    return np.nan


# ----------------------------
# Yahoo data
# ----------------------------
def _mid_price_from_row(row) -> float:
    bid = row.get("bid", 0.0)
    ask = row.get("ask", 0.0)
    last = row.get("lastPrice", 0.0)

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float(0.5 * (bid + ask))
    if pd.notna(last) and last > 0:
        return float(last)
    return np.nan


def get_market_prices_yahoo(ticker, T_days=None):
    stock = yf.Ticker(ticker)
    dates = getattr(stock, "options", None)
    if not dates:
        return {}, {}, None

    if T_days is not None:
        maturity = min(
            dates,
            key=lambda x: abs((pd.to_datetime(x) - pd.Timestamp.today()).days - int(T_days)),
        )
    else:
        maturity = dates[0]

    chain = stock.option_chain(maturity)
    df_calls = chain.calls.copy()
    df_puts = chain.puts.copy()

    df_calls["mid"] = df_calls.apply(_mid_price_from_row, axis=1)
    df_puts["mid"] = df_puts.apply(_mid_price_from_row, axis=1)

    df_calls = df_calls[(df_calls["mid"].notna()) & (df_calls["mid"] > 0)]
    df_puts = df_puts[(df_puts["mid"].notna()) & (df_puts["mid"] > 0)]

    calls = {float(row["strike"]): float(row["mid"]) for _, row in df_calls.iterrows()}
    puts = {float(row["strike"]): float(row["mid"]) for _, row in df_puts.iterrows()}

    return calls, puts, maturity


def get_all_option_maturities(ticker):
    stock = yf.Ticker(ticker)
    maturities = getattr(stock, "options", None)
    return maturities if maturities else []


# ----------------------------
# Build IV curve
# ----------------------------
def generate_vol_curve(S, maturity_date, r, q, market_calls, market_puts, option_type):
    option_type = option_type.lower()
    T = (pd.to_datetime(maturity_date) - pd.Timestamp.today()).days / 365.0
    if T <= 0:
        return [], [], np.nan, {}

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    strikes = sorted(set(market_calls.keys()).union(set(market_puts.keys())))
    prices_for_type = {}

    for K in strikes:
        c = market_calls.get(K, None)
        p = market_puts.get(K, None)

        if option_type == "call":
            if c is not None and np.isfinite(c):
                price = c
                if p is not None and np.isfinite(p):
                    c_par = p + (S * disc_q - K * disc_r)
                    price = 0.5 * (c + c_par)
                prices_for_type[K] = float(price)
            elif p is not None and np.isfinite(p):
                prices_for_type[K] = float(p + (S * disc_q - K * disc_r))
        else:
            if p is not None and np.isfinite(p):
                price = p
                if c is not None and np.isfinite(c):
                    p_par = c + (K * disc_r - S * disc_q)
                    price = 0.5 * (p + p_par)
                prices_for_type[K] = float(price)
            elif c is not None and np.isfinite(c):
                prices_for_type[K] = float(c + (K * disc_r - S * disc_q))

    strikes_out = sorted(prices_for_type.keys())
    vols = [implied_volatility(S, K, T, r, q, prices_for_type[K], option_type) for K in strikes_out]

    return strikes_out, vols, T, prices_for_type


# ----------------------------
# 2D plot
# ----------------------------
def smooth_vol_curve(strikes, vols, num_points=120, smoothing_factor=1.0):
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)

    mask = np.isfinite(vols)
    strikes = strikes[mask]
    vols = vols[mask]

    if strikes.size < 4:
        return strikes, vols

    spline = UnivariateSpline(strikes, vols, s=float(smoothing_factor), k=3)
    strikes_smooth = np.linspace(strikes.min(), strikes.max(), int(num_points))
    vols_smooth = spline(strikes_smooth)
    return strikes_smooth, vols_smooth


def plot_vol_curve(strikes, vols, maturity=None, K=None, title="Implied Volatility Curve"):
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    mask = np.isfinite(vols)

    strikes_plot = strikes[mask]
    vols_plot = vols[mask]
    strikes_smooth, vols_smooth = smooth_vol_curve(strikes_plot, vols_plot)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(strikes_plot, vols_plot, marker="x", s=60, label="Market IV points")
    if len(strikes_smooth) > 1:
        ax.plot(strikes_smooth, vols_smooth, lw=2, label="Smoothed spline")

    if K is not None and len(strikes_smooth) > 1:
        vol_at_K = np.interp(K, strikes_smooth, vols_smooth)
        ax.axvline(K, linestyle="--", lw=1.5, alpha=0.8)
        ax.scatter([K], [vol_at_K], s=80, zorder=5, label=f"IV@K≈{vol_at_K:.4f}")

    full_title = title
    if maturity is not None:
        full_title += f" — maturity: {maturity}"
    ax.set_title(full_title)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    return fig


# ----------------------------
# 3D surface plot 
# ----------------------------
def plot_vol_surface(vol_curves, grid_n=50):
    all_strikes, all_T, all_vols = [], [], []

    for maturity, data in vol_curves.items():
        strikes = np.asarray(data["strikes"], dtype=float)
        vols = np.asarray(data["vols"], dtype=float)

        T_years = data.get("T", None)
        if T_years is None or not np.isfinite(T_years):
            T_years = (pd.to_datetime(maturity) - pd.Timestamp.today()).days / 365.0

        T_arr = np.full_like(strikes, float(T_years), dtype=float)
        mask = np.isfinite(vols)

        all_strikes.extend(strikes[mask].tolist())
        all_T.extend(T_arr[mask].tolist())
        all_vols.extend(vols[mask].tolist())

    if len(all_strikes) < 10:
        fig = go.Figure()
        fig.update_layout(title="Not enough IV points to build a surface.")
        return fig

    all_strikes = np.asarray(all_strikes, dtype=float)
    all_T = np.asarray(all_T, dtype=float)
    all_vols = np.asarray(all_vols, dtype=float)

    strikes_grid = np.linspace(float(all_strikes.min()), float(all_strikes.max()), int(grid_n))
    T_grid = np.linspace(float(all_T.min()), float(all_T.max()), int(grid_n))
    X, Z = np.meshgrid(strikes_grid, T_grid)

    # IMPORTANT : nearest ne crash jamais (pas de QhullError)
    Y = griddata((all_strikes, all_T), all_vols, (X, Z), method="nearest")

    fig = go.Figure(data=[go.Surface(x=X, y=Z, z=Y)])
    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Maturity (years)",
            zaxis_title="Implied Volatility",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )
    return fig