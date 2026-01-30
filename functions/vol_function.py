# functions/vol_function.py
import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline, Rbf

import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ============================================================
# Black-Scholes pricing and implied volatility (robust)
# ============================================================

def bs_price(S, K, T, r, q, sigma, option_type: str) -> float:
    """
    Compute the Black-Scholes price of a European option.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Black-Scholes option price.
    """
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
    """
    Compute the Black-Scholes vega (sensitivity of price to volatility).

    Returns
    -------
    float
        Vega of the option.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def no_arb_bounds(S, K, T, r, q, option_type: str):
    """
    Compute no-arbitrage bounds for a European option price.

    Returns
    -------
    tuple(float, float)
        Lower and upper arbitrage-free bounds.
    """
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K), S
        return max(0.0, K - S), K

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_type == "call":
        lo = max(0.0, S * disc_q - K * disc_r)
        hi = S * disc_q
    else:
        lo = max(0.0, K * disc_r - S * disc_q)
        hi = K * disc_r

    return lo, hi


def implied_volatility_bs(S, K, T, r, q, market_price, option_type,
                          vol_lower=1e-6, vol_upper=5.0, tol=1e-8, max_iter=200) -> float:
    """
    Compute Black-Scholes implied volatility using Brent's root-finding method.

    Includes arbitrage checks and numerical safeguards.

    Returns
    -------
    float
        Implied volatility or NaN if inversion fails.
    """
    option_type = option_type.lower()
    if S <= 0 or K <= 0 or T <= 0:
        return np.nan
    if market_price is None or (not np.isfinite(market_price)) or market_price <= 0:
        return np.nan

    lo, hi = no_arb_bounds(S, K, T, r, q, option_type)
    if not (lo <= market_price <= hi):
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, q, sig, option_type) - market_price

    try:
        f_lo = f(vol_lower)
        f_hi = f(vol_upper)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return np.nan

        if f_lo * f_hi > 0:
            vol_upper2 = 10.0
            f_hi2 = f(vol_upper2)
            if np.isfinite(f_hi2) and f_lo * f_hi2 <= 0:
                vol_upper = vol_upper2
            else:
                return np.nan

        vol = brentq(f, vol_lower, vol_upper, xtol=tol, maxiter=max_iter)
        if np.isfinite(vol) and 1e-6 < vol <= 10.0:
            return float(vol)
        return np.nan
    except Exception:
        return np.nan


# ============================================================
# Yahoo Finance option chain fetch and mid-price computation
# ============================================================

def _mid_price_from_row(row) -> float:
    """
    Compute the mid price from bid/ask quotes.
    Falls back to last traded price if bid/ask are unavailable.
    """
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float(0.5 * (bid + ask))
    if pd.notna(last) and last > 0:
        return float(last)
    return np.nan


def get_all_option_maturities(ticker: str):
    """
    Retrieve all available option maturities from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    maturities = getattr(stock, "options", None)
    return maturities if maturities else []


def get_market_prices_yahoo(ticker: str, T_days: int = None):
    """
    Fetch option mid prices from Yahoo Finance for a selected maturity.

    Parameters
    ----------
    ticker : str
        Underlying ticker.
    T_days : int, optional
        Target maturity in days.

    Returns
    -------
    calls : dict
        Call option prices indexed by strike.
    puts : dict
        Put option prices indexed by strike.
    maturity : str
        Selected maturity date.
    """
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


# ============================================================
# Implied volatility curve construction and cleaning
# ============================================================

def generate_vol_curve(S, maturity_date, r, q, market_calls, market_puts, option_type):
    """
    Build an implied volatility curve by inverting Black-Scholes prices.

    Uses call-put parity blending to reduce microstructure noise.
    """
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
                price = float(c)
                if p is not None and np.isfinite(p):
                    c_par = float(p + (S * disc_q - K * disc_r))
                    price = 0.5 * (price + c_par)
                prices_for_type[K] = float(price)
            elif p is not None and np.isfinite(p):
                prices_for_type[K] = float(p + (S * disc_q - K * disc_r))
        else:
            if p is not None and np.isfinite(p):
                price = float(p)
                if c is not None and np.isfinite(c):
                    p_par = float(c + (K * disc_r - S * disc_q))
                    price = 0.5 * (price + p_par)
                prices_for_type[K] = float(price)
            elif c is not None and np.isfinite(c):
                prices_for_type[K] = float(c + (K * disc_r - S * disc_q))

    strikes_out = sorted(prices_for_type.keys())
    vols = [
        implied_volatility_bs(S, K, T, r, q, prices_for_type[K], option_type)
        for K in strikes_out
    ]
    return strikes_out, vols, T, prices_for_type


def clean_iv_points(S, strikes, vols, T, r, q,
                    vega_min=1e-4, iv_min=0.01, iv_max=3.0):
    """
    Filter implied volatility points using admissible IV bounds
    and minimum vega thresholds.
    """
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)

    mask = np.isfinite(vols)
    strikes = strikes[mask]
    vols = vols[mask]

    K_clean, iv_clean = [], []
    for K, iv in zip(strikes, vols):
        if not (iv_min <= iv <= iv_max):
            continue
        vega = bs_vega(S, K, T, r, q, iv)
        if vega < vega_min:
            continue
        K_clean.append(float(K))
        iv_clean.append(float(iv))

    return np.asarray(K_clean), np.asarray(iv_clean)


def smooth_smile_in_strike(S, strikes, vols, T, r, q, num_points=140, smoothing_factor=0.8):
    """
    Smooth the implied volatility smile in strike space
    using a vega-weighted cubic spline.
    """
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)
    if strikes.size < 5:
        return strikes, vols

    weights = np.array([bs_vega(S, K, T, r, q, iv) for K, iv in zip(strikes, vols)], dtype=float)
    weights = np.clip(weights, 1e-4, None)

    spline = UnivariateSpline(strikes, vols, w=weights, s=float(smoothing_factor), k=3)
    K_grid = np.linspace(strikes.min(), strikes.max(), int(num_points))
    iv_grid = spline(K_grid)
    return K_grid, iv_grid


def plot_vol_curve(S, strikes, vols, maturity=None, K=None, T=None, r=None, q=None,
                   title="Implied Volatility Curve"):
    """
    Plot implied volatility smile and optional spline smoothing.
    """
    strikes = np.asarray(strikes, dtype=float)
    vols = np.asarray(vols, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.scatter(strikes, vols, marker="x", s=70, color="orange", label="IV points")

    if T is not None and r is not None and q is not None and strikes.size >= 5:
        K_s, iv_s = smooth_smile_in_strike(S, strikes, vols, T, r, q)
        ax.plot(K_s, iv_s, lw=2, color="orange", label="Vega-weighted spline")

        if K is not None:
            iv_at_K = np.interp(float(K), K_s, iv_s)
            ax.axvline(float(K), linestyle="--", color="orange", lw=1.5, alpha=0.8)
            ax.scatter([float(K)], [iv_at_K], s=80, zorder=5,
                       color="orange", label=f"IV@K≈{iv_at_K:.4f}")

    full_title = title
    if maturity is not None:
        full_title += f" — maturity: {maturity}"

    ax.set_title(full_title, color="white")
    ax.set_xlabel("Strike K", color="white")
    ax.set_ylabel("Implied Volatility", color="white")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    return fig


def plot_iv_surface_KT(vol_curves, grid_K=70, grid_T=45, rbf_smooth=0.2,
                       title="Implied Volatility Surface"):
    """
    Build and plot a smooth implied volatility surface
    directly in (K, T) coordinates using RBF interpolation.
    """
    Ks, Ts, IVs = [], [], []

    for maturity, data in vol_curves.items():
        strikes = np.asarray(data["strikes"], dtype=float)
        vols = np.asarray(data["vols"], dtype=float)
        T = float(data["T"])

        mask = np.isfinite(vols)
        if np.sum(mask) < 4:
            continue

        Ks.extend(strikes[mask].tolist())
        Ts.extend([T] * int(np.sum(mask)))
        IVs.extend(vols[mask].tolist())

    if len(Ks) < 12:
        fig = go.Figure()
        fig.update_layout(title="Not enough IV points to build a surface.")
        return fig

    Ks = np.asarray(Ks, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    IVs = np.asarray(IVs, dtype=float)

    K_grid = np.linspace(Ks.min(), Ks.max(), int(grid_K))
    T_grid = np.linspace(Ts.min(), Ts.max(), int(grid_T))
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    rbf = Rbf(Ks, Ts, IVs, function="multiquadric", smooth=float(rbf_smooth))
    Z = rbf(K_mesh, T_mesh)

    fig = go.Figure(data=[go.Surface(x=K_mesh, y=T_mesh, z=Z)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike K",
            yaxis_title="Time to maturity T",
            zaxis_title="Implied Volatility",
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return fig


def generate_heston_iv_curve(S, r, q, option_type, strikes, T, heston_params, HestonModel):
    """
    Compute implied volatilities from a Heston model
    for a given maturity and set of strikes.
    """
    vols_model = []
    for K in strikes:
        try:
            m = HestonModel(
                float(S), float(K), float(r), float(T), float(q),
                option_type=option_type,
                position="long",
                option_class="vanilla",
                **heston_params
            )
            p = m.price()
            iv = m.implied_volatility(p)
            if not (np.isfinite(iv) and 0.01 <= iv <= 3.0):
                iv = np.nan
        except Exception:
            iv = np.nan
        vols_model.append(iv)

    return np.asarray(vols_model, dtype=float)


def build_vol_error_surface(market_surface, model_surface):
    """
    Compute the implied volatility error surface
    between a market surface and a model surface.
    """
    error_surface = {}
    for maturity, mkt in market_surface.items():
        if maturity not in model_surface:
            continue

        K_mkt = np.asarray(mkt["strikes"], dtype=float)
        iv_mkt = np.asarray(mkt["vols"], dtype=float)

        K_mod = np.asarray(model_surface[maturity]["strikes"], dtype=float)
        iv_mod = np.asarray(model_surface[maturity]["vols"], dtype=float)

        if K_mod.shape != K_mkt.shape or np.any(K_mod != K_mkt):
            mask_mod = np.isfinite(iv_mod)
            if np.sum(mask_mod) < 2:
                continue
            iv_mod = np.interp(
                K_mkt,
                K_mod[mask_mod],
                iv_mod[mask_mod],
                left=np.nan,
                right=np.nan
            )

        mask = np.isfinite(iv_mkt) & np.isfinite(iv_mod)
        if np.sum(mask) < 4:
            continue

        error_surface[maturity] = {
            "strikes": K_mkt[mask].tolist(),
            "vols": (iv_mod[mask] - iv_mkt[mask]).tolist(),
            "T": float(mkt["T"]),
        }
    return error_surface


def select_calibration_points(market_surface, S, max_maturities=3,
                              strikes_per_mat=10, moneyness_band=(0.8, 1.2)):
    """
    Select a reduced and informative set of calibration points.

    Strategy:
    - Limit the number of maturities
    - Focus on near-ATM strikes
    - Cap the number of strikes per maturity

    Returns
    -------
    list, list, list
        Lists of strikes, maturities, and option prices.
    """
    items = sorted(market_surface.items(), key=lambda kv: float(kv[1]["T"]))
    if len(items) == 0:
        return [], [], []

    idx = []
    if max_maturities >= 1:
        idx.append(0)
    if max_maturities >= 2 and len(items) > 1:
        idx.append(len(items) // 2)
    if max_maturities >= 3 and len(items) > 2:
        idx.append(len(items) - 1)
    idx = sorted(set(idx))[:max_maturities]

    K_list, T_list, P_list = [], [], []

    for i in idx:
        maturity, data = items[i]
        T = float(data["T"])
        strikes = np.asarray(data["strikes"], dtype=float)

        lo, hi = moneyness_band
        mask = (strikes >= lo * S) & (strikes <= hi * S)
        strikes_band = strikes[mask] if np.any(mask) else strikes

        order = np.argsort(np.abs(strikes_band - S))
        strikes_sel = strikes_band[order][:strikes_per_mat]

        for K in strikes_sel:
            price = data["prices"].get(float(K), np.nan)
            if np.isfinite(price) and price > 0:
                K_list.append(float(K))
                T_list.append(T)
                P_list.append(float(price))

    return K_list, T_list, P_list
