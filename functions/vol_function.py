import numpy as np
from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline, griddata
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def implied_volatility(
    S, K, T, r, q, market_price, option_type,
    sigma0=0.2, tol=1e-6, max_iter=100
):
    """
    Computes the implied volatility of a European vanilla option
    using the Newton-Raphson method, with protections against overflow and extreme values.
    """

    # Protect against invalid values
    if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
        return None

    # Limit initial sigma
    sigma0 = max(min(sigma0, 5.0), 1e-4)

    def bs_price(sigma):
        sigma = max(min(sigma, 5.0), 1e-8)
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        except FloatingPointError:
            return None

        if option_type.lower() == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def vega(sigma):
        sigma = max(min(sigma, 5.0), 1e-8)
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        except FloatingPointError:
            return 1e-8
        return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    try:
        vol = newton(
            func=lambda sigma: bs_price(sigma) - market_price,
            x0=sigma0,
            fprime=vega,
            tol=tol,
            maxiter=max_iter
        )
        if vol <= 0 or vol > 5:
            return None
        return vol
    except (RuntimeError, OverflowError, FloatingPointError):
        return None


def get_market_prices_yahoo(ticker, T_days=None):
    """
    Retrieves market option prices from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    dates = stock.options
    if not dates:
        return {}

    if T_days is not None:
        maturity = min(
            dates,
            key=lambda x: abs((pd.to_datetime(x) - pd.Timestamp.today()).days - T_days)
        )
    else:
        maturity = dates[0]

    chain = stock.option_chain(maturity)
    df_calls = chain.calls
    df_puts = chain.puts

    df_calls = df_calls[(df_calls["lastPrice"] > 0) & (df_calls["volume"] > 0)]
    df_puts = df_puts[(df_puts["lastPrice"] > 0) & (df_puts["volume"] > 0)]

    return (
        {row['strike']: row['lastPrice'] for _, row in df_calls.iterrows()},
        {row['strike']: row['lastPrice'] for _, row in df_puts.iterrows()},
        maturity
    )


def parity_call_put(S, K, T, r, q, price, option_type_wanted):
    """
    Adjusts option price using call-put parity.
    """
    if option_type_wanted.lower() == "call":
        price_corrected = price + (S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        price_corrected = price + (K * np.exp(-r * T) - S * np.exp(-q * T))
    return price_corrected


def generate_vol_curve(S, T, r, q, market_prices_calls, market_prices_puts, option_type):
    """
    Generates implied volatility curve across strikes.
    """
    T = (pd.to_datetime(T) - pd.Timestamp.today()).days / 365.0
    corrected_prices = {}

    if option_type.lower() == "put":
        for K in market_prices_calls:
            corrected_price = parity_call_put(S, K, T, r, q, market_prices_calls[K], "put")
            corrected_prices[K] = corrected_price
    else:
        for K in market_prices_puts:
            corrected_price = parity_call_put(S, K, T, r, q, market_prices_puts[K], "call")
            corrected_prices[K] = corrected_price

    market_prices = {}
    if option_type.lower() == "put":
        market_prices.update(market_prices_puts)
    else:
        market_prices.update(market_prices_calls)

    for K, price in corrected_prices.items():
        if K in market_prices:
            market_prices[K] = 0.5 * (market_prices[K] + price)
        else:
            market_prices[K] = price

    strikes = sorted(market_prices.keys())
    vols = [implied_volatility(S, K, T, r, q, market_prices[K], option_type) for K in strikes]
    return strikes, vols


def smooth_vol_curve(strikes, vols, num_points=100, smoothing_factor=1):
    """
    Smooths an implied volatility curve using a cubic spline.
    """
    strikes = np.array(strikes)
    vols = np.array(vols)
    mask = vols != None
    strikes = strikes[mask]
    vols = vols[mask]

    spline = UnivariateSpline(strikes, vols, s=smoothing_factor, k=3)
    strikes_smooth = np.linspace(strikes.min(), strikes.max(), num_points)
    vols_smooth = spline(strikes_smooth)

    return strikes_smooth, vols_smooth


def plot_vol_curve(strikes, vols, maturity=None, K=None, title="Implied Volatility Curve", smoothing_factor=1, num_points=100):
    """
    Plots implied volatility curve with optional smoothing and strike marker.
    """
    strikes = np.array(strikes)
    vols = np.array(vols)
    mask = vols != None
    strikes_plot = strikes[mask]
    vols_plot = vols[mask]

    from copy import deepcopy
    strikes_smooth, vols_smooth = smooth_vol_curve(
        deepcopy(strikes_plot),
        deepcopy(vols_plot),
        num_points=num_points,
        smoothing_factor=smoothing_factor
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.scatter(strikes_plot, vols_plot, marker="x", color="orange", s=80, label="Actual Points")
    ax.plot(strikes_smooth, vols_smooth, color="orange", lw=2, label="Smoothed Spline")

    if K is not None:
        vol_at_K = np.interp(K, strikes_smooth, vols_smooth)
        ax.axvline(K, color="orange", linestyle="--", lw=2, alpha=0.7)
        ax.scatter(K, vol_at_K, color="orange", s=100, zorder=5)
        legend_title = f"Estimated vol at strike {round(K,4)}: {round(vol_at_K,4)}"
    else:
        legend_title = None

    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color("orange")
    ax.tick_params(colors="orange")
    ax.set_xlabel("Strike", color="orange")
    ax.set_ylabel("Implied Volatility", color="orange")

    full_title = title
    if maturity is not None:
        full_title += f" - Maturity: {maturity}"
    ax.set_title(full_title, color="orange")
    ax.grid(True, linestyle="--", color="orange", alpha=0.3)

    legend = ax.legend(facecolor="black", edgecolor="orange", title=legend_title)
    if legend.get_title():
        legend.get_title().set_color("orange")
    for text in legend.get_texts():
        text.set_color("orange")

    if vols_plot.size > 0:
        ax.set_ylim(0, max(vols_plot) * 1.2)
    if strikes_plot.size > 0:
        buffer = (max(strikes_plot) - min(strikes_plot)) * 0.1
        ax.set_xlim(min(strikes_plot) - buffer, max(strikes_plot) + buffer)

    return fig


def get_all_option_maturities(ticker):
    """
    Returns all available option maturities for a ticker from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    maturities = stock.options
    if not maturities:
        print(f"No maturities available for {ticker}")
        return []
    return maturities


def generate_vol_curves_multiple_maturities(S, maturities, r, q, option_type, ticker):
    """
    Generates implied volatility curves for multiple maturities.
    """
    vol_curves = {}
    for T_date in maturities:
        market_prices_calls, market_prices_puts, real_maturity = get_market_prices_yahoo(
            ticker, T_days=(pd.to_datetime(T_date) - pd.Timestamp.today()).days
        )
        if not market_prices_calls and not market_prices_puts:
            print(f"No options available for maturity {T_date}")
            continue

        strikes, vols = generate_vol_curve(S, real_maturity, r, q, market_prices_calls, market_prices_puts, option_type)
        vol_curves[real_maturity] = {"strikes": strikes, "vols": vols}

    return vol_curves


def plot_vol_surface(vol_curves):
    """
    Plots a 3D implied volatility surface using Plotly.
    
    vol_curves: dict {maturity: {"strikes": [...], "vols": [...]}}

    X = Strike
    Y = Implied Volatility
    Z = Maturity (years)
    """
    all_strikes, all_vols, all_maturities = [], [], []

    for maturity, data in vol_curves.items():
        strikes = np.array(data["strikes"])
        vols = np.array(data["vols"])
        T_years = (pd.to_datetime(maturity) - pd.Timestamp.today()).days / 365.0
        maturities = np.full_like(strikes, T_years, dtype=float)
        mask = vols != None
        all_strikes.extend(strikes[mask])
        all_vols.extend(vols[mask])
        all_maturities.extend(maturities[mask])

    strikes_grid = np.linspace(min(all_strikes), max(all_strikes), 50)
    maturities_grid = np.linspace(min(all_maturities), max(all_maturities), 50)
    X, Z = np.meshgrid(strikes_grid, maturities_grid)
    Y = griddata((all_strikes, all_maturities), all_vols, (X, Z), method='linear')

    fig = go.Figure(data=[go.Surface(x=X, y=Z, z=Y, colorscale='Viridis')])
    fig.update_layout(
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity (years)',
            zaxis_title='Implied Volatility'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700
    )
    return fig
