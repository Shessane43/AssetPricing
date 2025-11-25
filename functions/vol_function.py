import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

def implied_volatility(S, K, T, r, q, market_price, option_type, tol=1e-6, max_iter=100):
    """
    Calcule la volatilité implicite d'une option vanille européenne via Newton-Raphson
    """
    sigma = 0.2  # guess initiale

    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "Call":
            price_est = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            price_est = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

        vega = S * np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-8:
            return None

        sigma -= (price_est - market_price) / vega
        if abs(price_est - market_price) < tol:
            return sigma

    return None


def get_market_prices_yahoo(ticker, option_type, T_days=None):
    """
    Récupère les prix d'options sur Yahoo Finance
    - option_type: "call" ou "put"
    - T_days : maturité approximative en jours (None = prochaine date disponible)
    """
    stock = yf.Ticker(ticker)
    dates = stock.options
    if not dates:
        return {}

    # choisir la date la plus proche de T_days
    if T_days is not None:
        maturity = min(dates, key=lambda x: abs(pd.to_datetime(x) - pd.Timestamp.today()).days - T_days)
    else:
        maturity = dates[0]

    chain = stock.option_chain(maturity)
    df = chain.calls if option_type.lower() == "Call" else chain.puts
    return {row['strike']: row['lastPrice'] for _, row in df.iterrows()}


def generate_vol_curve(S, T, r, q, market_prices, option_type, strikes=None):
    """
    Génère la courbe de volatilité implicite pour plusieurs strikes
    """
    if strikes is None:
        strikes = np.array(sorted(market_prices.keys()))
    
    vols = np.array([
        implied_volatility(S, K, T, r, q, market_prices[K], option_type)
        for K in strikes
    ])
    return strikes, vols


def plot_vol_curve(strikes, vols, strike_point=None, vol_point=None, title="Courbe de volatilité implicite"):
    """
    Trace la courbe de volatilité implicite
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12,6))

    # filtrer les vols valides
    strikes_plot = [s for s, v in zip(strikes, vols) if v is not None]
    vols_plot = [v for v in vols if v is not None]

    ax.plot(strikes_plot, vols_plot, marker="o", color="orange", lw=2)

    if strike_point is not None and vol_point is not None:
        if vol_point is not None:
            ax.scatter(strike_point, vol_point, color="black", s=100, zorder=5, label=f"Strike choisi: {strike_point}")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Volatilité implicite")
    ax.set_title(title)
    ax.grid(True)

    if vols_plot:
        ax.set_ylim(0, max(vols_plot)*1.2)
    if strikes_plot:
        buffer = (max(strikes_plot) - min(strikes_plot)) * 0.1
        ax.set_xlim(min(strikes_plot)-buffer, max(strikes_plot)+buffer)

    ax.legend()
    return fig
