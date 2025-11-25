import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def implied_volatility(S, K, T, r, q, market_price, option_type, tol=1e-6, max_iter=100):
    """
    Calcule la volatilité implicite d'une option vanille européenne par Newton-Raphson.
    """
    sigma = 0.2  # valeur initiale

    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "call":
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


def generate_vol_curve(S, T, r, q, market_prices, option_type, strikes=None):
    """
    Génère la courbe de volatilité implicite pour plusieurs strikes.

    Parameters:
    - S, T, r, q : paramètres de l'option
    - market_prices : dict {K: prix_option}
    - option_type : "Call" ou "Put"
    - strikes : liste de strikes à calculer (facultatif)

    Returns:
    - strikes : array des strikes
    - vols : array des vols implicites correspondantes
    """
    if strikes is None:
        strikes = np.array(list(market_prices.keys()))
    
    vols = np.array([implied_volatility(S, K, T, r, q, market_prices[K], option_type) for K in strikes])
    return strikes, vols


def plot_vol_curve(strikes, vols, strike_point=None, vol_point=None, title="Courbe de volatilité implicite"):
    """
    Trace la courbe de volatilité implicite.
    Optionnel : afficher un point particulier (strike_point, vol_point)
    """
    plt.style.use("ggplot")  # ou "bmh"
    fig, ax = plt.subplots()
    ax.plot(strikes, vols, marker="o", color="orange", lw=2)
    
    if strike_point is not None and vol_point is not None:
        ax.scatter(strike_point, vol_point, color="black", s=100, zorder=5, label=f"Strike choisi: {strike_point}")
    
    ax.set_xlabel("Strike")
    ax.set_ylabel("Vol implicite")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig
