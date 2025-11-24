import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Vega ---
def bs_vega(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

# --- BS price ---
def bs_price(S, K, T, r, sigma, q=0.0, option_type="Call"):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "Call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# --- Implied vol via Newton-Raphson ---
def implied_vol(C_market,price_BS, S, K, T, r, q):
    sigma = 0.2
    tol, max_iter = 1e-6, 100
    for _ in range(max_iter):
        vega = bs_vega(S, K, T, r, sigma, q)
        if vega == 0: break
        sigma -= (price_BS - C_market)/vega
        if abs(price_BS - C_market) < tol: break
    return sigma

# --- Page Streamlit ---
def app():
    st.header("Volatility Smile (Call/Put)")

    # Vérification des paramètres
    required_keys = ['S', 'K', 'T', 'r', 'q', 'option_type', 'C_BS', 'ticker']
    if not all(k in st.session_state for k in required_keys):
        st.info("Remplissez d'abord les paramètres dans Accueil et Pricing.")
        return

    S = st.session_state.S
    T = st.session_state.T
    r = st.session_state.r
    q = st.session_state.q
    option_type = st.session_state.option_type
    C_BS = st.session_state.C_BS
    ticker = st.session_state.ticker

    if option_type not in ["Call", "Put"]:
        st.warning("Cette page est uniquement pour Call et Put.")
        return

    # --- Prix marché approximatif via Yahoo Finance ---
    try:
        spot_data = yf.Ticker(ticker).history(period="1d")
        C_market = spot_data['Close'].iloc[-1]
    except:
        st.warning("Impossible de récupérer le prix du marché, utilisation du spot comme approximation.")
        C_market = S

    # --- Range de strikes autour du spot ---
    K_range = np.linspace(0.5*S, 1.5*S, 50)
    sigma_range = [implied_vol(C_market, S, K_i, T, r, q, option_type) for K_i in K_range]

    # --- Plot dark/orange épuré ---
    fig, ax = plt.subplots(facecolor="black")
    ax.plot(K_range, sigma_range, color="orange", lw=2, label=f"{option_type} Smile")
    ax.set_facecolor("black")
    ax.grid(True, color="grey", linestyle="--", alpha=0.3)
    ax.set_xlabel("Strike (K)", color="orange")
    ax.set_ylabel("Volatilité implicite", color="orange")
    ax.tick_params(axis="x", colors="orange")
    ax.tick_params(axis="y", colors="orange")
    ax.legend(facecolor="black", edgecolor="orange", labelcolor="orange")
    st.pyplot(fig)
