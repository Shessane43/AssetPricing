import numpy as np
from scipy.stats import norm
import yfinance as yf
import streamlit as st


class VolFunction:
    """Outils liés à la volatilité implicite & smile."""

    @staticmethod
    def bs_vega(S, K, T, r, sigma, q=0.0):
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

    def get_market_price():
        """Récupère le prix marché du ticker en session state"""
        ticker = st.session_state.get('ticker', None)
        if ticker is None:
            return None
        try:
            data = yf.Ticker(ticker).history(period="1d")
            return data['Close'].iloc[-1]
        except:
            return None

        
    @staticmethod
    def implied_vol(C_BS, S, K, T, r, q=0.0,sigma_init=0.2, tol=1e-6, max_iter=100): 
        """
        Newton-Raphson sur la différence (BS - marché), C_BS doit être fourni par la page Pricing.
        """
        C_market = VolFunction.get_market_price()
        sigma = sigma_init
        for _ in range(max_iter):
            vega = VolFunction.bs_vega(S, K, T, r, sigma, q)
            if vega == 0:
                break
            sigma -= (C_BS - C_market) / vega
            if abs(C_BS - C_market) < tol:
                break
        return max(sigma, 0.0001)  # sécurité numérique
