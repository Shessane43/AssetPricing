import streamlit as st
import numpy as np
import seaborn
from functions.vol_function import implied_volatility, generate_vol_curve, plot_vol_curve

def app():
    st.title("Volatilité implicite")

    # ---------- 1. Récupérer les paramètres depuis la session ----------
    S = float(st.session_state.get("S", 0))  # convertit en float
    K = float(st.session_state.get("K", 0))
    T = float(st.session_state.get("T", 1))
    r = float(st.session_state.get("r", 0.02))
    q = float(st.session_state.get("q", 0.0))
    option_type = st.session_state["option_type"]  
    buy_sell= st.session_state["buy_sell"]
    sigma_calibree = st.session_state.get("sigma", 0.2)

    if S is None or K is None:
        st.error("Les paramètres ne sont pas initialisés. Veuillez d'abord choisir les paramètres dans l'accueil.")
        return

    # ---------- 2. Définir un range de strikes ----------
    strikes_range = np.arange(S*0.8, S*1.2+1, 5)  # pas d'int()
    market_prices = {float(K): S for K in strikes_range}
    # ---------- 3. Simuler les prix de marché pour chaque strike ----------
    market_prices = {K: S for K in strikes_range}  # ici on peut juste mettre le spot ou un prix théorique

    # ---------- 4. Calculer la courbe de volatilité implicite ----------
    strikes, vols = generate_vol_curve(S, T, r, q, market_prices, option_type)

    # ---------- 5. Calculer le vol implicite du strike choisi ----------
    vol_point = implied_volatility(S=S,K=K,
    T=T,
    r=r,
    q=q,
    market_price=market_prices[min(market_prices.keys(), key=lambda x: abs(x-K))],  # prendre le plus proche si pas exact
    option_type=option_type
)
    # ---------- 6. Tracer la courbe ----------
    fig = plot_vol_curve(strikes, vols, strike_point=K, vol_point=vol_point, title=f"Volatilité implicite ({option_type})")
    st.pyplot(fig)

    # ---------- 7. Afficher la vol implicite du strike choisi ----------
    st.write(f"Volatilité implicite pour le strike choisi {K}: **{vol_point:.4f}**")
