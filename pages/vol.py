import streamlit as st
from functions.vol_function import (
    get_market_prices_yahoo,
    generate_vol_curve,
    plot_vol_curve,
    implied_volatility
)

def app():
    st.title("Volatilité implicite")

    # ---------- 1. Récupérer les paramètres depuis la session ----------
    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")

    st.write(f"Ticker choisi : {ticker}")
    st.write(f"Spot (S) : {S}, Strike (K) : {K}, T : {T} an(s), r : {r}, q : {q}, type : {option_type}")

    # ---------- 2. Récupérer les prix de marché ----------
    market_prices = get_market_prices_yahoo(ticker, option_type, T_days=int(T*365))
    if not market_prices:
        st.error("Impossible de récupérer les prix de marché pour ce ticker/maturité. Vérifiez le ticker ou la connexion.")
        return

    # ---------- 3. Générer la courbe de volatilité implicite ----------
    strikes, vols = generate_vol_curve(S, T, r, q, market_prices, option_type)

    # ---------- 4. Calculer la vol implicite du strike choisi ----------
    closest_strike = min(market_prices.keys(), key=lambda x: abs(x - K))
    vol_point = implied_volatility(
        S=S,
        K=closest_strike,
        T=T,
        r=r,
        q=q,
        market_price=market_prices[closest_strike],
        option_type=option_type
    )

    # ---------- 5. Affichage graphique ----------
    fig = plot_vol_curve(
        strikes,
        vols,
        strike_point=closest_strike,
        vol_point=vol_point,
        title=f"Volatilité implicite ({option_type})"
    )
    st.pyplot(fig)

    # ---------- 6. Afficher la vol implicite du strike choisi ----------
    if vol_point is not None:
        st.success(f"Volatilité implicite pour le strike choisi {closest_strike}: **{vol_point:.4f}**")
    else:
        st.warning("Impossible de calculer la volatilité implicite pour le strike choisi.")
