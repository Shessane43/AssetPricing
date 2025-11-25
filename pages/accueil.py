import streamlit as st
import numpy as np
from functions.accueil_function import (
    ALL_TICKERS,
    MarketDataFetcher,
    OptionParameters,
    PayoffCalculator,
    PayoffPlotter
)

def app():

    st.title("Asset Pricing & Option Greeks")
    st.subheader("Bienvenue dans l'application")
    st.markdown("---")

    st.write("Choisissez le ticker, les paramètres, puis visualisez le payoff.")

    st.session_state.ticker = st.selectbox("Sélectionnez l'action", ALL_TICKERS)

    last_price = MarketDataFetcher.get_last_price(st.session_state.ticker)
    st.session_state["S"]= last_price
    st.info(f"Prix actuel de {st.session_state.ticker} : {last_price:.2f} $")

    st.session_state["buy_sell"] = st.selectbox("Position", ["Buy", "Sell"])

    st.session_state["option_class"] = st.selectbox("Classe d’option", ["Vanille", "Exotique"])

    if st.session_state.option_class == "Vanille":
        st.session_state["option_type"] = st.selectbox("Type d’option", ["Call", "Put"])
    else:
        st.session_state["option_type"] = st.selectbox("Type exotique", ["Asian", "Lookback"])

    st.session_state["K"] = st.number_input("Strike (K)", value=last_price)
    st.session_state["T"] = st.number_input("Maturité (T années)", value=1.0)
    st.session_state["r"] = st.number_input("Taux sans risque (r)", value=0.02)
    st.session_state["sigma"] = st.number_input("Volatilité (σ)", value=0.2)
    st.session_state["q"]= st.number_input("Dividende (q)", value=0.0)

    params = OptionParameters(
        ticker=st.session_state["ticker"],
        S=st.session_state["S"],
        K=st.session_state["K"],
        T=st.session_state["T"],
        r=st.session_state["r"],
        sigma=st.session_state["sigma"],
        q=st.session_state["q"],
        option_class=st.session_state["option_class"],
        option_type=st.session_state["option_type"],
        buy_sell=st.session_state["buy_sell"])
    
    S_range, payoff = PayoffCalculator.choose_payoff(params)
    fig = PayoffPlotter.plot(
        S_range,
        payoff,
        st.session_state.K,
        f"{st.session_state.buy_sell} {st.session_state.option_type}"
    )

    st.pyplot(fig)
