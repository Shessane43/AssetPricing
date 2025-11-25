import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from functions.vol_function import VolFunction

def vol():
    st.header("Volatility Smile")

    required = ["S", "T", "r", "q", "option_type", "C_market", "C_BS"]
    if not all(k in st.session_state for k in required):
        st.info("Complétez d'abord Accueil & Pricing.")
        return

    S = st.session_state.S
    T = st.session_state.T
    r = st.session_state.r
    q = st.session_state.q
    option_type = st.session_state.option_type
    C_market = st.session_state.C_market
    C_BS = st.session_state.C_BS

    if option_type not in ["Call", "Put"]:
        st.warning("Uniquement Call ou Put.")
        return

    # Range de strikes pour tracer le smile
    K_range = np.linspace(0.5*S, 1.5*S, 40)

    # calcul des implied vol
    iv = [VolFunction.implied_vol(C_market, C_BS, S, K, T, r, q)for K in K_range]

    # plot clean dark/orange
    fig, ax = plt.subplots(facecolor="black")
    ax.plot(K_range, iv, color="orange", lw=2)

    ax.set_facecolor("black")
    ax.set_xlabel("Strike (K)", color="orange")
    ax.set_ylabel("Implied Vol", color="orange")
    ax.grid(True, linestyle="--", alpha=0.3, color="grey")
    ax.tick_params(colors="orange")

    st.pyplot(fig)
