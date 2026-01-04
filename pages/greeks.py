# pages/greeks.py
import streamlit as st
from Models.blackscholes import BlackScholes
from functions.greeks_function import Greeks

def app():

    # Vérification des paramètres
    required = ["S","K","T","r","sigma","option_type","buy_sell","model_name","option_class"]
    if not all(k in st.session_state for k in required):
        st.error("Certains paramètres manquent dans la session.")
        return

    try:
        S = float(st.session_state["S"])
        K = float(st.session_state["K"])
        T = float(st.session_state["T"])
        r = float(st.session_state["r"])
        sigma = float(st.session_state["sigma"])
        q = float(st.session_state.get("q",0))
    except:
        st.error("Erreur : S, K, T, r, sigma et q doivent être des nombres.")
        return

    option_type = str(st.session_state["option_type"])
    buy_sell = str(st.session_state["buy_sell"])
    option_class = str(st.session_state["option_class"])
    model_name = str(st.session_state["model_name"])

    st.subheader(f"Modèle choisi : {model_name}")

    
    greeks = Greeks(option_type, model_name, S, K, T, r, sigma=sigma, buy_sell=buy_sell)
    fig = greeks.plot_all_greeks()
    st.pyplot(fig)

    
