import streamlit as st
from functions.pricing_function import price_option, MODELS

def init_session_state():
    for key, value in {
        "ticker": None,
        "S": None,
        "K": None,
        "T": None,
        "r": None,
        "sigma": None,
        "q": None,
        "option_class": None,
        "option_type": None,
        "buy_sell": None
    }.items():
        st.session_state.setdefault(key, value)

def app():
    st.title("Pricing des options")

    st.subheader("Paramètres sélectionnés")

    if not all(k in st.session_state for k in ["S", "K", "r", "sigma", "T", "option_type","buy_sell"]):
        st.error("Les paramètres ne sont pas présents dans session_state. Veuillez retourner à l'accueil.")
        return

    st.write(f"**Spot (S)** : {st.session_state['S']}")
    st.write(f"**Strike (K)** : {st.session_state['K']}")
    st.write(f"**Taux sans risque (r)** : {st.session_state['r']}")
    st.write(f"**Volatilité (σ)** : {st.session_state['sigma']}")
    st.write(f"**Maturité (T)** : {st.session_state['T']}")
    st.write(f"**Dividendes (q)** : {st.session_state['q']}")
    st.write(f"**Type d'option** : {st.session_state['option_type']}")
    st.write(f"**Position** : {st.session_state['buy_sell']}")


    st.subheader("Choix du modèle")
    model_name = st.selectbox("Modèle de pricing", list(MODELS.keys()))

    if st.button("Calculer le prix"):
        params = {
            "S": st.session_state["S"],
            "K": st.session_state["K"],
            "r": st.session_state["r"],
            "sigma": st.session_state["sigma"],
            "T": st.session_state["T"],
            "option_type": st.session_state["option_type"],  
            "q": st.session_state["q"],               
            "buy_sell": st.session_state["buy_sell"] 
         }


        price = price_option(model_name, params)
        st.success(f"Prix ({model_name}) : **{price:.4f}**")

