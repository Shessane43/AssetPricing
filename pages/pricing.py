import streamlit as st
from functions.pricing_function import price_option, MODELS

def app():

    if not all(k in st.session_state for k in [
        "S", "K", "r", "sigma", "T", "q", "option_type", 
        "option_class", "buy_sell"
    ]):
        st.error("Paramètres manquants. Retournez à l'accueil.")
        return

    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    sigma = st.session_state.get("sigma")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")

    with st.container(border=True):
        st.markdown(
            f"""
            **Ticker** : **{ticker}**

            **Spot (S)** : {S:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Strike (K)** : {K:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Maturité (T)** : {T} an(s)


            **Taux sans risque (r)** : {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Volatilité (σ)** : {sigma:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Dividendes (q)** : {q:.2%} 

            **Type d'option** : **{option_type}**
            """
        )

            
    st.subheader("Choix du modèle")
    model_name = st.selectbox("Modèle de pricing", list(MODELS.keys()))
    st.session_state["model_name"] = model_name

    if model_name == "Heston":
        st.subheader("Paramètres du modèle de Heston")
        st.session_state.v0 = st.number_input("Volatilité initiale (v0)", value=0.04)
        st.session_state.kappa = st.number_input("Retour à la moyenne (κ)", value=2.0)
        st.session_state.theta = st.number_input("Vol long-terme (θ)", value=0.04)
        st.session_state.sigma_v = st.number_input("Volatilité de la variance (σ_v)", value=0.3)
        st.session_state.rho = st.number_input("Corrélation (ρ)", value=-0.5)

    if st.button("Calculer le prix"):
        params = {
            "S": st.session_state["S"],
            "K": st.session_state["K"],
            "r": st.session_state["r"],
            "sigma": st.session_state["sigma"],
            "T": st.session_state["T"],
            "q": st.session_state["q"],
            "option_type": st.session_state["option_type"],
            "buy_sell": st.session_state["buy_sell"],
            "option_class": st.session_state["option_class"]
        }

        if model_name == "Heston":
            params.update({
                "v0": st.session_state["v0"],
                "kappa": st.session_state["kappa"],
                "theta": st.session_state["theta"],
                "sigma_v": st.session_state["sigma_v"],
                "rho": st.session_state["rho"]
            })

        try:
            price = price_option(model_name, params)
            st.success(f"Prix ({model_name}) : **{price:.4f}**")
        except Exception as e:
            st.error(f"Erreur : {e}")
