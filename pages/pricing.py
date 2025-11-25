import streamlit as st
from functions.pricing_function import price_option, MODELS

def init_session_state():
    for key, value in {
        "ticker": None, "S": None, "K": None, "T": None, "r": None, "sigma": None,
        "q": None, "option_class": None, "option_type": None, "buy_sell": None,
        "v0": None, "kappa": None, "theta": None, "sigma_v": None, "rho": None,
        "model_name": None
    }.items():
        st.session_state.setdefault(key, value)

def app():
    st.title("Pricing des options")

    if not all(k in st.session_state for k in [
        "S", "K", "r", "sigma", "T", "q", "option_type", 
        "option_class", "buy_sell"
    ]):
        st.error("Paramètres manquants. Retournez à l'accueil.")
        return

    st.subheader("Paramètres sélectionnés")
    st.write(f"Spot (S) : {st.session_state['S']}")
    st.write(f"Strike (K) : {st.session_state['K']}")
    st.write(f"Taux sans risque (r) : {st.session_state['r']}")
    st.write(f"Volatilité (σ) : {st.session_state['sigma']}")
    st.write(f"Maturité (T) : {st.session_state['T']}")
    st.write(f"Dividendes (q) : {st.session_state['q']}")
    st.write(f"Type d'option : {st.session_state['option_type']}")
    st.write(f"Classe : {st.session_state['option_class']}")
    st.write(f"Position : {st.session_state['buy_sell']}")

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
