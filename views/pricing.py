import streamlit as st
from functions.pricing_function import price_option, MODELS
from functions.model_explanations import MODEL_EXPLANATIONS


def app():

    required_keys = [
        "S", "K", "r", "sigma", "T", "q",
        "option_type", "option_class", "buy_sell"
    ]

    if not all(k in st.session_state for k in required_keys):
        st.error("Missing parameters. Please go back to the Parameters page.")
        return

    ticker = st.session_state.get("ticker", "N/A")
    S = st.session_state["S"]
    K = st.session_state["K"]
    T = st.session_state["T"]
    r = st.session_state["r"]
    sigma = st.session_state["sigma"]
    q = st.session_state["q"]
    option_type = st.session_state["option_type"]
    buy_sell = st.session_state["buy_sell"]

    st.markdown(
        f"""
        **Ticker**: **{ticker}**

        **Spot (S)**: {S:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Strike (K)**: {K:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Maturity (T)**: {T} year(s)

        **Risk-free rate (r)**: {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Volatility (σ)**: {sigma:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Dividend yield (q)**: {q:.2%}

        **Option type**: **{option_type}**  
        **Position**: **{buy_sell}**
        """
    )

    st.subheader("Pricing Model")
    model_name = st.selectbox("Select model", list(MODELS.keys()))
    st.session_state["model_name"] = model_name

    show_explanation = st.checkbox("Show model explanation", value=False)
    if show_explanation and model_name in MODEL_EXPLANATIONS:
        st.markdown("---")
        st.markdown(MODEL_EXPLANATIONS[model_name], unsafe_allow_html=False)



    if model_name == "Heston":
        st.subheader("Heston Parameters")
        st.number_input("Initial variance v₀", value=0.04, min_value=1e-4, key="v0")
        st.number_input("Mean reversion κ", value=1.5, min_value=1e-4, key="kappa")
        st.number_input("Long-term variance θ", value=0.04, min_value=1e-4, key="theta")
        st.number_input("Vol-of-vol σᵥ", value=0.30, min_value=1e-4, key="sigma_v")
        st.number_input("Correlation ρ", value=-0.7, min_value=-0.95, max_value=0.95, key="rho")

    if model_name == "Gamma Variance":
        st.subheader("Variance Gamma Parameters")
        st.number_input("Theta (VG)", value=0.0, step=0.05, key="theta")
        st.number_input("Nu (VG)", min_value=1e-4, value=0.20, step=0.05, key="nu")

    if model_name == "Trinomial Tree":
        st.subheader("Trinomial Tree Parameters")
        st.slider(
            "Number of steps",
            min_value=50,
            max_value=500,
            step=10,
            key="n_steps",
        )
        st.radio(
            "Exercise type",
            ["European", "American"],
            horizontal=True,
            key="exercise",
        )

    if model_name == "Merton Jump Diffusion":
        st.subheader("Merton Jump Diffusion Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Jump intensity λ", value=0.2, min_value=0.0, key="lambd")
            st.number_input("Jump mean μ_J", value=-0.1, key="mu_j")
        with col2:
            st.number_input("Jump volatility σ_J", value=0.3, min_value=1e-4, key="sigma_j")



    if model_name in [
        "Black-Scholes", "Bachelier",
        "Gamma Variance", "Merton Jump Diffusion", "Trinomial Tree"
    ] and option_type not in ["call", "put"]:
        st.warning(f"{model_name} only supports vanilla call / put options.")
        return

  

    if st.button("Compute Price"):

        params = {
            "S": S,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "q": q,
            "option_type": option_type,
            "buy_sell": buy_sell,
            "option_class": st.session_state["option_class"],
        }

        if model_name == "Heston":
            params.update({
                "v0": st.session_state["v0"],
                "kappa": st.session_state["kappa"],
                "theta": st.session_state["theta"],
                "sigma_v": st.session_state["sigma_v"],
                "rho": st.session_state["rho"],
            })

        elif model_name == "Gamma Variance":
            params.update({
                "theta": st.session_state["theta"],
                "nu": st.session_state["nu"],
            })

        elif model_name == "Trinomial Tree":
            params.update({
                "n_steps": st.session_state["n_steps"],
                "exercise": st.session_state["exercise"].lower(),
            })

        elif model_name == "Merton Jump Diffusion":
            params.update({
                "lambd": st.session_state["lambd"],
                "mu_j": st.session_state["mu_j"],
                "sigma_j": st.session_state["sigma_j"],
            })

        try:
            price = price_option(model_name, params)
            st.success(f"Price ({model_name}): **{price:.4f}**")
        except Exception as e:
            st.error(f"Pricing error: {e}")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
