import streamlit as st
from functions.pricing_function import price_option, MODELS


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


    if model_name == "Heston":
        st.subheader("Heston Parameters")

        st.session_state["v0"] = st.number_input(
            "Initial variance v₀", value=st.session_state.get("v0", 0.04)
        )
        st.session_state["kappa"] = st.number_input(
            "Mean reversion κ", value=st.session_state.get("kappa", 2.0)
        )
        st.session_state["theta"] = st.number_input(
            "Long-term variance θ", value=st.session_state.get("theta", 0.04)
        )
        st.session_state["sigma_v"] = st.number_input(
            "Vol-of-vol σᵥ", value=st.session_state.get("sigma_v", 0.30)
        )
        st.session_state["rho"] = st.number_input(
            "Correlation ρ", value=st.session_state.get("rho", -0.5)
        )

 
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

        try:
            price = price_option(model_name, params)
            st.success(f"Price ({model_name}): **{price:.4f}**")
        except Exception as e:
            st.error(f"Pricing error: {e}")
