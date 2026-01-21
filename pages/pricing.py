import streamlit as st
from functions.pricing_function import price_option, MODELS

def app():

    # Check if all required parameters exist in session_state
    required_keys = [
        "S", "K", "r", "sigma", "T", "q", "option_type", 
        "option_class", "buy_sell"
    ]
    if not all(k in st.session_state for k in required_keys):
        st.error("Missing parameters. Please return to the home page.")
        return

    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    sigma = st.session_state.get("sigma")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")

    with st.container():
        st.markdown(
            f"""
            **Ticker**: **{ticker}**

            **Spot (S)**: {S:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Strike (K)**: {K:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Maturity (T)**: {T} year(s)

            **Risk-free rate (r)**: {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Volatility (σ)**: {sigma:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Dividend (q)**: {q:.2%}

            **Option type**: **{option_type}**
            """
        )

    st.subheader("Select Pricing Model")
    model_name = st.selectbox("Pricing Model", list(MODELS.keys()))
    st.session_state["model_name"] = model_name

    # Heston model parameters
    if model_name == "Heston":
        st.subheader("Heston Model Parameters")
        st.session_state.v0 = st.number_input("Initial variance (v0)", value=0.04)
        st.session_state.kappa = st.number_input("Mean reversion (κ)", value=2.0)
        st.session_state.theta = st.number_input("Long-term variance (θ)", value=0.04)
        st.session_state.sigma_v = st.number_input("Variance volatility (σ_v)", value=0.3)
        st.session_state.rho = st.number_input("Correlation (ρ)", value=-0.5)

    # Compute price button
    if st.button("Compute Price"):
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
            st.success(f"Price ({model_name}): **{price:.4f}**")
        except Exception as e:
            st.error(f"Error: {e}")
