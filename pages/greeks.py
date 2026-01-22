# pages/greeks.py
import streamlit as st
from Models.blackscholes import BlackScholes
from functions.greeks_function import Greeks
import pandas as pd

def app():

    # -------------------------
    # Check required parameters
    # -------------------------
    required_keys = [
        "S", "K", "r", "sigma", "T", "q", "option_type", 
        "option_class", "buy_sell"
    ]
    if not all(k in st.session_state for k in required_keys):
        st.error("Missing parameters. Please return to the Parameters page.")
        return

    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    sigma = st.session_state.get("sigma")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")
    buy_sell = str(st.session_state["buy_sell"])
    model_name = "Black-Scholes"

    # -------------------------
    # Display parameters
    # -------------------------
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

            **Option type**: **{option_type}** &nbsp;&nbsp;|&nbsp;&nbsp;
            **Position**: **{buy_sell}** &nbsp;&nbsp;|&nbsp;&nbsp;
            **Model**: **{model_name}**
            """
        )

    # -------------------------
    # Compute Greeks
    # -------------------------
    greeks = Greeks(option_type, model_name, S, K, T, r, sigma=sigma, buy_sell=buy_sell)

    # 1️⃣ Plot Greeks
    fig = greeks.plot_all_greeks()
    st.subheader("Greek Plots")
    st.pyplot(fig)

    # 2️⃣ Display exact Greek values
    greek_values = {
        "Delta": greeks.delta(),
        "Gamma": greeks.gamma(),
        "Vega": greeks.vega(),
        "Theta": greeks.theta(),
        "Rho": greeks.rho()
    }

    df_greeks = pd.DataFrame([greek_values]).T.rename(columns={0: "Value"})
    df_greeks["Value"] = df_greeks["Value"].apply(lambda x: round(x, 4))

    st.dataframe(
        df_greeks.style
            .set_properties(**{
                'background-color': '#1e1e1e',
                'color': 'orange',
                'border-color': 'orange',
                'font-size': '14px',
                'text-align': 'center'
            })
            .format("{:.4f}")
    )
