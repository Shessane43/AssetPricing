# pages/greeks.py
import streamlit as st
from Models.blackscholes import BlackScholes
from functions.greeks_function import Greeks

def app():

    # Check required parameters
    required = ["ticker","S","K","T","r","sigma","option_type","buy_sell","model_name","option_class"]
    if not all(k in st.session_state for k in required):
        st.error("Some parameters are missing in the session.")
        return

    try:
        ticker = st.session_state["ticker"]
        S = float(st.session_state["S"])
        K = float(st.session_state["K"])
        T = float(st.session_state["T"])
        r = float(st.session_state["r"])
        sigma = float(st.session_state["sigma"])
        q = float(st.session_state.get("q",0))
    except:
        st.error("Error: S, K, T, r, sigma, and q must be numbers.")
        return

    option_type = str(st.session_state["option_type"])
    buy_sell = str(st.session_state["buy_sell"])
    option_class = str(st.session_state["option_class"])
    model_name = str(st.session_state["model_name"])

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
            **Position**: **{buy_sell}**
            """
        )

    st.subheader(f"Selected Model: {model_name}")

    # Compute and plot greeks
    greeks = Greeks(option_type, model_name, S, K, T, r, sigma=sigma, buy_sell=buy_sell)
    fig = greeks.plot_all_greeks()
    st.pyplot(fig)
