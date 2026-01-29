import streamlit as st
import numpy as np
from functions.parameters_function import (
    ALL_TICKERS,
    MarketDataFetcher,
    OptionParameters,
    PayoffCalculator,
    PayoffPlotter
)

def app():
    st.write(
        "Select a ticker, set the parameters, and visualize the payoff. "
        "These parameters will be used for pricing, Greeks, the volatility surface and structured products pricing."
    )

    # -------- Session defaults --------
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"
    if "K" not in st.session_state:
        st.session_state["K"] = MarketDataFetcher.get_last_price("AAPL")
    if "S" not in st.session_state:
        st.session_state["S"] = MarketDataFetcher.get_last_price(st.session_state.ticker)
    if "T" not in st.session_state:
        st.session_state["T"] = 1.0
    if "r" not in st.session_state:
        st.session_state["r"] = 0.02
    if "sigma" not in st.session_state:
        st.session_state["sigma"] = 0.2
    if "q" not in st.session_state:
        st.session_state["q"] = 0.0
    if "buy_sell" not in st.session_state:
        st.session_state["buy_sell"] = "Long"
    if "option_class" not in st.session_state:
        st.session_state["option_class"] = "Vanilla"
    if "option_type" not in st.session_state:
        st.session_state["option_type"] = "call"

    # -------- Ticker --------
    current_ticker = st.session_state.ticker
    ticker_index = ALL_TICKERS.index(current_ticker) if current_ticker in ALL_TICKERS else 0

    selected_ticker = st.selectbox(
        "Select a stock",
        ALL_TICKERS,
        index=ticker_index,
        key="ticker_selectbox"
    )

    if selected_ticker != st.session_state.ticker:
        st.session_state.ticker = selected_ticker
        st.session_state.S = MarketDataFetcher.get_last_price(selected_ticker)

    st.info(f"Current price of {st.session_state.ticker}: ${st.session_state.S:.2f}")

    # -------- Position --------
    st.session_state.buy_sell = st.selectbox(
        "Position",
        ["Long", "Short"],
        index=["Long", "Short"].index(st.session_state.buy_sell),
        key="buy_sell_selectbox"
    )

    # -------- Option class --------
    st.session_state.option_class = st.selectbox(
        "Option class",
        ["Vanilla", "Exotic"],
        index=["Vanilla", "Exotic"].index(st.session_state.option_class),
        key="option_class_selectbox"
    )

    # -------- Option type (CLEAN MAPPING) --------
    if st.session_state.option_class == "Vanilla":
        cp = st.selectbox(
            "Option type",
            ["Call", "Put"],
            key="vanilla_cp"
        )
        st.session_state.option_type = cp.lower()

    else:
        exotic_family = st.selectbox(
            "Exotic type",
            ["Asian", "Lookback"],
            key="exotic_family"
        )
        exotic_cp = st.selectbox(
            "Direction",
            ["Call", "Put"],
            key="exotic_cp"
        )

        # 🔥 mapping EXACT attendu par Heston / pricing engine
        st.session_state.option_type = f"{exotic_family.lower()}_{exotic_cp.lower()}"

    # -------- Numeric parameters --------
    st.session_state.K = st.number_input("Strike (K)", value=st.session_state.K)
    st.session_state.T = st.number_input("Maturity (T in years)", value=st.session_state.T)
    st.session_state.r = st.number_input("Risk-free rate (r)", value=st.session_state.r)
    st.session_state.sigma = st.number_input(
        "Volatility (σ)", value=st.session_state.sigma, min_value=0.0, step=0.01, format="%.4f"
    )
    st.session_state.q = st.number_input(
        "Dividend (q)", value=st.session_state.q, min_value=0.0, step=0.001, format="%.4f"
    )

    # -------- Parameters object --------
    params = OptionParameters(
        ticker=st.session_state.ticker,
        S=st.session_state.S,
        K=st.session_state.K,
        T=st.session_state.T,
        r=st.session_state.r,
        sigma=st.session_state.sigma,
        q=st.session_state.q,
        option_class=st.session_state.option_class,
        option_type=st.session_state.option_type,   # ✅ engine-ready
        buy_sell=st.session_state.buy_sell
    )

    # -------- Payoff --------
    S_range, payoff = PayoffCalculator.choose_payoff(params)

    label = st.session_state.option_type.replace("_", " ").title()
    fig = PayoffPlotter.plot(
        S_range,
        payoff,
        st.session_state.K,
        f"{st.session_state.buy_sell} {label}"
    )

    st.pyplot(fig)

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
