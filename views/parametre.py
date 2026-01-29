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

    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"
    if "K" not in st.session_state:
        last_price = MarketDataFetcher.get_last_price("AAPL")
        st.session_state["K"] = last_price
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
        st.session_state["option_type"] = "Call"

    # Select ticker
    current_ticker = st.session_state.get("ticker", "AAPL")
    if current_ticker in ALL_TICKERS:
        ticker_index = ALL_TICKERS.index(current_ticker)
    else:
        ticker_index = 0

    selected_ticker = st.selectbox(
        "Select a stock",
        ALL_TICKERS,
        index=ticker_index,
        key="ticker_selectbox"
    )
    
    # Only update S if ticker changed
    if selected_ticker != st.session_state["ticker"]:
        st.session_state["ticker"] = selected_ticker
        st.session_state["S"] = MarketDataFetcher.get_last_price(selected_ticker)
    else:
        st.session_state["ticker"] = selected_ticker

    st.info(f"Current price of {st.session_state.ticker}: ${st.session_state['S']:.2f}")

    # Select position and option class with keys to preserve selection
    buy_sell_options = ["Long", "Short"]
    buy_sell_index = buy_sell_options.index(st.session_state["buy_sell"])
    st.session_state["buy_sell"] = st.selectbox(
        "Position", 
        buy_sell_options,
        index=buy_sell_index,
        key="buy_sell_selectbox"
    )

    option_class_options = ["Vanilla", "Exotic"]
    option_class_index = option_class_options.index(st.session_state["option_class"])
    st.session_state["option_class"] = st.selectbox(
        "Option class",
        option_class_options,
        index=option_class_index,
        key="option_class_selectbox"
    )

    if st.session_state.option_class == "Vanilla":
        option_type_options = ["Call", "Put"]
        if st.session_state["option_type"] not in option_type_options:
            st.session_state["option_type"] = "Call"
        option_type_index = option_type_options.index(st.session_state["option_type"])
        st.session_state["option_type"] = st.selectbox(
            "Option type",
            option_type_options,
            index=option_type_index,
            key="option_type_selectbox"
        )
    else:
        option_type_options = ["Asian", "Lookback"]
        if st.session_state["option_type"] not in option_type_options:
            st.session_state["option_type"] = "Asian"
        option_type_index = option_type_options.index(st.session_state["option_type"])
        st.session_state["option_type"] = st.selectbox(
            "Exotic type",
            option_type_options,
            index=option_type_index,
            key="option_type_exotic_selectbox"
        )

    st.session_state["K"] = st.number_input(
        "Strike (K)",
        value=st.session_state["K"],
        key="K_input"
    )
    st.session_state["T"] = st.number_input(
        "Maturity (T in years)",
        value=st.session_state["T"],
        key="T_input"
    )
    st.session_state["r"] = st.number_input(
        "Risk-free rate (r)",
        value=st.session_state["r"],
        key="r_input"
    )
    st.session_state["sigma"] = st.number_input(
        "Volatility (σ)",
        value=float(st.session_state["sigma"]),
        min_value=0.0,
        step=0.01,
        format="%.4f",
        key="sigma_input"
    )
    st.session_state["q"] = st.number_input(
        "Dividend (q)",
        value=float(st.session_state["q"]),
        min_value=0.0,
        step=0.001,
        format="%.4f",
        key="q_input"
    )

    params = OptionParameters(
        ticker=st.session_state["ticker"],
        S=st.session_state["S"],
        K=st.session_state["K"],
        T=st.session_state["T"],
        r=st.session_state["r"],
        sigma=st.session_state["sigma"],
        q=st.session_state["q"],
        option_class=st.session_state["option_class"],
        option_type=st.session_state["option_type"],
        buy_sell=st.session_state["buy_sell"]
    )

    S_range, payoff = PayoffCalculator.choose_payoff(params)

    fig = PayoffPlotter.plot(
        S_range,
        payoff,
        st.session_state.K,
        f"{st.session_state.buy_sell} {st.session_state.option_type}"
    )

    st.pyplot(fig)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()