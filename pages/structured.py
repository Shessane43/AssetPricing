import streamlit as st
from functions.structured_function import (
    straddle_price, strangle_price, bull_spread_price, bear_spread_price,
    plot_structured_payoff, butterfly_spread_price, collar_price
)

def app():

     # Check if all required parameters exist
    if not all(k in st.session_state for k in [
        "S", "K", "r", "sigma", "T", "q", "option_type", 
        "option_class", "buy_sell"
    ]):
        st.error("Missing parameters. Please go back to the Parameters page.")
        return

    # Retrieve parameters from session state
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
            **Maturity (T)**: {T} year(s)

            **Risk-free rate (r)**: {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Dividend (q)**: {q:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            """
        )

    # --- Structured product selection ---
    st.subheader("Select the type of structured product")
    product_type = st.selectbox(
        "Structured Product",
        ["Straddle", "Strangle", "Bull Spread", "Bear Spread", "Butterfly Spread", "Collar"]
    )

    # --- Parameters depending on the product ---
    if product_type == "Straddle":
        st.write("Long Call + Short Put on the same strike and maturity. Bets on high volatility regardless of market direction.")
        K = st.number_input("Strike", value=S)
        sigma = st.number_input("Volatility", value=0.2)

    elif product_type == "Strangle":
        st.write("Long Call + Short Put with different strikes. Profits from a large move in the underlying while costing less than a straddle.")
        K_put = st.number_input("Strike Put", value=S*0.95)
        K_call = st.number_input("Strike Call", value=S*1.05)
        sigma_put = st.number_input("Volatility Put", value=0.2)
        sigma_call = st.number_input("Volatility Call", value=0.2)

    elif product_type == "Bull Spread":
        st.write("Long Call low strike + Short Call high strike. Bets on a moderate rise in the underlying with limited risk.")
        K_low = st.number_input("Strike Long Call", value=S*0.95)
        K_high = st.number_input("Strike Short Call", value=S*1.05)
        sigma = st.number_input("Volatility", value=0.2)

    elif product_type == "Bear Spread":
        st.write("Long Put high strike + Short Put low strike. Bets on a moderate decline in the underlying with limited risk.")
        K_high = st.number_input("Strike Long Put", value=S*1.05)
        K_low = st.number_input("Strike Short Put", value=S*0.95)
        sigma = st.number_input("Volatility", value=0.2)

    elif product_type == "Butterfly Spread":
        st.write("Long Call low strike + Short 2 Calls middle strike + Long Call high strike. Bets on a small movement around the middle strike.")
        K_low = st.number_input("Strike Low", value=S*0.95)
        K_mid = st.number_input("Strike Middle", value=S)
        K_high = st.number_input("Strike High", value=S*1.05)
        sigma = st.number_input("Volatility", value=0.2)
        
    elif product_type == "Collar":
        st.write("Long Put low strike + Short Call high strike. Limits losses while capping potential gains.")
        K_put = st.number_input("Strike Put (protection)", value=S*0.95)
        K_call = st.number_input("Strike Call (cap)", value=S*1.05)
        sigma_put = st.number_input("Volatility Put", value=0.2)
        sigma_call = st.number_input("Volatility Call", value=0.2)
        
    # --- Price calculation ---
    if st.button("Calculate structured product price"):
        try:
            if product_type == "Straddle":
                total_price = straddle_price(S, K, T, r, q, sigma)
            elif product_type == "Strangle":
                total_price = strangle_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call)
            elif product_type == "Bull Spread":
                total_price = bull_spread_price(S, K_low, K_high, T, r, q, sigma)
            elif product_type == "Bear Spread":
                total_price = bear_spread_price(S, K_high, K_low, T, r, q, sigma)
            elif product_type == "Butterfly Spread":
                total_price = butterfly_spread_price(S, K_low, K_mid, K_high, T, r, q, sigma)
            elif product_type == "Collar":
                st.info(
                    "WARNING: The price of a collar can be negative if the premium of the sold Call exceeds "
                    "the premium of the purchased Put. This reflects that you receive net cash to create the position."
                )
                total_price = collar_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call)

            st.success(f"Total price for {product_type}: **{total_price:.4f}**")
        except Exception as e:
            st.error(f"Error: {e}")

    # --- Display payoff ---
    if st.button("Show payoff"):
        products = []
        if product_type == "Straddle":
            products.append({"option_type": "Call", "K": K, "weight": 1})
            products.append({"option_type": "Put", "K": K, "weight": 1})
        elif product_type == "Strangle":
            products.append({"option_type": "Call", "K": K_call, "weight": 1})
            products.append({"option_type": "Put", "K": K_put, "weight": 1})
        elif product_type == "Bull Spread":
            products.append({"option_type": "Call", "K": K_low, "weight": 1})
            products.append({"option_type": "Call", "K": K_high, "weight": -1})
        elif product_type == "Bear Spread":
            products.append({"option_type": "Put", "K": K_high, "weight": 1})
            products.append({"option_type": "Put", "K": K_low, "weight": -1})
        elif product_type == "Butterfly Spread":
            products.append({"option_type": "Call", "K": K_low, "weight": 1, "sigma": sigma})
            products.append({"option_type": "Call", "K": K_mid, "weight": -2, "sigma": sigma})
            products.append({"option_type": "Call", "K": K_high, "weight": 1, "sigma": sigma})
        elif product_type == "Collar":
            products.append({"option_type": "Put", "K": K_put, "weight": 1, "sigma": sigma_put})
            products.append({"option_type": "Call", "K": K_call, "weight": -1, "sigma": sigma_call})

        fig = plot_structured_payoff(products, S)
        st.pyplot(fig)
