import streamlit as st
from functions.vol_function import (
    get_market_prices_yahoo,
    generate_vol_curve,
    plot_vol_curve,
    get_all_option_maturities,
    generate_vol_curves_multiple_maturities,
    plot_vol_surface
)

def app():

    required_keys = ["ticker","S", "K","r", "q", "T"]
    if not all(k in st.session_state for k in required_keys):
        st.error("Missing parameters. Please return to the Parameters page.")
        return

    # ---------- 1. Fetch parameters from session ----------
    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")

    with st.container():
        st.markdown(
            f"""
            **Ticker**: **{ticker}**

            **Spot (S)**: {S:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Strike (K)**: {K:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Maturity (T)**: {T} year(s)

            **Risk-free rate (r)**: {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Dividend (q)**: {q:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Option type**: **{option_type}**
            """
        )

    # Fetch market prices
    market_prices_calls, market_prices_puts, maturity = get_market_prices_yahoo(ticker, T_days=int(T*365))
    if not market_prices_calls and not market_prices_puts:
        st.error("Unable to fetch market prices for this ticker/maturity. Check the ticker or your connection.")
        return

    # It's important to take the maturity of the actual market product
    strikes, vols = generate_vol_curve(S, maturity, r, q, market_prices_calls, market_prices_puts, option_type)

    # Plot the implied volatility curve
    fig = plot_vol_curve(
        strikes,
        vols,
        maturity=maturity,
        K=K,
    )
    st.subheader("Implied Volatility Curve")
    st.pyplot(fig)

    # Plot the implied volatility surface
    all_maturities = get_all_option_maturities(ticker)
    vol_curves = generate_vol_curves_multiple_maturities(S, all_maturities, r, q, option_type, ticker)
    fig_3d = plot_vol_surface(vol_curves)
    st.subheader("Implied Volatility Surface")
    st.plotly_chart(fig_3d, use_container_width=True)
