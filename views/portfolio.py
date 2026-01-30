# pages/portfolio.py
import streamlit as st
import pandas as pd

from functions.portfolio_function import (
    add_position,
    remove_position,
    calculate_prices_and_greeks,
    calculate_portfolio_greeks,
    calculate_portfolio_value,
    delta_hedge_portfolio
)
from functions.parameters_function import ALL_TICKERS
from functions.pricing_function import price_option

def app():
    # =========================
    # Initialization
    # =========================
    st.session_state.setdefault("portfolio", [])

    st.title("Portfolio Management")

    # =========================
    # Add a position
    # =========================
    with st.expander("Add a position"):
        position_type = st.selectbox("Instrument type", ["Option", "Stock"], key="position_type_new")
        ticker = st.selectbox("Ticker", ALL_TICKERS, key="ticker_new")
        qty = st.number_input("Quantity (negative for short)", value=1.0, step=0.0001, format="%.4f", key="qty_new")
        buy_sell_flag = "buy" if qty > 0 else "sell"

        # ---------- Stock ----------
        if position_type == "Stock":
            S = st.number_input("Spot price (S)", value=100.0, key="S_stock")
            if st.button("Add stock"):
                position = {
                    "instrument_type": "Stock",
                    "ticker": ticker,
                    "S": S,
                    "qty": qty,
                    "buy_sell": buy_sell_flag
                }
                st.session_state["portfolio"] = add_position(st.session_state["portfolio"], position)
                st.success(f"Stock {ticker} added")

        # ---------- Option ----------
        else:
            option_type = st.selectbox("Option type", ["Call", "Put"], key="option_type_new")
            S = st.number_input("Spot (S)", value=100.0, key="S_option")
            K = st.number_input("Strike (K)", value=100.0, key="K_option")
            T = st.number_input("Maturity (T, years)", value=1.0, key="T_option")
            r = st.number_input("Risk-free rate (r)", value=0.02, key="r_option")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="sigma_option")
            q = st.number_input("Dividend yield (q)", value=0.0, key="q_option")
            model_name = st.selectbox("Pricing model", ["Black-Scholes"], key="model_option")

            if st.button("Add option"):
                position = {
                    "instrument_type": "Option",
                    "ticker": ticker,
                    "S": S,
                    "K": K,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                    "q": q,
                    "qty": qty,
                    "buy_sell": buy_sell_flag,
                    "option_type": option_type,
                    "model_name": model_name,
                    "option_class": "Vanilla"
                }
                st.session_state["portfolio"] = add_position(st.session_state["portfolio"], position)
                st.success(f"{option_type} option on {ticker} added")

    # =========================
    # Display Portfolio
    # =========================
    st.subheader("1 - Current Portfolio (T1)")
    if not st.session_state["portfolio"]:
        st.info("Portfolio is empty.")
        return

    df = calculate_prices_and_greeks(st.session_state["portfolio"])
    st.dataframe(df, use_container_width=True)

    # =========================
    # Portfolio value & P&L
    # =========================
    market_value, total_cost = calculate_portfolio_value(st.session_state["portfolio"])
    pnl = market_value - total_cost

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost", f"{total_cost:.2f}")
    col2.metric("Market value", f"{market_value:.2f}")
    col3.metric("P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}")

    # =========================
    # Portfolio Greeks
    # =========================
    st.subheader("Portfolio Greeks")
    greeks = calculate_portfolio_greeks(st.session_state["portfolio"])
    st.table(greeks)

    # =========================
    # Delta Hedge
    # =========================
    st.subheader("Delta Hedging")
    hedge_ticker = st.selectbox(
        "Underlying used for hedging",
        list({p["ticker"] for p in st.session_state["portfolio"]}),
        key="hedge_ticker"
    )

    if st.button("⚖️ Delta Hedge Portfolio"):
        st.session_state["portfolio"] = delta_hedge_portfolio(st.session_state["portfolio"], hedge_ticker)
        st.success("Portfolio delta-hedged")
        st.rerun()

    # =========================
    # Remove positions
    # =========================
    st.subheader("Remove a position")
    to_delete = None
    for i, pos in enumerate(st.session_state["portfolio"]):
        name = pos["ticker"] if pos["instrument_type"] == "Stock" else f"{pos.get('option_type','Option')}_{i}"
        if st.button(f"Delete {name}", key=f"del_{i}"):
            to_delete = i

    if to_delete is not None:
        removed = st.session_state["portfolio"][to_delete]
        st.session_state["portfolio"] = [p for j, p in enumerate(st.session_state["portfolio"]) if j != to_delete]
        st.warning(f"Do you want to remove {removed.get('option_type', removed['instrument_type'])} {removed['ticker']} ? Click again to remove.")
        st.stop()

    # =========================
    # Scenario Analysis (T2)
    # =========================
    st.write("-------")
    st.subheader("2 - Scenario Analysis : Parameter Change (T2)")

    portfolio_t2 = []
    for i, pos in enumerate(st.session_state["portfolio"]):
        new_pos = pos.copy()
        new_pos["price_paid"] = pos.get("price_paid", 0)

        if pos["instrument_type"] == "Stock":
            new_pos["S"] = st.number_input(f"{pos['ticker']} | Stock | Spot",
                                           value=float(pos["S"]), key=f"S_t2_{i}")
            new_pos["price"] = new_pos["S"] * new_pos["qty"]
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                new_pos["S"] = st.number_input(f"{pos['ticker']} | {pos.get('option_type','Call')} | Spot",
                                               value=float(pos["S"]), key=f"S_t2_{i}")
            with col2:
                new_pos["sigma"] = st.number_input(f"{pos['ticker']} | {pos.get('option_type','Call')} | Volatility",
                                                  value=float(pos.get("sigma", 0.2)), key=f"sigma_t2_{i}")
            with col3:
                new_pos["T"] = st.number_input(f"{pos['ticker']} | {pos.get('option_type','Call')} | Maturity",
                                               value=float(pos["T"]), min_value=0.0, key=f"T_t2_{i}")

            # Recalcul du prix de l'option avec les nouveaux paramètres
            params = {
                "S": new_pos["S"],
                "K": new_pos["K"],
                "T": new_pos["T"],
                "r": new_pos["r"],
                "sigma": new_pos.get("sigma", 0.2),
                "q": new_pos.get("q", 0.0),
                "option_type": new_pos.get("option_type", "Call"),
                "buy_sell": new_pos.get("buy_sell", "buy"),
                "option_class": new_pos.get("option_class", "Vanilla")
            }
            model_name = new_pos.get("model_name", "Black-Scholes")
            try:
                new_pos["price"] = price_option(model_name, params) * new_pos.get("qty", 1)
            except Exception:
                new_pos["price"] = 0

        portfolio_t2.append(new_pos)

    df_t2 = calculate_prices_and_greeks(portfolio_t2)

    market_value_t2, _ = calculate_portfolio_value(portfolio_t2)
    pnl_t2 = market_value_t2 - total_cost

    st.subheader("Portfolio at new parameters (T2)")
    st.dataframe(df_t2, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost (T1)", f"{total_cost:.2f}")
    col2.metric("Market value (T2)", f"{market_value_t2:.2f}")
    col3.metric("P&L (T2)", f"{pnl_t2:.2f}", delta=f"{pnl_t2:.2f}")

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
