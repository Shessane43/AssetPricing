# pages/portfolio.py
import streamlit as st
import pandas as pd

from functions.portfolio_function import (
    add_position,
    remove_position,
    calculate_prices_and_greeks,
    calculate_portfolio_greeks,
    calculate_portfolio_value
)
from functions.parameters_function import ALL_TICKERS


def app():
    # =========================
    # Initialisation
    # =========================
    st.session_state.setdefault("portfolio", [])

    st.title("Portfolio Management")

    # =========================
    # Add a position
    # =========================
    with st.expander("Add a position"):
        position_type = st.selectbox(
            "Instrument type",
            ["Option", "Stock"],
            key="position_type_new"
        )

        ticker = st.selectbox(
            "Ticker",
            ALL_TICKERS,
            key="ticker_new"
        )

        qty = st.number_input(
            "Quantity (negative for short)",
            value=1.0,
            step=0.0001,
            format="%.4f",
            key="qty_new"
        )

        # ---------- Stock ----------
        if position_type == "Stock":
            S = st.number_input("Spot price (S)", value=100.0, key="S_stock")

            if st.button("Add stock"):
                position = {
                    "position_type": "Stock",
                    "ticker": ticker,
                    "S": S,
                    "qty": qty
                }
                st.session_state["portfolio"] = add_position(
                    st.session_state["portfolio"],
                    position
                )
                st.success(f"Stock {ticker} added")

        # ---------- Option ----------
        else:
            option_type = st.selectbox(
                "Option type",
                ["Call", "Put"],
                key="option_type_new"
            )

            S = st.number_input("Spot (S)", value=100.0)
            K = st.number_input("Strike (K)", value=100.0)
            T = st.number_input("Maturity (T, years)", value=1.0)
            r = st.number_input("Risk-free rate (r)", value=0.02)
            sigma = st.number_input("Volatility (σ)", value=0.2)
            q = st.number_input("Dividend yield (q)", value=0.0)
            model_name = st.selectbox("Pricing model", ["Black-Scholes"])

            if st.button("Add option"):
                position = {
                    "position_type": "Option",
                    "ticker": ticker,
                    "S": S,
                    "K": K,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                    "q": q,
                    "qty": qty,
                    "option_type": option_type,
                    "model_name": model_name,
                    "option_class": "Vanilla"
                }
                st.session_state["portfolio"] = add_position(
                    st.session_state["portfolio"],
                    position
                )
                st.success(f"{option_type} option on {ticker} added")

    # =========================
    # Portfolio display
    # =========================
    st.subheader("Current portfolio")

    if not st.session_state["portfolio"]:
        st.info("Portfolio is empty.")
        return

    df = calculate_prices_and_greeks(st.session_state["portfolio"])
    st.dataframe(df, use_container_width=True)

    # =========================
    # Portfolio value & PnL
    # =========================
    market_value, cost = calculate_portfolio_value(
        st.session_state["portfolio"]
    )

    pnl = market_value - cost

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost", f"{cost:.2f}")
    col2.metric("Market value", f"{market_value:.2f}")
    col3.metric("P&L", f"{pnl:.2f}", delta=f"{pnl:.2f}")

    # =========================
    # Portfolio Greeks
    # =========================
    st.subheader("Portfolio Greeks")
    greeks = calculate_portfolio_greeks(st.session_state["portfolio"])
    st.table(greeks)

    # =========================
    # Remove positions
    # =========================
    st.subheader("Remove a position")

    # --- Supprimer ---
    to_delete = None

    for i, pos in enumerate(st.session_state["portfolio"]):
        if pos["position_type"] == "Stock":
            name = f"Stock ({pos['ticker']})"
        else:
            name = f"{pos.get('option_type', pos['position_type'])}_{i}"

        if st.button(f"Supprimer {name}", key=f"del_{i}"):
            to_delete = i

    if to_delete is not None:
        removed = st.session_state["portfolio"][to_delete]
        # Réassigner la liste pour que Streamlit détecte le changement
        st.session_state["portfolio"] = [
            p for j, p in enumerate(st.session_state["portfolio"]) if j != to_delete
        ]
        st.warning(
    f"Voulez-vous vraiment supprimer le {removed.get('option_type', removed['position_type'])} "
    f"{removed['ticker']} ? - Cliquez à nouveau pour confirmer")
        st.stop()  # arrête le script ici et relance automatiquement avec l'état mis à jour

