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
    st.session_state.setdefault("portfolio", [])

    st.title("Gestion de Portfolio")

    # --- Ajouter position ---
    with st.expander("Ajouter une position"):
        position_type = st.selectbox("Type", ["Option", "Stock"], key="position_type_new")
        ticker = st.selectbox("Ticker", ALL_TICKERS, key="ticker_new")
        qty = st.number_input("Quantité (peut être négative pour short)", value=1.0, step=0.0001, format="%.4f", key="qty_new")

        if position_type == "Stock":
            S = st.number_input("Spot (S)", value=100.0, key="S_stock")
            if st.button("Ajouter Stock", key="add_stock"):
                position = {
                    "position_type": "Stock",
                    "ticker": ticker,
                    "S": S,
                    "qty": qty
                }
                st.session_state["portfolio"] = add_position(st.session_state["portfolio"], position)
                st.success(f"Stock {ticker} ajouté !")
        else:
            option_type = st.selectbox("Type d'option", ["Call", "Put"], key="option_type_new")
            S = st.number_input("Spot (S)", value=100.0, key="S_option")
            K = st.number_input("Strike (K)", value=100.0, key="K_option")
            T = st.number_input("Maturité (T années)", value=1.0, key="T_option")
            r = st.number_input("Taux sans risque (r)", value=0.02, key="r_option")
            sigma = st.number_input("Volatilité (σ)", value=0.2, key="sigma_option")
            q = st.number_input("Dividende (q)", value=0.0, key="q_option")
            model_name = st.selectbox("Modèle", ["Black-Scholes"], key="model_option")

            if st.button("Ajouter Option", key="add_option"):
                position = {
                    "position_type": "Option",
                    "ticker": ticker,
                    "S": S,
                    "K": K,
                    "T": T,
                    "r": r,
                    "q": q,
                    "qty": qty,
                    "option_type": option_type,
                    "model_name": model_name,
                    "option_class": "Vanilla"
                }
                st.session_state["portfolio"] = add_position(st.session_state["portfolio"], position)
                st.success(f"Option {option_type} sur {ticker} ajoutée !")

    # --- Portfolio actuel ---
    st.subheader("Portfolio actuel")
    if not st.session_state["portfolio"]:
        st.info("Le portfolio est vide.")
        return

    df = calculate_prices_and_greeks(st.session_state["portfolio"])
    st.dataframe(df)

    # Coût total
    _, cost = calculate_portfolio_value(st.session_state["portfolio"])
    st.markdown(f"**Coût total du portefeuille : {cost:.2f}**")

    # Greeks du portefeuille
    st.subheader("Greeks du portefeuille")
    greeks = calculate_portfolio_greeks(st.session_state["portfolio"])
    st.table(greeks)

    # --- Supprimer ---
    # --- Supprimer ---
    # --- Supprimer ---
    to_delete = None

    for i, pos in enumerate(st.session_state["portfolio"]):
        if pos["position_type"] == "Stock":
            name = f"Stock ({pos['ticker']})"
        else:
            name = f"{pos.get('option_type', pos['position_type'])} {i}"

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
            f"{removed['ticker']}  ? - Cliquez à nouveau pour confirmer"
        )
        st.stop()  # arrête le script ici et relance automatiquement avec l'état mis à jour
