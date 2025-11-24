import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.header("Market Data")

    # Vérifie si un ticker a été sélectionné dans l'accueil
    if 'ticker' not in st.session_state:
        st.info("Veuillez d'abord choisir un ticker dans l'onglet Accueil.")
        return

    ticker = st.session_state.ticker
    st.subheader(f"Données pour {ticker}")

    # Choix de la période
    period = st.selectbox("Période des données", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)

    # Récupération des données
    data = yf.Ticker(ticker).history(period=period)

    if data.empty:
        st.warning("Pas de données disponibles pour ce ticker.")
        return

    # Affiche les 5 premières lignes
    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    # Graphique des prix
    st.subheader("Graphique des prix de clôture")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix ($)")
    ax.legend()
    st.pyplot(fig)

    # Optionnel : calcule volatilité historique
    st.subheader("Volatilité historique (annuelle)")
    data['returns'] = data['Close'].pct_change()
    volatility = data['returns'].std() * (252**0.5)
    st.write(f"Volatilité annuelle approximative : {volatility:.2%}")
