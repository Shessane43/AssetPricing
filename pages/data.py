import streamlit as st
import yfinance as yf
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

    # Graphique des prix de clôture avec design dark/orange
    st.subheader("Graphique des prix de clôture")
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('black')  # fond figure
    ax.set_facecolor('black')          # fond axes

    # Courbe
    ax.plot(data.index, data['Close'], label='Close', color='orange', linewidth=2)

    # Axes et ticks
    ax.spines['bottom'].set_color('orange')
    ax.spines['top'].set_color('orange')
    ax.spines['left'].set_color('orange')
    ax.spines['right'].set_color('orange')
    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')

    # Labels
    ax.set_xlabel("Date", color='orange', fontsize=12)
    ax.set_ylabel("Prix ($)", color='orange', fontsize=12)

    # Grille
    ax.grid(True, color='orange', linestyle='--', alpha=0.3)

    # Légende
    legend = ax.legend(facecolor='black', edgecolor='orange')
    for text in legend.get_texts():
        text.set_color('orange')

    st.pyplot(fig)
