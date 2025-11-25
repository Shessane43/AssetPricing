import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

class MarketData:
    """Gestion de la récupération & visualisation des données de marché."""

    def __init__(self, ticker: str):
        self.ticker = ticker

    def fetch(self, period="1y"):
        """Télécharge l'historique des prix via Yahoo Finance."""
        try:
            data = yf.Ticker(self.ticker).history(period=period)
        except:
            data = None
        return data

    @staticmethod
    def plot_close_price(data):
        """Affichage stylisé du prix de clôture."""
        if data is None or data.empty:
            st.warning("Impossible d'afficher le graphique : aucune donnée.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.plot(data.index, data["Close"], color="orange", linewidth=2, label="Close Price")

        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("orange")

        ax.tick_params(colors="orange")
        ax.set_xlabel("Date", color="orange")
        ax.set_ylabel("Prix ($)", color="orange")
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        legend = ax.legend(facecolor="black", edgecolor="orange")
        for text in legend.get_texts():
            text.set_color("orange")

        st.pyplot(fig)


def show_data_page():
    """Interface affichée dans l’onglet Data."""
    st.header("Market Data")

    if "ticker" not in st.session_state:
        st.info("Veuillez d'abord sélectionner un ticker dans l'onglet Accueil.")
        return

    ticker = st.session_state.ticker
    st.subheader(f"Données pour {ticker}")

    #stocke la période sélectionnée
    st.session_state.period = st.selectbox(
        "Période des données",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3
    )

    # si pas déjà téléchargées ou si période a changé, on télécharge
    if ("market_data" not in st.session_state 
        or st.session_state.market_data is None 
        or st.session_state.market_data_period != st.session_state.period):

        market = MarketData(ticker)
        st.session_state.market_data = market.fetch(st.session_state.period)
        st.session_state.market_data_period = st.session_state.period

    data = st.session_state.market_data

    if data is None or data.empty:
        st.warning("Aucune donnée disponible pour ce ticker.")
        return

    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    st.subheader("Graphique du prix de clôture")
    MarketData.plot_close_price(data)
