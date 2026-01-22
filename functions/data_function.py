import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

class MarketData:
    """Handles market data retrieval & visualization."""

    def __init__(self, ticker: str):
        self.ticker = ticker

    def fetch(self, period="1y"):
        """Downloads historical price data via Yahoo Finance."""
        try:
            data = yf.Ticker(self.ticker).history(period=period)
        except:
            data = None
        return data

    @staticmethod
    def plot_close_price(data):
        """Stylized plot of the closing price."""
        if data is None or data.empty:
            st.warning("Cannot display plot: no data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.plot(data.index, data["Close"], color="orange", linewidth=2, label="Close Price")

        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("orange")

        ax.tick_params(colors="orange")
        ax.set_xlabel("Date", color="orange")
        ax.set_ylabel("Price ($)", color="orange")
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        legend = ax.legend(facecolor="black", edgecolor="orange")
        for text in legend.get_texts():
            text.set_color("orange")

        st.pyplot(fig)


def show_data_page():
    """Interface displayed in the Data tab."""

    if "ticker" not in st.session_state:
        st.error("Missing parameters. Please return to the Parameters page.")
        return

    ticker = st.session_state.ticker
    st.subheader(f"Data for {ticker}")

    # Store the selected period
    st.session_state.period = st.selectbox(
        "Data period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3
    )

    # If not already downloaded or period changed, fetch data
    if ("market_data" not in st.session_state 
        or st.session_state.market_data is None 
        or st.session_state.market_data_period != st.session_state.period):

        market = MarketData(ticker)
        st.session_state.market_data = market.fetch(st.session_state.period)
        st.session_state.market_data_period = st.session_state.period

    data = st.session_state.market_data

    if data is None or data.empty:
        st.warning("No data available for this ticker.")
        return

    st.subheader("Data Preview")
    st.dataframe(data.head())

    st.subheader("Closing Price Chart")
    MarketData.plot_close_price(data)
