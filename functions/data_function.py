import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

class MarketData:
    """
    Handles market data retrieval and visualization for a given ticker.

    Attributes:
        ticker (str): Ticker symbol of the asset (e.g., 'AAPL').
    """

    def __init__(self, ticker: str):
        """
        Initialize MarketData with a ticker.

        Args:
            ticker (str): Asset ticker symbol
        """
        self.ticker = ticker

    def fetch(self, period="1y"):
        """
        Download historical price data using Yahoo Finance.

        Args:
            period (str): Time period for historical data 
                          (e.g., '1mo', '6mo', '1y', '5y', 'max').

        Returns:
            pd.DataFrame: Historical OHLCV data or None if download fails.
        """
        try:
            data = yf.Ticker(self.ticker).history(period=period)
        except Exception as e:
            st.error(f"Error fetching data for {self.ticker}: {e}")
            data = None
        return data

    @staticmethod
    def plot_close_price(data):
        """
        Plot the closing price of the asset with a dark-orange theme.

        Args:
            data (pd.DataFrame): Historical price data with 'Close' column.

        Behavior:
            If data is empty or None, displays a warning in Streamlit.
        """
        if data is None or data.empty:
            st.warning("Cannot display plot: no data available.")
            return

        # --- Figure setup ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # --- Plot closing price ---
        ax.plot(data.index, data["Close"], color="orange", linewidth=2, label="Close Price")

        # --- Axes styling ---
        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("orange")
        ax.tick_params(colors="orange")
        ax.set_xlabel("Date", color="orange")
        ax.set_ylabel("Price ($)", color="orange")
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        # --- Legend styling ---
        legend = ax.legend(facecolor="black", edgecolor="orange")
        for text in legend.get_texts():
            text.set_color("orange")

        st.pyplot(fig)


def show_data_page():
    """
    Streamlit interface for the Market Data page.

    - Allows user to select a data period
    - Downloads historical data if necessary
    - Displays a preview table and a closing price chart
    """

    if "ticker" not in st.session_state:
        st.error("Missing parameters. Please return to the Parameters page.")
        return

    ticker = st.session_state.ticker
    st.subheader(f"Data for {ticker}")

    # --- User selects the period for historical data ---
    st.session_state.period = st.selectbox(
        "Data period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3  # default '1y'
    )

    # --- Fetch data if not already available or period changed ---
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

    # --- Display data preview ---
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # --- Display closing price chart ---
    st.subheader("Closing Price Chart")
    MarketData.plot_close_price(data)
