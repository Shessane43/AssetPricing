import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# List of tickers available for selection
# ============================================================
ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "BABA", "ORCL"]


class MarketDataFetcher:
    """
    Fetch market data using Yahoo Finance.
    """

    @staticmethod
    def get_last_price(ticker: str):
        """
        Fetch the last closing price for a given ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

        Returns
        -------
        float or None
            Last closing price, or None if data cannot be fetched.
        """
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty:
            return None
        return data['Close'].iloc[-1]


class OptionParameters:
    """
    Encapsulates all input parameters for an option.
    """

    def __init__(self, ticker, S, K, T, r, sigma, q, option_class, option_type, buy_sell):
        self.ticker = ticker
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_class = option_class
        self.option_type = option_type
        self.buy_sell = buy_sell


class PayoffCalculator:
    """
    Provides static methods to calculate option payoffs for vanilla and
    some exotic options (visual approximation only for exotics).
    """

    @staticmethod
    def vanilla_payoff(S_range, K, option_type):
        """
        Compute vanilla option payoff (call/put).

        Parameters
        ----------
        S_range : np.ndarray
            Array of underlying prices.
        K : float
            Strike price.
        option_type : str
            "Call" or "Put".

        Returns
        -------
        np.ndarray
            Payoff for each price in S_range.
        """
        if option_type == "Call":
            return np.maximum(S_range - K, 0)
        elif option_type == "Put":
            return np.maximum(K - S_range, 0)

    @staticmethod
    def asian_payoff(S_range, S0, K, option_type):
        """
        Compute Asian option payoff (visual approximation using simple average).

        Parameters
        ----------
        S_range : np.ndarray
            Array of underlying prices.
        S0 : float
            Spot price at t=0.
        K : float
            Strike price.
        option_type : str
            "Call" or "Put".

        Returns
        -------
        np.ndarray
            Payoff for each price in S_range.
        """
        avg = (S_range + S0) / 2
        if option_type == "Call":
            return np.maximum(avg - K, 0)
        else:
            return np.maximum(K - avg, 0)

    @staticmethod
    def lookback_payoff(S_range, K, option_type):
        """
        Compute Lookback option payoff (visual approximation using max/min).

        Parameters
        ----------
        S_range : np.ndarray
            Array of underlying prices.
        K : float
            Strike price.
        option_type : str
            "Call" or "Put".

        Returns
        -------
        np.ndarray
            Payoff for each price in S_range (all set to max payoff).
        """
        if option_type == "Call":
            base = np.maximum(S_range - K, 0)
        else:
            base = np.maximum(K - S_range, 0)
        return np.full_like(S_range, np.max(base))

    @staticmethod
    def choose_payoff(params: OptionParameters):
        """
        Compute the payoff for the given OptionParameters object.
        Supports vanilla, Asian, and Lookback options (approximation for exotics).

        Parameters
        ----------
        params : OptionParameters
            Object containing option parameters.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            S_range and corresponding payoff array.
        """
        S0 = params.S
        K = params.K
        option_type = params.option_type.lower()
        position = params.buy_sell.lower()

        S_range = np.linspace(0.5 * S0, 1.5 * S0, 120)

        # ---------- Vanilla ----------
        if option_type == "call":
            payoff = np.maximum(S_range - K, 0.0)
        elif option_type == "put":
            payoff = np.maximum(K - S_range, 0.0)
        # ---------- Exotic (visual approximation only) ----------
        elif option_type in ["asian_call", "call_asian"]:
            payoff = np.maximum(S_range - K, 0.0)
        elif option_type in ["asian_put", "put_asian"]:
            payoff = np.maximum(K - S_range, 0.0)
        elif option_type in ["lookback_call", "call_lookback"]:
            payoff = np.maximum(S_range - K, 0.0)
        elif option_type in ["lookback_put", "put_lookback"]:
            payoff = np.maximum(K - S_range, 0.0)
        else:
            raise ValueError(f"Unsupported option_type in payoff: {option_type}")

        # ---------- Adjust for position ----------
        if position in ["short", "sell"]:
            payoff = -payoff

        return S_range, payoff


class PayoffPlotter:
    """
    Static methods to plot option payoffs with dark theme.
    """

    @staticmethod
    def plot(S_range, payoff, K, label):
        """
        Plot payoff as a function of underlying price.

        Parameters
        ----------
        S_range : np.ndarray
            Array of underlying prices.
        payoff : np.ndarray
            Payoff corresponding to S_range.
        K : float
            Strike price.
        label : str
            Label for the payoff line.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Dark theme
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        # Payoff line
        ax.plot(S_range, payoff, color="orange", linewidth=2, label=label)
        ax.axvline(K, color="orange", linestyle="--", linewidth=2, label="Strike")

        # Spine colors
        for side in ["bottom", "top", "left", "right"]:
            ax.spines[side].set_color("orange")

        # Tick colors
        ax.tick_params(axis="x", colors="orange")
        ax.tick_params(axis="y", colors="orange")

        # Axis labels
        ax.set_xlabel("Underlying price at maturity", color="white")
        ax.set_ylabel("Payoff", color="white")

        # Grid
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        # Legend
        legend = ax.legend(facecolor="#0e1117", edgecolor="white")
        for text in legend.get_texts():
            text.set_color("orange")

        return fig
