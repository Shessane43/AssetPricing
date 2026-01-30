import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "BABA", "ORCL"]


class MarketDataFetcher:
    @staticmethod
    def get_last_price(ticker: str):
        data = yf.Ticker(ticker).history(period="1d")
        if data.empty:
            return None
        return data['Close'].iloc[-1]


class OptionParameters:
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

    @staticmethod
    def vanilla_payoff(S_range, K, option_type):
        if option_type == "Call":
            return np.maximum(S_range - K, 0)
        elif option_type == "Put":
            return np.maximum(K - S_range, 0)

    @staticmethod
    def asian_payoff(S_range, S0, K, option_type):
        avg = (S_range + S0) / 2
        if option_type == "Call":
            return np.maximum(avg - K, 0)
        else:
            return np.maximum(K - avg, 0)

    @staticmethod
    def lookback_payoff(S_range, K, option_type):
        if option_type == "Call":
            base = np.maximum(S_range - K, 0)
        else:
            base = np.maximum(K - S_range, 0)
        return np.full_like(S_range, np.max(base))

    @staticmethod
    def choose_payoff(params: OptionParameters):
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
        # ---------- Position ----------
        if position in ["short", "sell"]:
            payoff = -payoff

        return S_range, payoff


class PayoffPlotter:

    @staticmethod
    def plot(S_range, payoff, K, label):
        fig, ax = plt.subplots(figsize=(8,5))

        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        ax.plot(S_range, payoff, color="orange", linewidth=2, label=label)
        ax.axvline(K, color="orange", linestyle="--", linewidth=2, label="Strike")

        for side in ["bottom", "top", "left", "right"]:
            ax.spines[side].set_color("orange")

        ax.tick_params(axis="x", colors="orange")
        ax.tick_params(axis="y", colors="orange")

        ax.set_xlabel("Prix du sous-jacent à maturité", color="white")
        ax.set_ylabel("Payoff", color="white")

        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        legend = ax.legend(facecolor="#0e1117", edgecolor="white")
        for text in legend.get_texts():
            text.set_color("orange")

        return fig
