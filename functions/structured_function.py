from functions.pricing_function import price_option
import numpy as np
import matplotlib.pyplot as plt

def straddle_price(S, K, T, r, q, sigma):
    """
    Computes the price of a Straddle: Long Call + Long Put at the same strike.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price for both Call and Put.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility of the underlying.

    Returns
    -------
    float
        Total price of the Straddle (Call + Put).
    """
    call_params = {
        "S": S, "K": K, "T": T, "r": r, "q": q,
        "sigma": sigma, "option_type": "Call",
        "buy_sell": "Long", "option_class": "Vanilla"
    }
    put_params = call_params.copy()
    put_params["option_type"] = "Put"

    call_price = price_option("Black-Scholes", call_params)
    put_price = price_option("Black-Scholes", put_params)
    return call_price + put_price


def strangle_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call):
    """
    Computes the price of a Strangle: Long Put + Long Call at different strikes.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K_put : float
        Strike price of the Put.
    K_call : float
        Strike price of the Call.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma_put : float
        Volatility of the Put.
    sigma_call : float
        Volatility of the Call.

    Returns
    -------
    float
        Total price of the Strangle (Long Put + Long Call).
    """
    call_params = {
        "S": S, "K": K_call, "T": T, "r": r, "q": q,
        "sigma": sigma_call, "option_type": "Call",
        "buy_sell": "Long", "option_class": "Vanilla"
    }
    put_params = {
        "S": S, "K": K_put, "T": T, "r": r, "q": q,
        "sigma": sigma_put, "option_type": "Put",
        "buy_sell": "Long", "option_class": "Vanilla"
    }

    call_price = price_option("Black-Scholes", call_params)
    put_price = price_option("Black-Scholes", put_params)
    return call_price + put_price


def bull_spread_price(S, K_low, K_high, T, r, q, sigma):
    """
    Computes the price of a Bull Spread: Long Call (low strike) + Short Call (high strike).

    Returns
    -------
    float
        Net price of the Bull Spread.
    """
    long_call = price_option("Black-Scholes", {
        "S": S, "K": K_low, "T": T, "r": r, "q": q,
        "sigma": sigma, "option_type": "Call",
        "buy_sell": "Long", "option_class": "Vanilla"
    })
    short_call = price_option("Black-Scholes", {
        "S": S, "K": K_high, "T": T, "r": r, "q": q,
        "sigma": sigma, "option_type": "Call",
        "buy_sell": "Short", "option_class": "Vanilla"
    })
    return long_call + short_call


def bear_spread_price(S, K_high, K_low, T, r, q, sigma):
    """
    Computes the price of a Bear Spread: Long Put (high strike) + Short Put (low strike).

    Returns
    -------
    float
        Net price of the Bear Spread.
    """
    long_put = price_option("Black-Scholes", {
        "S": S, "K": K_high, "T": T, "r": r, "q": q,
        "sigma": sigma, "option_type": "Put",
        "buy_sell": "Long", "option_class": "Vanilla"
    })
    short_put = price_option("Black-Scholes", {
        "S": S, "K": K_low, "T": T, "r": r, "q": q,
        "sigma": sigma, "option_type": "Put",
        "buy_sell": "Short", "option_class": "Vanilla"
    })
    return long_put + short_put


def butterfly_spread_price(S, K_low, K_mid, K_high, T, r, q, sigma):
    """
    Computes the price of a Butterfly Spread:
    Long Call (low strike), Short 2 Calls (mid strike), Long Call (high strike).

    Returns
    -------
    float
        Net price of the Butterfly Spread.
    """
    long_low = price_option("Black-Scholes", {"S": S, "K": K_low, "T": T, "r": r, "q": q,
                                              "sigma": sigma, "option_type": "Call",
                                              "buy_sell": "Long", "option_class": "Vanilla"})
    short_mid = 2 * price_option("Black-Scholes", {"S": S, "K": K_mid, "T": T, "r": r, "q": q,
                                                   "sigma": sigma, "option_type": "Call",
                                                   "buy_sell": "Short", "option_class": "Vanilla"})
    long_high = price_option("Black-Scholes", {"S": S, "K": K_high, "T": T, "r": r, "q": q,
                                               "sigma": sigma, "option_type": "Call",
                                               "buy_sell": "Long", "option_class": "Vanilla"})
    return long_low + long_high + short_mid


def collar_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call):
    """
    Computes the price of a Collar: Long Put (low strike) + Short Call (high strike).

    Returns
    -------
    float
        Net price of the Collar.
    """
    long_put = price_option("Black-Scholes", {"S": S, "K": K_put, "T": T, "r": r, "q": q,
                                              "sigma": sigma_put, "option_type": "Put",
                                              "buy_sell": "Long", "option_class": "Vanilla"})
    short_call = price_option("Black-Scholes", {"S": S, "K": K_call, "T": T, "r": r, "q": q,
                                                "sigma": sigma_call, "option_type": "Call",
                                                "buy_sell": "Short", "option_class": "Vanilla"})
    return long_put + short_call


def plot_structured_payoff(products, S0, S_range=None):
    """
    Plots the payoff of a structured product composed of vanilla options.

    Parameters
    ----------
    products : list of dict
        Each dict must have keys: "option_type" ("Call"/"Put"), "K" (strike), "weight".
    S0 : float
        Current spot price.
    S_range : np.ndarray, optional
        Array of spot prices to evaluate payoff. If None, default to [0.5*S0, 1.5*S0].

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing total payoff over S_range.
    """
    if S_range is None:
        S_min = S0 * 0.5
        S_max = S0 * 1.5
        S_range = np.linspace(S_min, S_max, 200)

    payoff_total = np.zeros_like(S_range)

    for p in products:
        if p["option_type"] == "Call":
            payoff = np.maximum(S_range - p["K"], 0)
        elif p["option_type"] == "Put":
            payoff = np.maximum(p["K"] - S_range, 0)
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")

        payoff_total += p["weight"] * payoff

    # --- Plot with black/orange theme ---
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(S_range, payoff_total, color="orange", lw=2, label="Payoff total")
    ax.axhline(0, color="white", lw=1, linestyle="--")
    ax.axvline(S0, color="cyan", lw=1, linestyle="--", label="Current spot")

    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color("orange")

    ax.tick_params(colors="orange")
    ax.set_xlabel("Spot at maturity", color="orange")
    ax.set_ylabel("Payoff", color="orange")
    ax.grid(True, linestyle="--", color="orange", alpha=0.3)

    legend = ax.legend(facecolor="black", edgecolor="orange")
    for text in legend.get_texts():
        text.set_color("orange")

    return fig
