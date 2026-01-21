from functions.pricing_function import price_option
import numpy as np
import matplotlib.pyplot as plt

def straddle_price(S, K, T, r, q, sigma):
    """
    Prix d'un Straddle : Long Call + Long Put, même strike
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
    Prix d'un Strangle : Long Put + Long Call, strikes différents
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
    Prix d'un Bull Spread : Long Call strike bas + Short Call strike haut
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
    Prix d'un Bear Spread : Long Put strike haut + Short Put strike bas
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
    Long Call strike bas, Short 2 Calls strike milieu, Long Call strike haut
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
    Long Put strike bas + Short Call strike haut
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
    Affiche le payoff d'un produit structuré.
    products: liste de dictionnaires {"option_type","K","weight"}
    S0: spot actuel
    S_range: array de spots à simuler (optionnel)
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
            raise ValueError("option_type doit être 'Call' ou 'Put'")

        payoff_total += p["weight"] * payoff

    # --- Plot thème noir / orange ---
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.plot(S_range, payoff_total, color="orange", lw=2, label="Payoff total")
    ax.axhline(0, color="white", lw=1, linestyle="--")
    ax.axvline(S0, color="cyan", lw=1, linestyle="--", label="Spot actuel")

    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color("orange")

    ax.tick_params(colors="orange")
    ax.set_xlabel("Spot à maturité", color="orange")
    ax.set_ylabel("Payoff", color="orange")
    ax.grid(True, linestyle="--", color="orange", alpha=0.3)

    legend = ax.legend(facecolor="black", edgecolor="orange")
    for text in legend.get_texts():
        text.set_color("orange")

    return fig
