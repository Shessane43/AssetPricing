# functions/portfolio_function.py
import numpy as np
import pandas as pd
from functions.pricing_function import price_option, MODELS
from functions.greeks_function import Greeks

# ----------------- Gestion du portfolio -----------------
def add_position(portfolio, position):
    """Ajoute une position et calcule le prix payé"""
    pos = position.copy()
    qty = pos.get("qty", 1)

    if pos.get("position_type", "Option") == "Stock":
        pos["price_paid"] = pos["S"] * qty
    else:
        # Calcul du prix payé de l'option
        params = {
            "S": pos["S"],
            "K": pos["K"],
            "T": pos["T"],
            "r": pos["r"],
            "sigma": pos.get("sigma",0.2),
            "q": pos.get("q",0.0),
            "option_type": pos.get("option_type","Call"),
            "buy_sell": "Buy",  # prix payé = achat
            "option_class": pos.get("option_class","Vanille")
        }
        model_name = pos.get("model_name", "Black-Scholes")
        try:
            pos["price_paid"] = price_option(model_name, params) * qty
        except Exception:
            pos["price_paid"] = 0

    portfolio.append(pos)
    return portfolio

def remove_position(portfolio, index):
    if 0 <= index < len(portfolio):
        portfolio.pop(index)
    return portfolio

# ----------------- Prix par position -----------------
def calculate_prices_and_greeks(portfolio):
    results = []
    for pos in portfolio:
        res = pos.copy()
        qty = pos.get("qty",1)

        if pos.get("position_type","Option") == "Stock":
            res["price"] = pos["S"] * qty
        else:
            params = {
                "S": pos["S"],
                "K": pos["K"],
                "T": pos["T"],
                "r": pos["r"],
                "sigma": pos.get("sigma",0.2),
                "q": pos.get("q",0.0),
                "option_type": pos.get("option_type","Call"),
                "buy_sell": "Buy",
                "option_class": pos.get("option_class","Vanille")
            }
            model_name = pos.get("model_name","Black-Scholes")
            try:
                res["price"] = price_option(model_name, params) * qty
            except Exception:
                res["price"] = 0

        results.append(res)
    return pd.DataFrame(results)

# ----------------- Greeks globaux -----------------
def calculate_portfolio_greeks(portfolio):
    total_delta = 0
    total_gamma = 0
    total_vega  = 0
    total_theta = 0
    total_rho   = 0

    for pos in portfolio:
        qty = pos.get("qty",1)

        if pos.get("position_type","Option") == "Stock":
            total_delta += qty
        else:
            try:
                g = Greeks(
                    option_type=pos["option_type"],
                    model=pos.get("model_name","Black-Scholes"),
                    S=pos["S"],
                    K=pos["K"],
                    T=pos["T"],
                    r=pos["r"],
                    sigma=pos.get("sigma",0.2),
                    buy_sell="Long" if qty>0 else "Short"
                )
                total_delta += g.delta() * qty
                total_gamma += g.gamma() * qty
                total_vega  += g.vega()  * qty
                total_theta += g.theta() * qty
                total_rho   += g.rho()   * qty
            except Exception:
                pass

    return pd.DataFrame([{
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
        "theta": total_theta,
        "rho": total_rho
    }])

# ----------------- Coût total -----------------
def calculate_portfolio_value(portfolio):
    """
    Retourne :
        - market_value : valeur actuelle du portfolio
        - cost : coût total payé à l'achat
    """
    total_market_value = 0.0
    total_cost = 0.0

    for pos in portfolio:
        qty = pos.get("qty", 1)

        # Stock
        if pos.get("position_type", "Option") == "Stock":
            price = pos.get("S", 0)  # prix actuel du stock
            cost = pos.get("price_paid", price)
        else:
            price = pos.get("price", 0)  # prix actuel de l'option
            cost = pos.get("price_paid", 0)

        total_market_value += price * qty
        total_cost += cost

    return total_market_value, total_cost

