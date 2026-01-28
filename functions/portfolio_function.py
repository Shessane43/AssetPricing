# functions/portfolio_function.py
import numpy as np
import pandas as pd
from functions.pricing_function import price_option, MODELS
from functions.greeks_function import Greeks

# ----------------- Portfolio management -----------------
def add_position(portfolio, position):
    """
    Adds a position to the portfolio and calculates the price paid.

    Parameters
    ----------
    portfolio : list of dict
        Current portfolio positions.
    position : dict
        Dictionary describing the new position, e.g.:
        {
            "position_type": "Option" or "Stock",
            "S": spot price,
            "K": strike (for options),
            "T": maturity (for options),
            "r": risk-free rate,
            "sigma": volatility (optional),
            "option_type": "Call" or "Put",
            "option_class": "Vanilla", etc.
        }

    Returns
    -------
    list of dict
        Updated portfolio including the new position with "price_paid" added.
    """
    pos = position.copy()
    qty = pos.get("qty", 1)

    if pos.get("position_type", "Option") == "Stock":
        pos["price_paid"] = pos["S"] * qty
    else:
        # Calculate option price paid
        params = {
            "S": pos["S"],
            "K": pos["K"],
            "T": pos["T"],
            "r": pos["r"],
            "sigma": pos.get("sigma", 0.2),
            "q": pos.get("q", 0.0),
            "option_type": pos.get("option_type", "Call"),
            "buy_sell": "Long",
            "option_class": pos.get("option_class", "Vanilla")
        }
        model_name = pos.get("model_name", "Black-Scholes")
        try:
            pos["price_paid"] = price_option(model_name, params) * qty
        except Exception:
            pos["price_paid"] = 0

    portfolio.append(pos)
    return portfolio

def remove_position(portfolio, index):
    """
    Removes a position from the portfolio by index.

    Parameters
    ----------
    portfolio : list of dict
        Current portfolio.
    index : int
        Index of the position to remove.

    Returns
    -------
    list of dict
        Updated portfolio after removal.
    """
    if 0 <= index < len(portfolio):
        portfolio.pop(index)
    return portfolio

# ----------------- Pricing per position -----------------
def calculate_prices_and_greeks(portfolio):
    """
    Calculates current price for each position in the portfolio.

    Parameters
    ----------
    portfolio : list of dict
        Portfolio positions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with pricing information per position, including
        columns: Position, Type, Spot, Strike, Maturity, Volatility, Price, Quantity, Dividend, etc.
    """
    results = []

    for pos in portfolio:
        res = pos.copy()
        qty = pos.get("qty", 1)

        # LONG / SHORT
        res["position_type"] = "Long" if qty > 0 else "Short"

        # OPTION / STOCK
        if pos.get("position_type") == "Stock":
            res["option_type"] = "Stock"
            res["price"] = pos["S"] * qty
        else:
            res["option_type"] = pos.get("option_type", "Call")

            params = {
                "S": pos["S"],
                "K": pos["K"],
                "T": pos["T"],
                "r": pos["r"],
                "sigma": pos.get("sigma", 0.2),
                "q": pos.get("q", 0.0),
                "option_type": pos.get("option_type", "Call"),
                "buy_sell": "Long",
                "option_class": pos.get("option_class", "Vanilla")
            }

            model_name = pos.get("model_name", "Black-Scholes")
            try:
                res["price"] = price_option(model_name, params) * qty
            except Exception:
                res["price"] = 0.0

        results.append(res)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns for display
    cols = ["position_type", "option_type"] + [c for c in df.columns if c not in ["position_type", "option_type"]]
    df = df[cols]

    # Rename columns for cleaner display
    rename_dict = {
        "position_type": "Position",
        "option_type": "Type",
        "ticker": "Ticker",
        "S": "Spot",
        "K": "Strike",
        "T": "Maturity",
        "sigma": "Volatility",
        "price": "Price",
        "qty": "Quantity",
        "q": "Dividend"
    }
    df.rename(columns=rename_dict, inplace=True)

    return df

# ----------------- Portfolio Greeks -----------------
def calculate_portfolio_greeks(portfolio):
    """
    Calculates aggregate Greeks for the portfolio.

    Parameters
    ----------
    portfolio : list of dict
        Portfolio positions.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with total delta, gamma, vega, theta, and rho.
    """
    total_delta = 0
    total_gamma = 0
    total_vega  = 0
    total_theta = 0
    total_rho   = 0

    for pos in portfolio:
        qty = pos.get("qty", 1)

        if pos.get("position_type", "Option") == "Stock":
            total_delta += qty
        else:
            try:
                g = Greeks(
                    option_type=pos["option_type"],
                    model=pos.get("model_name", "Black-Scholes"),
                    S=pos["S"],
                    K=pos["K"],
                    T=pos["T"],
                    r=pos["r"],
                    sigma=pos.get("sigma", 0.2),
                    buy_sell="Long" if qty > 0 else "Short"
                )
                total_delta += g.delta() * qty
                total_gamma += g.gamma() * qty
                total_vega  += g.vega() * qty
                total_theta += g.theta() * qty
                total_rho   += g.rho() * qty
            except Exception:
                pass

    return pd.DataFrame([{
        "delta": total_delta,
        "gamma": total_gamma,
        "vega": total_vega,
        "theta": total_theta,
        "rho": total_rho
    }])

# ----------------- Total portfolio value -----------------
def calculate_portfolio_value(portfolio):
    """
    Computes total market value and total cost of the portfolio.

    Parameters
    ----------
    portfolio : list of dict
        Portfolio positions.

    Returns
    -------
    tuple
        total_market_value : float
            Current market value of the portfolio.
        total_cost : float
            Total amount paid to acquire the positions.
    """
    total_market_value = 0.0
    total_cost = 0.0

    for pos in portfolio:
        qty = pos.get("qty", 1)

        if pos.get("position_type", "Option") == "Stock":
            price = pos.get("S", 0)
            cost = pos.get("price_paid", price)
        else:
            price = pos.get("price", 0)
            cost = pos.get("price_paid", 0)

        total_market_value += price * qty
        total_cost += cost

    return total_market_value, total_cost

def delta_hedge_portfolio(portfolio, ticker):
    """
    Adds a stock position to delta-hedge the portfolio (~delta = 0).

    Parameters
    ----------
    portfolio : list of dict
        Portfolio positions.
    ticker : str
        Ticker of the stock to use for hedging.

    Returns
    -------
    list of dict
        Updated portfolio including the delta-hedge stock position.
    """
    greeks = calculate_portfolio_greeks(portfolio)
    total_delta = greeks.loc[0, "delta"]

    if abs(total_delta) < 1e-4:
        return portfolio

    hedge_position = {
        "position_type": "Stock",
        "ticker": ticker,
        "S": next(p["S"] for p in portfolio if p["ticker"] == ticker),
        "qty": -total_delta
    }

    portfolio.append(hedge_position)
    return portfolio
