# functions/portfolio_function.py
import numpy as np
import pandas as pd
from functions.pricing_function import price_option
from functions.greeks_function import Greeks

# ----------------- Portfolio Management -----------------
def add_position(portfolio, position):
    """
    Add a new position to the portfolio.

    Parameters
    ----------
    portfolio : list
        List of existing positions (each position is a dict).
    position : dict
        Position to add. Must contain at least:
        - instrument_type : "Stock" or "Option"
        - qty : signed quantity (positive = long, negative = short)

    Notes
    -----
    - buy_sell is stored as a string:
        "buy"  -> Long position
        "sell" -> Short position
    - price_paid is always stored as a positive number (cash outflow).
    - price (market value) reflects the sign of the position via qty.
    """
    pos = position.copy()
    qty = pos.get("qty", 1)

    # Convert quantity sign into buy/sell flag
    buy_sell_flag = "buy" if qty > 0 else "sell"
    pos["buy_sell"] = buy_sell_flag

    if pos.get("instrument_type") == "Stock":
        # Cost is always positive, market value reflects long/short
        pos["price_paid"] = pos.get("S", 0) * abs(qty)
        pos["price"] = pos.get("S", 0) * qty
    else:
        # Option pricing parameters
        params = {
            "S": pos["S"],
            "K": pos["K"],
            "T": pos["T"],
            "r": pos["r"],
            "sigma": pos.get("sigma", 0.2),
            "q": pos.get("q", 0.0),
            "option_type": pos.get("option_type", "Call"),
            "buy_sell": buy_sell_flag,
            "option_class": pos.get("option_class", "Vanilla")
        }
        model_name = pos.get("model_name", "Black-Scholes")

        try:
            price = price_option(model_name, params)
        except Exception:
            price = 0

        # Market value uses signed quantity
        pos["price"] = price * qty
        # Paid cost is always positive
        pos["price_paid"] = price * abs(qty)

    portfolio.append(pos)
    return portfolio


def remove_position(portfolio, index):
    """
    Remove a position from the portfolio by index.

    Parameters
    ----------
    portfolio : list
        Portfolio of positions.
    index : int
        Index of the position to remove.
    """
    if 0 <= index < len(portfolio):
        portfolio.pop(index)
    return portfolio


# ----------------- Prices and Greeks -----------------
def calculate_prices_and_greeks(portfolio):
    """
    Compute individual prices (market values) and display information
    for each position in the portfolio.

    Parameters
    ----------
    portfolio : list
        Portfolio of positions.

    Returns
    -------
    pandas.DataFrame
        Table containing position info, prices, and parameters.
    """
    results = []

    for pos in portfolio:
        res = pos.copy()
        qty = pos.get("qty", 1)
        buy_sell_flag = pos.get("buy_sell", "buy")

        res["Position"] = buy_sell_flag

        if pos.get("instrument_type") == "Stock":
            res["Type"] = "Stock"
            res["Price"] = pos.get("S", 0) * qty
        else:
            res["Type"] = pos.get("option_type", "Call")

            params = {
                "S": pos["S"],
                "K": pos["K"],
                "T": pos["T"],
                "r": pos["r"],
                "sigma": pos.get("sigma", 0.2),
                "q": pos.get("q", 0.0),
                "option_type": pos.get("option_type", "Call"),
                "buy_sell": buy_sell_flag,
                "option_class": pos.get("option_class", "Vanilla")
            }
            model_name = pos.get("model_name", "Black-Scholes")

            try:
                res["Price"] = price_option(model_name, params) * qty
            except Exception:
                res["Price"] = 0

        results.append(res)

    df = pd.DataFrame(results)

    # Reorder and rename columns for display
    cols = ["Position", "Type"] + [c for c in df.columns if c not in ["Position", "Type"]]
    df = df[cols]

    df.rename(columns={
        "ticker": "Ticker",
        "S": "Spot",
        "K": "Strike",
        "T": "Maturity",
        "sigma": "Volatility",
        "price_paid": "Cost",
        "qty": "Quantity",
        "q": "Dividend"
    }, inplace=True)

    return df


# ----------------- Portfolio Greeks -----------------
def calculate_portfolio_greeks(portfolio):
    """
    Aggregate Greeks at the portfolio level.

    Parameters
    ----------
    portfolio : list
        Portfolio of positions.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing total:
        delta, gamma, vega, theta, rho
    """
    total_delta = 0
    total_gamma = 0
    total_vega  = 0
    total_theta = 0
    total_rho   = 0

    for pos in portfolio:
        qty = pos.get("qty")
        buy_sell_flag = pos.get("buy_sell", "buy")

        # Sign convention for long/short
        sign = 1 if buy_sell_flag == "buy" else -1

        if pos.get("instrument_type") == "Stock":
            # Stock delta = 1 per share
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
                    buy_sell=buy_sell_flag
                )

                total_delta += g.delta() * qty * sign
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


# ----------------- Portfolio Value -----------------
def calculate_portfolio_value(portfolio):
    """
    Compute total portfolio market value and total cost.

    Parameters
    ----------
    portfolio : list
        Portfolio of positions.

    Returns
    -------
    tuple
        (total_market_value, total_cost)
    """
    total_market_value = 0.0
    total_cost = 0.0

    for pos in portfolio:
        qty = pos.get("qty", 1)

        if pos.get("instrument_type") == "Stock":
            spot = pos.get("S", 0)
            total_market_value += spot * qty
            total_cost += pos.get("price_paid", abs(qty) * spot)
        else:
            total_market_value += pos.get("price", 0)
            total_cost += pos.get("price_paid", 0)

    return total_market_value, total_cost


# ----------------- Delta Hedge -----------------
def delta_hedge_portfolio(portfolio, ticker):
    """
    Delta-hedge the portfolio using the underlying stock.

    Parameters
    ----------
    portfolio : list
        Portfolio of positions.
    ticker : str
        Ticker of the underlying used for hedging.

    Notes
    -----
    - Existing hedge positions are removed.
    - A new stock position is added with quantity = -total_delta
      in order to neutralize portfolio delta.
    """
    # Remove existing hedges
    portfolio = [p for p in portfolio if p.get("Position") != "Hedge"]

    greeks = calculate_portfolio_greeks(portfolio)
    total_delta = greeks.loc[0, "delta"]

    # Already delta-neutral
    if abs(total_delta) < 1e-8:
        return portfolio

    # Find spot price of the underlying
    spot = next((p["S"] for p in portfolio if p.get("ticker") == ticker), 0)

    hedge_qty = -total_delta

    hedge_position = {
        "instrument_type": "Stock",
        "ticker": ticker,
        "S": spot,
        "qty": hedge_qty,
        "buy_sell": "buy" if hedge_qty > 0 else "sell",
        "Position": "Hedge",
        "price": spot * hedge_qty,
        "price_paid": spot * hedge_qty
    }

    portfolio.append(hedge_position)
    return portfolio
