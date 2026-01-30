from pricers.tree import TreeModel
from option.option import Option
from option.option import OptionPortfolio

def price_tree(
    market,
    S, K, r, q, sigma,
    T,
    option_type="call",
    exercise="european",
    n_steps=100
):
    """
    Price an option using a binomial/trinomial tree.

    Parameters:
    - market: Market environment or object providing rates, dividends, etc.
    - S: Spot price of the underlying
    - K: Strike price
    - r: Risk-free interest rate
    - q: Dividend yield (if applicable)
    - sigma: Volatility of the underlying
    - T: Time to maturity (in years)
    - option_type: 'call' or 'put'
    - exercise: 'european' or 'american'
    - n_steps: Number of steps in the tree

    Returns:
    - price: Option price calculated via the tree
    """

    # Create an Option object
    option = Option(
        K=K,
        T=T,
        option_type=option_type,
        exercise=exercise
    )

    # Wrap the option in a portfolio with weight 1
    portfolio = OptionPortfolio([option], [1.0])

    # Build the tree model and price the option
    tree = TreeModel(
        market=market,
        option=portfolio,
        pricing_date=0.0,
        n_steps=n_steps
    )

    return tree.price()
