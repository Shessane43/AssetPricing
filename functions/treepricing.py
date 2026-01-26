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
    option = Option(
        K=K,
        T=T,
        option_type=option_type,
        exercise=exercise
    )

    portfolio = OptionPortfolio([option], [1.0])

    tree = TreeModel(
        market=market,
        option=portfolio,
        pricing_date=0.0,
        n_steps=n_steps
    )

    return tree.price()
