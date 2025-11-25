import numpy as np
from scipy.stats import norm
from Models.models import Model


class BlackScholes(Model):
    def __init__(self, S, K, r, sigma, T, q, option_type="Call", buy_sell="Buy", option_class="Vanille"):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type.lower()
        self.buy_sell = buy_sell.lower()
        self.option_class = option_class.lower()  # "vanille" ou "exotique"
        self.q = q  # dividend yield
     
    def price(self):
        if self.option_class != "Vanille":
            raise ValueError("Black-Scholes ne s'applique qu’aux options vanilles européennes.")

        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "Call":
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return -price if self.buy_sell == "Sell" else price

    
    