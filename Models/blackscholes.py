import numpy as np
from scipy.stats import norm
from Models.models import Model


class BlackScholes(Model):
    def __init__(self, S, K, r, sigma, T, option_type="Call", buy_sell="Buy", option_class="Vanille",q):
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
        if self.option_class != "vanille":
            raise ValueError("Black-Scholes ne s'applique qu’aux options vanilles européennes.")

        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return -price if self.buy_sell == "sell" else price

    def implied_volatility(self, market_price, tol=1e-6, max_iter=100):
        if self.option_class != "vanille":
            raise ValueError("La volatilité implicite n'est définie que pour les options vanilles.")

        sigma = 0.2  # initial guess

        for _ in range(max_iter):
            # recalcul de d1 et d2 avec dividendes
            d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)

            if self.option_type == "call":
                price_est = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            else:
                price_est = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

            vega = self.S * np.exp(-self.q * self.T) * np.sqrt(self.T) * norm.pdf(d1)

            if vega < 1e-8:
                return None

            sigma -= (price_est - market_price) / vega

            if abs(price_est - market_price) < tol:
                return sigma

        return None
    