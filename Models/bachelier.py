import numpy as np
from scipy.stats import norm
from Models.models import Model

class Bachelier(Model):
    def __init__(self, S, K, r, T, sigma, option_type="call"):
        super().__init__(S, K, r, T, option_type, position="long", option_class="vanilla")
        self.sigma = max(float(sigma), 1e-8)

    def price(self):
        d = (self.S - self.K) / (self.sigma * np.sqrt(self.T))

        call_price = np.exp(-self.r * self.T) * (
            (self.S - self.K) * norm.cdf(d)
            + self.sigma * np.sqrt(self.T) * norm.pdf(d)
        )

        if self.option_type == "put":
            price = (call_price - self.S + self.K * np.exp(-self.r * self.T))*self.S
        else:
            price = (call_price)*self.S

        return max(price, 0.0)
