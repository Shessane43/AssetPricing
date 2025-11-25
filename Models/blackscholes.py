import numpy as np
from scipy.stats import norm
from Models.models import Model

class BlackScholes(Model):
    def __init__(self, S, K, r, sigma, T, type="Call"):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.type = type

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.type == "Call":
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

   