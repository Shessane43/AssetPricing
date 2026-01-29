import numpy as np
from scipy.stats import norm
from Models.models import Model

class Bachelier(Model):
    def __init__(self, S, K, r, T, sigma, option_type="call", position="buy"):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma  
        self.option_type = option_type.lower()  
        self.position = position.lower()      

    def price(self):
        d = (self.S - self.K) / (self.sigma * np.sqrt(self.T))

        # Prix du call via Bachelier
        call_price = np.exp(-self.r * self.T) * (
            (self.S - self.K) * norm.cdf(d) + self.sigma * np.sqrt(self.T) * norm.pdf(d)
        )

        # Si option put → parité call-put
        if self.option_type == "put":
            price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        else:
            price = call_price

        # Gestion de la position acheteur/vendeur
        if self.position == "sell":
            return -price
        return price

    