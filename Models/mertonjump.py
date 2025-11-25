import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model

class MertonJumpDiffusion(Model):
    """
    Modèle de Merton :
    dS/S = (r - λ*k)dt + σdW + JdN
    J ~ lognormal(mu_j, sigma_j)
    """
    def __init__(self, S, K, r, T, sigma, lambd, mu_j, sigma_j, option_type="call", position="buy"):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.lambd = lambd
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.option_type = option_type.lower()
        self.position = position.lower()  # "buy" ou "sell"

    def char_func(self, u):
        drift = self.r - self.lambd * (np.exp(self.mu_j + 0.5*self.sigma_j**2) - 1)
        jump = np.exp(1j*u*self.mu_j - 0.5*self.sigma_j**2*u**2)
        return np.exp(
            1j*u*(np.log(self.S) + drift*self.T)
            - 0.5*self.sigma**2*u**2*self.T
            + self.lambd*self.T*(jump - 1)
        )

    def price(self, n=64):
        x, w = laggauss(n)
        integrand = np.exp(-x) * np.real(
            np.exp(-1j * x * np.log(self.K)) * self.char_func(x - 1j) / (1j * x)
        )
        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        #  Parité Call-Put
        if self.option_type == "put":
            price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        else:
            price = call_price

        #  Gestion position
        return -price if self.position == "sell" else price

    