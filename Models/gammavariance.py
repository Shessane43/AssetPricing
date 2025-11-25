import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model

class VarianceGamma(Model):
    def __init__(self, S, K, r, T, sigma, theta, nu, option_type="call", position="buy"):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma      # Volatilité brownienne
        self.theta = theta      # Drift du processus Gamma
        self.nu = nu            # Intensité des sauts
        self.option_type = option_type.lower()  # "call" ou "put"
        self.position = position.lower()        # "buy" ou "sell"

    def char_func(self, u):
        # Fonction caractéristique du modèle Variance Gamma
        return np.exp(1j * u * np.log(self.S * np.exp(self.r * self.T))) * \
               (1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u**2) ** (-self.T / self.nu)

    def price(self, n=64):
        # Intégration par quadrature de Gauss-Laguerre (comme Lewis)
        x, w = laggauss(n)
        integrand = np.exp(-x) * np.real(
            np.exp(-1j * x * np.log(self.K)) * self.char_func(x - 1j) / (1j * x)
        )
        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "put":
            price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        else:
            price = call_price

        #  Gestion acheteur/vendeur
        return -price if self.position == "sell" else price

    