import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model

class VarianceGamma(Model):
    def __init__(self, S, K, r, T, sigma, theta, nu, option_type="call"):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.option_type = option_type

    def char_func(self, u):
        return np.exp(1j * u * np.log(self.S * np.exp(self.r * self.T))) * \
               (1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u**2) ** (-self.T / self.nu)

    def price(self, n=64):
        x, w = laggauss(n)
        integrand = np.exp(-x) * np.real(np.exp(-1j * x * np.log(self.K)) *
                                         self.char_func(x - 1j) / (1j * x))
        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "call":
            return call_price
        else:  # Parité put-call
            return call_price - self.S + self.K * np.exp(-self.r * self.T)

