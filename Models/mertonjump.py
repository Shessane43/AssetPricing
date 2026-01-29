import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model

class MertonJumpDiffusion(Model):
    def __init__(
        self, S, K, r, T,
        sigma, lambd, mu_j, sigma_j,
        option_type="call"
    ):
        super().__init__(S, K, r, T, option_type, position="long", option_class="vanilla")

        self.sigma = max(float(sigma), 1e-8)
        self.lambd = max(float(lambd), 0.0)
        self.mu_j = float(mu_j)
        self.sigma_j = max(float(sigma_j), 1e-8)

    def char_func(self, u):
        drift = self.r - self.lambd * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)
        jump = np.exp(1j*u*self.mu_j - 0.5*self.sigma_j**2*u**2)

        return np.exp(
            1j*u*(np.log(self.S) + drift*self.T)
            - 0.5*self.sigma**2*u**2*self.T
            + self.lambd*self.T*(jump - 1)
        )

    def price(self, n=64):
        if self.option_type not in ["call", "put"]:
            return np.nan

        x, w = laggauss(n)
        lnK = np.log(self.K)

        phi = self.char_func(x - 1j)
        integrand = np.real(
            np.exp(-1j * x * lnK) * phi / (1j * x)
        )

        integrand = np.nan_to_num(integrand, nan=0.0)
        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "put":
            price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        else:
            price = call_price

        return max(price, 0.0)
