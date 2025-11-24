import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model

class HestonLewis(Model):
    def __init__(self, S, K, r, T, v0, kappa, theta, sigma_v, rho, option_type="call"):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.option_type = option_type

    def char_func(self, u):
        a = self.kappa * self.theta
        b = self.kappa
        sigma = self.sigma_v
        rho = self.rho
        r = self.r
        T = self.T
        v0 = self.v0

        d = np.sqrt((rho * sigma * 1j * u - b)**2 + sigma**2 * (1j*u + u**2))
        g = (b - rho * sigma * 1j * u - d) / (b - rho * sigma * 1j * u + d)

        C = r * 1j * u * T + a/sigma**2 * ((b - rho*sigma*1j*u - d)*T - 2*np.log((1 - g*np.exp(-d*T))/(1 - g)))
        D = (b - rho*sigma*1j*u - d) * (1 - np.exp(-d*T)) / (sigma**2 * (1 - g*np.exp(-d*T)))

        return np.exp(C + D * v0 + 1j * u * np.log(self.S))

    def price(self, n=64):
        x, w = laggauss(n)     # Gauss-Laguerre 64 points
        integrand = np.exp(-x) * np.real(np.exp(-1j*x*np.log(self.K)) * self.char_func(x - 1j) / (1j*x))
        call_price = np.exp(-self.r*self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "call":
            return call_price
        else:
            return call_price - self.S + self.K * np.exp(-self.r*self.T)

