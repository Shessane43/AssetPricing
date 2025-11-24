import numpy as np
from scipy.integrate import quad
from model import HestonModel

class Greeks_Heston:

    def __init__(self, S, K, T, r, kappa, theta, sigma_v, rho, v0,
                 option_type='call', buyer=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r

        # Paramètres Heston
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0

        self.option_type = option_type
        self.buyer = buyer

    def delta(self, h=1e-4):
        return (Greeks_Heston(self.S+h, self.K, self.T, self.r,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price() -
                Greeks_Heston(self.S-h, self.K, self.T, self.r,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price()) / (2*h)

    def gamma(self, h=1e-4):
        C_plus = Greeks_Heston(self.S+h, self.K, self.T, self.r,
                               self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                               self.option_type, self.buyer).price()
        C_mid = self.price()
        C_minus = Greeks_Heston(self.S-h, self.K, self.T, self.r,
                                self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                self.option_type, self.buyer).price()
        return (C_plus - 2*C_mid + C_minus) / h**2

    def vega(self, h=1e-4):
        return (Greeks_Heston(self.S, self.K, self.T, self.r,
                              self.kappa, self.theta, self.sigma_v+h, self.rho, self.v0,
                              self.option_type, self.buyer).price() -
                Greeks_Heston(self.S, self.K, self.T, self.r,
                              self.kappa, self.theta, self.sigma_v-h, self.rho, self.v0,
                              self.option_type, self.buyer).price()) / (2*h)

    def theta(self, h=1e-4):
        return (Greeks_Heston(self.S, self.K, self.T+h, self.r,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price() -
                Greeks_Heston(self.S, self.K, self.T-h, self.r,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price()) / (2*h)

    def rho(self, h=1e-4):
        return (Greeks_Heston(self.S, self.K, self.T, self.r+h,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price() -
                Greeks_Heston(self.S, self.K, self.T, self.r-h,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buyer).price()) / (2*h)
