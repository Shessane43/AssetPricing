import numpy as np
from scipy.integrate import quad
from Models.gammavariance import VarianceGamma

class Greeks_VarianceGamma:

    def __init__(self, S, K, r, T, sigma, theta, nu, option_type, buy_sell=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.option_type = option_type
        self.buy_sell = buy_sell

    def delta(self, h=1e-4):
        C_plus = VarianceGamma(self.S+h, self.K, self.r, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        C_minus = VarianceGamma(self.S-h, self.K, self.r, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else: 
            return - (C_plus - C_minus) / (2*h)

    def gamma(self, h=1e-4):
        C_plus = VarianceGamma(self.S+h, self.K, self.r, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        C_mid = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        C_minus =  VarianceGamma(self.S-h, self.K, self.r, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        if self.buy_sell:    
            return (C_plus - 2*C_mid + C_minus) / h**2
        else:
            return - (C_plus - 2*C_mid + C_minus) / h**2

    def vega(self, h=1e-4):
        C_minus = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma - h, self.theta, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma + h, self.theta, self.nu, option_type=self.option_type).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else:
            return - (C_plus - C_minus) / (2*h)

    def theta(self, h=1e-4):
        C_minus = VarianceGamma(self.S, self.K, self.r, self.T - h, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r, self.T + h, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        if self.buy_sell:
            return - (C_plus - C_minus) / (2*h)
        else:
            return (C_plus - C_minus) / (2*h)

    def rho(self, h=1e-4):
        C_minus = VarianceGamma(self.S, self.K, self.r - h, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r + h, self.T, self.sigma, self.theta, self.nu, option_type=self.option_type).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else:
            return - (C_plus - C_minus) / (2*h)