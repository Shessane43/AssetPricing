import numpy as np
from scipy.integrate import quad
from Models.gammavariance import VarianceGamma

class Greeks_VarianceGamma:
    """
    Greeks for Variance Gamma model using finite differences.

    Supports all option types (call, put, exotic) with buy/sell position.
    """

    def __init__(self, S, K, r, T, sigma, theta, nu, option_type, buy_sell=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.theta_vg = theta
        self.nu = nu
        self.option_type = option_type
        self.buy_sell = buy_sell  # True=long/buy, False=short/sell

    def delta(self, h=1e-4):
        """
        Delta = sensitivity to underlying price S
        """
        C_plus = VarianceGamma(self.S+h, self.K, self.r, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_minus = VarianceGamma(self.S-h, self.K, self.r, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        factor = 1 if self.buy_sell else -1
        return factor * (C_plus - C_minus) / (2*h)

    def gamma(self, h=1e-4):
        """
        Gamma = second derivative with respect to underlying S
        """
        C_plus = VarianceGamma(self.S+h, self.K, self.r, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_mid = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_minus = VarianceGamma(self.S-h, self.K, self.r, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        factor = 1 if self.buy_sell else -1
        return factor * (C_plus - 2*C_mid + C_minus) / h**2

    def vega(self, h=1e-4):
        """
        Vega = sensitivity to volatility parameter sigma
        """
        C_minus = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma - h, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r, self.T, self.sigma + h, self.theta_vg, self.nu, option_type=self.option_type).price()
        factor = 1 if self.buy_sell else -1
        return factor * (C_plus - C_minus) / (2*h)

    def theta(self, h=1e-4):
        """
        Theta = sensitivity to time to maturity T
        """
        C_minus = VarianceGamma(self.S, self.K, self.r, self.T - h, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r, self.T + h, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        factor = 1 if self.buy_sell else -1
        return -factor * (C_plus - C_minus) / (2*h)

    def rho(self, h=1e-4):
        """
        Rho = sensitivity to risk-free rate r
        """
        C_minus = VarianceGamma(self.S, self.K, self.r - h, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        C_plus = VarianceGamma(self.S, self.K, self.r + h, self.T, self.sigma, self.theta_vg, self.nu, option_type=self.option_type).price()
        factor = 1 if self.buy_sell else -1
        return factor * (C_plus - C_minus) / (2*h)
