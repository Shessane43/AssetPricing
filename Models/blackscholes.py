import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Models.models import Model
import streamlit as st


class BlackScholes(Model):
    """
    Black-Scholes Model for European Option Pricing.

    This class implements the Black-Scholes framework for pricing European
    vanilla options and computing their Greeks. The model assumes that the
    underlying asset price follows a geometric Brownian motion with constant
    volatility and interest rate.
    """

    def __init__(self, S, K, r, sigma, T, q, option_type, buy_sell, option_class):
        """
        Initialize a Black-Scholes option pricing model.

        Parameters
        ----------
        S : float
            Current price of the underlying asset.
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (continuously compounded).
        sigma : float
            Volatility of the underlying asset.
        T : float
            Time to maturity (in years).
        q : float
            Continuous dividend yield.
        option_type : str
            Type of the option ("call" or "put").
        buy_sell : str
            Position type ("buy" or "sell").
        option_class : str
            Option class (only "vanilla" is supported).
        """
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.option_type = option_type.lower()
        self.buy_sell = buy_sell.lower()
        self.option_class = option_class.lower()

    def price(self):
        """
        Compute the Black-Scholes price of the European option.

        Returns
        -------
        float
            The option price, adjusted for a long or short position.

        Raises
        ------
        Exception
            If the option class is not vanilla.
        """
        if self.option_class != "vanilla":
            raise Exception("Black-Scholes applies only to European vanilla options.")

        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
                     - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                     - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))

        return -price if self.buy_sell == "sell" else price

    def delta(self):
        """
        Compute the Delta of the option.

        Returns
        -------
        float
            Delta of the option.
        """
        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.option_type == "call":
            delta = np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            delta = np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)

        return -delta if self.buy_sell == "short" else delta

    def lists_delta(self, points=0.3, number=100, h=1e-4):
        """
        Compute Delta values over a range of underlying prices.

        Parameters
        ----------
        points : float
            Relative variation around the spot price.
        number : int
            Number of evaluation points.
        h : float
            Unused parameter (kept for consistency).

        Returns
        -------
        tuple
            Underlying price values and corresponding Delta values.
        """
        delta_list = []
        S_values = np.linspace(self.S - points * self.S, self.S + points * self.S, number)

        for S in S_values:
            modele = BlackScholes(S, self.K, self.r, self.sigma,
                                  self.T, self.q,
                                  self.option_type, self.buy_sell, self.option_class)
            delta_list.append(modele.delta())

        return S_values, delta_list

    def gamma(self):
        """
        Compute the Gamma of the option.

        Returns
        -------
        float
            Gamma of the option.
        """
        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        gamma = (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S * self.sigma * np.sqrt(self.T))
        return -gamma if self.buy_sell == "short" else gamma

    def lists_gamma(self, points=0.3, number=100, h=1e-4):
        """
        Compute Gamma values over a range of underlying prices.

        Returns
        -------
        tuple
            Underlying price values and corresponding Gamma values.
        """
        gamma_list = []
        S_values = np.linspace(self.S - points * self.S, self.S + points * self.S, number)

        for S in S_values:
            modele = BlackScholes(S, self.K, self.r, self.sigma,
                                  self.T, self.q,
                                  self.option_type, self.buy_sell, self.option_class)
            gamma_list.append(modele.gamma())

        return S_values, gamma_list

    def vega(self):
        """
        Compute the Vega of the option.

        Returns
        -------
        float
            Vega of the option.
        """
        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
        return -vega if self.buy_sell == "short" else vega

    def lists_vega(self, points=0.3, number=100, h=1e-4):
        """
        Compute Vega values over a range of volatilities.

        Returns
        -------
        tuple
            Volatility values and corresponding Vega values.
        """
        vega_list = []
        sigma_values = np.linspace(self.sigma - points * self.sigma,
                                   self.sigma + points * self.sigma, number)

        for sigma in sigma_values:
            modele = BlackScholes(self.S, self.K, self.r, sigma,
                                  self.T, self.q,
                                  self.option_type, self.buy_sell, self.option_class)
            vega_list.append(modele.vega())

        return sigma_values, vega_list

    def theta(self):
        """
        Compute the Theta of the option.

        Returns
        -------
        float
            Theta of the option.
        """
        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            theta = (- (self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T))
                     / (2 * np.sqrt(self.T))
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                     + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1))
        else:
            theta = (- (self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T))
                     / (2 * np.sqrt(self.T))
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                     - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))

        return -theta if self.buy_sell == "short" else theta

    def lists_theta(self, points=0.3, number=100, h=1e-4):
        """
        Compute Theta values over a range of maturities.

        Returns
        -------
        tuple
            Maturity values and corresponding Theta values.
        """
        theta_list = []
        T_values = np.linspace(self.T - points * self.T,
                               self.T + points * self.T, number)

        for T in T_values:
            modele = BlackScholes(self.S, self.K, self.r, self.sigma,
                                  T, self.q,
                                  self.option_type, self.buy_sell, self.option_class)
            theta_list.append(modele.theta())

        return T_values, theta_list

    def rho(self):
        """
        Compute the Rho of the option.

        Returns
        -------
        float
            Rho of the option.
        """
        d1 = (np.log(self.S / self.K)
              + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return -rho if self.buy_sell == "short" else rho

    def lists_rho(self, points=0.3, number=100, h=1e-4):
        """
        Compute Rho values over a range of interest rates.

        Returns
        -------
        tuple
            Interest rate values and corresponding Rho values.
        """
        rho_list = []
        r_values = np.linspace(self.r - points * self.r,
                                self.r + points * self.r, number)

        for r in r_values:
            modele = BlackScholes(self.S, self.K, r, self.sigma,
                                  self.T, self.q,
                                  self.option_type, self.buy_sell, self.option_class)
            rho_list.append(modele.rho())

        return r_values, rho_list

    def plot_all_greeks(self, points=0.3, number=100, h=1e-4):
        """
        Plot all major Greeks on a single figure.
        """
        S_delta, delta_values = self.lists_delta(points, number, h)
        S_gamma, gamma_values = self.lists_gamma(points, number, h)
        sigma_vega, vega_values = self.lists_vega(points, number, h)
        T_theta, theta_values = self.lists_theta(points, number, h)
        r_rho, rho_values = self.lists_rho(points, number, h)

        plt.figure(figsize=(12, 8))
        plt.plot(S_delta, delta_values, label="Delta")
        plt.plot(S_gamma, gamma_values, label="Gamma")
        plt.plot(sigma_vega, vega_values, label="Vega")
        plt.plot(T_theta, theta_values, label="Theta")
        plt.plot(r_rho, rho_values, label="Rho")
        plt.title("Greeks")
        plt.ylabel("Greeks")
        plt.grid(True)
        plt.legend()
        plt.show()
