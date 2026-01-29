from scipy.stats import norm
import numpy as np


class Greeks_BS:
    """
    Black–Scholes Greeks calculator.

    This class computes option Greeks under the Black–Scholes model
    for European call and put options.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (continuously compounded).
    sigma : float
        Volatility of the underlying asset.
    option_type : str
        Option type. Expected values:
        - 'call' or 'put' (case-sensitive in some methods).
    buy_sell : bool
        Position flag:
        - True  -> long position
        - False -> short position

    Notes
    -----
    - The sign of each Greek depends on the buy/sell flag.
    - d1 and d2 are computed once at initialization and reused.
    """

    def __init__(self, S, K, T, r, sigma, option_type, buy_sell):
        """
        Initialize the Black–Scholes Greeks object and compute d1 and d2.
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.buy_sell = buy_sell

        # Black–Scholes auxiliary variables
        self.d1 = (
            np.log(self.S / self.K)
            + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def delta(self):
        """
        Compute the Delta of the option.

        Returns
        -------
        float
            Delta of the option, adjusted for long/short position.
        """
        if self.option_type == 'call':
            if self.buy_sell:
                return norm.cdf(self.d1)
            else:
                return -norm.cdf(self.d1)
        else:
            if self.buy_sell:
                return norm.cdf(self.d1) - 1
            else:
                return 1 - norm.cdf(self.d1)

    def gamma(self):
        """
        Compute the Gamma of the option.

        Returns
        -------
        float
            Gamma of the option, adjusted for long/short position.
        """
        if self.buy_sell:
            return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        else:
            return -norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """
        Compute the Vega of the option.

        Returns
        -------
        float
            Vega of the option (per 1 unit of volatility), adjusted
            for long/short position.
        """
        if self.buy_sell:
            return self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        else:
            return -self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self):
        """
        Compute the Theta of the option.

        Returns
        -------
        float
            Theta of the option, adjusted for long/short position.
        """
        if self.option_type == 'Call':
            if self.buy_sell:
                return (
                    - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
                )
            else:
                return (
                    (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
                )
        else:
            if self.buy_sell:
                return (
                    - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
                )
            else:
                return (
                    (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
                )

    def rho(self):
        """
        Compute the Rho of the option.

        Returns
        -------
        float
            Rho of the option (per 1% change in interest rate),
            adjusted for long/short position.
        """
        if self.option_type == 'Call':
            if self.buy_sell:
                return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
            else:
                return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:
            if self.buy_sell:
                return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
            else:
                return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
