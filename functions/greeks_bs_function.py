from scipy.stats import norm
import numpy as np


class Greeks_BS:

    def __init__(self, S, K, T, r, sigma, option_type, buy_sell):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.buy_sell = buy_sell
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def delta(self):
        if self.option_type == 'Call':
            if self.buy_sell == "Long":
                return norm.cdf(self.d1)
            else:
                return - norm.cdf(self.d1)
        else:
            if self.buy_sell == 'Long':
                return norm.cdf(self.d1) - 1
            else:
                return 1 - norm.cdf(self.d1)
        
    def gamma(self):
        if self.buy_sell:
            return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        else:
            return - norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        if self.buy_sell:
            return self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        else:
            return - self.S * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self):
        if self.option_type == 'Call':
            if self.buy_sell:
                return (- (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
            else:
                return ( (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        else:
            if self.buy_sell:
                return (- (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
            else:
                return ( (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
    
    def rho(self):
        if self.option_type == 'Call':
            if self.buy_sell:
                return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
            else:
                return - self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:
            if self.buy_sell:
                return - self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
            else:
                return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
            