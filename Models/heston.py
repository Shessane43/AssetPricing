import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import brentq, minimize
from Models.models import Model
from Models.blackscholes import BlackScholes


class HestonModel(Model):
    """
    Modèle de Heston avec :
    - Pricing Fourier (Lewis)
    - Monte Carlo (automatique si option exotique)
    - Vol implicite, approximation Lewis
    - Calibration Nelder-Mead
    """

    def __init__(self, params, v0, kappa, theta, sigma_v, rho):
        self.S = params.S
        self.K = params.K
        self.T = params.T
        self.r = params.r
        self.option_type = params.option_type.lower()      # "call" / "put"
        self.position = params.position.lower()            # "buy" / "sell"
        self.option_style = params.option_class.lower()    # "vanille" / "exotique"

        # Paramètres du modèle Heston
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def char_func(self, u):
        a = self.kappa * self.theta
        b = self.kappa
        sigma = self.sigma_v

        d = np.sqrt((self.rho * sigma * 1j * u - b) ** 2 + sigma ** 2 * (1j * u + u ** 2))
        g = (b - self.rho * sigma * 1j * u - d) / (b - self.rho * sigma * 1j * u + d)

        C = self.r * 1j * u * self.T + (a / sigma ** 2) * (
            (b - self.rho * sigma * 1j * u - d) * self.T - 2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))
        )
        D = (b - self.rho * sigma * 1j * u - d) * (1 - np.exp(-d * self.T)) / (
            sigma ** 2 * (1 - g * np.exp(-d * self.T))
        )

        return np.exp(C + D * self.v0 + 1j * u * np.log(self.S))

    def price(self, n=64):
        if self.option_style == "exotique":
            return self.price_mc()  # Monte Carlo auto si exotique

        x, w = laggauss(n)
        integrand = np.exp(-x) * np.real(
            np.exp(-1j * x * np.log(self.K)) * self.char_func(x - 1j) / (1j * x)
        )
        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        price = call_price if self.option_type == "call" else call_price - self.S + self.K * np.exp(-self.r * self.T)
        return -price if self.position == "sell" else price

    def simulate_paths(self, n_paths=10000, n_steps=200):
        dt = self.T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.S
        v[:, 0] = max(self.v0, 1e-8)

        for t in range(1, n_steps + 1):
            Z1 = np.random.normal(size=n_paths)
            Z2 = np.random.normal(size=n_paths)

            W1 = np.sqrt(dt) * Z1
            W2 = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2)

            v[:, t] = np.maximum(
                v[:, t - 1] + self.kappa * (self.theta - v[:, t - 1]) * dt +
                self.sigma_v * np.sqrt(v[:, t - 1]) * W2,
                1e-8
            )

            S[:, t] = S[:, t - 1] * np.exp(
                (self.r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1
            )

        return S, v

    def price_mc(self, n_paths=10000, n_steps=200):
        S, _ = self.simulate_paths(n_paths, n_steps)
        payoff = np.maximum(S[:, -1] - self.K, 0) if self.option_type == "call" else np.maximum(self.K - S[:, -1], 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return -price if self.position == "sell" else price

    def implied_volatility(self, price):
        #Newton-Raphsoon pour vol implicite via Black-Scholes
        try:
            return brentq(
                lambda vol: BlackScholes(self.S, self.K, self.r, vol, self.T, self.option_type).price() - price,
                1e-6, 5
            )
        except ValueError:
            return None

    def lewis_approx_vol(self):
        return np.sqrt(self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * self.T))

    @staticmethod
    def calibrate(params, K_list, market_prices, initial_guess):
        #Algorithme de Nelder-mead (optimisation des paramètres)
        def objective(opt_params):
            kappa, theta, sigma_v, rho, v0 = opt_params
            prices = [
                HestonModel(params=ModelParameters(params.S, K, params.T, params.r, params.option_type, params.position, params.option_class),
                            v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho).price()
                for K in K_list
            ]
            return np.mean((np.array(prices) - np.array(market_prices)) ** 2)

        res = minimize(objective, initial_guess, method="Nelder-Mead")
        return res.x
