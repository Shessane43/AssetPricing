import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import minimize, brentq

from Models.models import Model
from Models.blackscholes import BlackScholes


class HestonModel(Model):
    """
    Modèle de Heston (1993)

    dS_t/S_t = (r - q) dt + sqrt(v_t) dW1_t
    dv_t     = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t
    Corr(dW1, dW2) = rho
    """

    def __init__(
        self,
        S, K, r, T, q,
        option_type="call",
        position="buy",
        option_class="vanilla",
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7
    ):
        super().__init__(S, K, r, T, option_type, position, option_class)

        self.q = q
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.option_type = option_type.lower()
        self.option_class = option_class.lower()
    def _sign(self):
        return -1.0 if self.position == "sell" else 1.0

    def _r_adj(self):
        return self.r - self.q

    def _char_func(self, u, j):
    
        u = np.asarray(u, dtype=np.complex128)
        i = 1j
        tau = self.T
        x0 = np.log(self.S)

        u_j = 0.5 if j == 1 else -0.5
        b_j = (self.kappa - self.rho * self.sigma_v) if j == 1 else self.kappa

        a = self.kappa * self.theta
        sigma = self.sigma_v
        rho = self.rho

        # sqrt en complexe garanti
        d = np.sqrt((rho * sigma * i * u - b_j) ** 2 - sigma**2 * (2 * u_j * i * u - u**2) + 0j)

        # "trap-safe" g (important)
        g = (b_j - rho * sigma * i * u - d) / (b_j - rho * sigma * i * u + d)

        exp_dt = np.exp(-d * tau)

        # eps pour éviter divisions par ~0
        eps = 1e-14 + 0j
        one_minus_g = 1.0 - g
        one_minus_gexp = 1.0 - g * exp_dt

        # log en complexe (et stable)
        log_term = np.log((one_minus_gexp + eps) / (one_minus_g + eps))

        C = self._r_adj() * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u - d) * tau - 2.0 * log_term)
        D = (b_j - rho * sigma * i * u - d) * ((1.0 - exp_dt) / (sigma**2 * (one_minus_gexp + eps)))

        return np.exp(C + D * self.v0 + i * u * x0)

    
    def _Pj(self, j, n=64):
        x, w = laggauss(n)
        i = 1j
        lnK = np.log(self.K)

        u = (x + 1e-6).astype(np.complex128)  
        phi = self._char_func(u, j)

        integrand = np.real(np.exp(-i * u * lnK) * phi / (i * u))

        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        val = np.sum(w * np.exp(x) * integrand)
        return 0.5 + val / np.pi
    
    def price_closed_form(self, n=64):
        P1 = self._Pj(1, n)
        P2 = self._Pj(2, n)
        if not np.isfinite(P1) or not np.isfinite(P2):
            return self.price_mc(n_paths=200_000, n_steps=300)

        disc_r = np.exp(-self.r * self.T)
        disc_q = np.exp(-self.q * self.T)

        call = self.S * disc_q * P1 - self.K * disc_r * P2

        if self.option_type == "call":
            price = call
        else:
            price = call - self.S * disc_q + self.K * disc_r

        return self._sign() * price

    def simulate_paths(self, n_paths=50_000, n_steps=200, seed=None):
        rng = np.random.default_rng(seed)
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.S
        v[:, 0] = max(self.v0, 0.0)

        for t in range(1, n_steps + 1):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            dW1 = sqrt_dt * Z1
            dW2 = sqrt_dt * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)

            v_pos = np.maximum(v[:, t - 1], 0.0)

            v[:, t] = np.maximum(
                v[:, t - 1]
                + self.kappa * (self.theta - v_pos) * dt
                + self.sigma_v * np.sqrt(v_pos) * dW2,
                0.0
            )

            S[:, t] = S[:, t - 1] * np.exp(
                (self.r - self.q - 0.5 * v_pos) * dt
                + np.sqrt(v_pos) * dW1
            )

        return S, v

    def price_mc(self, n_paths=50_000, n_steps=200):
        S, _ = self.simulate_paths(n_paths, n_steps)
        ST = S[:, -1]

        if self.option_type == "call":
            payoff = np.maximum(ST - self.K, 0)
        else:
            payoff = np.maximum(self.K - ST, 0)

        price = np.exp(-self.r * self.T) * payoff.mean()
        return self._sign() * price

    def price(self, **kwargs):
        if self.option_class != "vanilla":
            return self.price_mc()
        return self.price_closed_form()

    # ---------------------------------------------------
    # Implied volatility (BS)
    # ---------------------------------------------------
    def implied_volatility(self, market_price):
        target = market_price * self._sign()

        def f(sig):
            bs = BlackScholes(
                self.S, self.K, self.r, sig, self.T, self.q,
                self.option_type, "buy", "vanilla"
            )
            return bs.price() - target

        try:
            return brentq(f, 1e-6, 5.0)
        except ValueError:
            return None

    # ---------------------------------------------------
    # Calibration
    # ---------------------------------------------------
    @staticmethod
    def calibrate(
        S, r, T, q,
        option_type, position,
        K_list, market_prices,
        initial_guess
    ):
        def objective(x):
            v0, kappa, theta, sigma_v, rho = x

            if min(x) <= 0 or abs(rho) >= 1:
                return 1e8

            model_prices = []
            for K in K_list:
                model = HestonModel(
                    S, K, r, T, q,
                    option_type, position, "vanilla",
                    v0, kappa, theta, sigma_v, rho
                )
                model_prices.append(model.price())

            err = np.array(model_prices) - np.array(market_prices)
            return np.mean(err**2)

        res = minimize(objective, initial_guess, method="Nelder-Mead")
        return res.x
