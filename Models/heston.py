import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import minimize, brentq
from Models.models import Model
from Models.blackscholes import BlackScholes


class HestonModel(Model):
    # Cache quadrature nodes/weights for speed
    _LAG_CACHE = {}

    @staticmethod
    def _laguerre_nodes(n: int):
        if n not in HestonModel._LAG_CACHE:
            x, w = laggauss(n)
            # store as float64
            HestonModel._LAG_CACHE[n] = (x.astype(float), w.astype(float))
        return HestonModel._LAG_CACHE[n]

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

    def _sign(self):
        return -1.0 if self.position == "sell" else 1.0

    def _r_adj(self):
        return self.r - self.q

    def _char_func(self, u, j):
        i = 1j
        tau = self.T
        x0 = np.log(self.S)

        u_j = 0.5 if j == 1 else -0.5
        b_j = self.kappa - self.rho * self.sigma_v if j == 1 else self.kappa
        a = self.kappa * self.theta
        sigma = self.sigma_v
        rho = self.rho
        d = np.sqrt((rho * sigma * i * u - b_j) ** 2 - sigma**2 * (2 * u_j * i * u - u**2) + 0j)

        g = (b_j - rho * sigma * i * u - d) / (b_j - rho * sigma * i * u + d)
        exp_dt = np.exp(-d * tau)

        eps = 1e-14 + 0j
        one_minus_g = 1.0 - g
        one_minus_gexp = 1.0 - g * exp_dt

        log_term = np.log((one_minus_gexp + eps) / (one_minus_g + eps))

        C = self._r_adj() * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u - d) * tau - 2.0 * log_term)
        D = (b_j - rho * sigma * i * u - d) * ((1.0 - exp_dt) / (sigma**2 * (one_minus_gexp + eps)))

        return np.exp(C + D * self.v0 + i * u * x0)

    def _Pj(self, j, n=32):
        x, w = self._laguerre_nodes(n)
        u = x + 1e-10
        phi = self._char_func(u, j)
        integrand = np.real(np.exp(-1j * u * np.log(self.K)) * phi / (1j * u))
        return 0.5 + np.sum(w * np.exp(x) * integrand) / np.pi
        
    def _simulate_heston_paths(self, n_paths=40_000, n_steps=200, seed=42):
        if seed is not None:
            np.random.seed(seed)

        dt = self.T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S
        v[:, 0] = self.v0

        Z1 = np.random.normal(size=(n_paths, n_steps))
        Z2 = np.random.normal(size=(n_paths, n_steps))
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

        for t in range(n_steps):
            v_pos = np.maximum(v[:, t], 0.0)

            v[:, t + 1] = (
                v[:, t]
                + self.kappa * (self.theta - v_pos) * dt
                + self.sigma_v * np.sqrt(v_pos * dt) * Z2[:, t]
            )

            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - 0.5 * v_pos) * dt
                + np.sqrt(v_pos * dt) * Z1[:, t]
            )

        return S

    def _price_exotic_mc(self, n_paths=40_000, n_steps=200):
        paths = self._simulate_heston_paths(n_paths, n_steps)

        # Asian payoff
        avgS = paths[:, 1:].mean(axis=1)

        if self.option_type.lower() in ["asian", "call", "asian_call"]:
            payoff = np.maximum(avgS - self.K, 0.0)
        elif self.option_type.lower() in ["asian_put", "put"]:
            payoff = np.maximum(self.K - avgS, 0.0)
        else:
            raise ValueError(f"Unsupported exotic payoff: {self.option_type}")

        return self._sign() * np.exp(-self.r * self.T) * payoff.mean()

    def price(self, n=32):
        if self.option_class.lower() == "vanilla":
            P1 = self._Pj(1, n=n)
            P2 = self._Pj(2, n=n)

    def implied_volatility(self, market_price):
        target = market_price * self._sign()

        def f(sig):
            bs = BlackScholes(
                self.S, self.K, self.r, sig, self.T, self.q,
                self.option_type, "buy", "vanilla"
            )
            return bs.price() - target

        try:
            return brentq(f, 1e-6, 3.0)
        except ValueError:
            return np.nan


    @staticmethod
    def calibrate(
        S, r, q, option_type,
        K_list, T_list, market_prices,
        initial_guess=(0.04, 1.5, 0.04, 0.30, -0.50),
        enforce_feller=True,
        maxiter_coarse=60,
        maxiter_refine=80,
        n_coarse=16,
        n_refine=32,
        use_relative_error=True
    ):
        """
        Returns params [v0, kappa, theta, sigma_v, rho]
        Two-stage calibration: coarse (fast) then refine.
        Objective: price error, optionally relative (recommended).
        """

        K_list = np.asarray(K_list, dtype=float)
        T_list = np.asarray(T_list, dtype=float)
        P_mkt = np.asarray(market_prices, dtype=float)

        # Filter invalid
        m = np.isfinite(K_list) & np.isfinite(T_list) & np.isfinite(P_mkt) & (K_list > 0) & (T_list > 0) & (P_mkt > 0)
        K_list, T_list, P_mkt = K_list[m], T_list[m], P_mkt[m]
        if len(K_list) < 8:
            return np.array(initial_guess, dtype=float)

        bounds = [
            (1e-4, 0.5),     # v0
            (0.1, 6.0),      # kappa
            (1e-4, 0.5),     # theta
            (0.05, 1.5),     # sigma_v
            (-0.95, 0.95),   # rho
        ]

        res = minimize(objective, initial_guess, method="Nelder-Mead")
        return res.x
    
    @staticmethod
    def calibrate_multi_maturity(
        S, r, q,
        option_type, position,
        data_by_maturity,
        initial_guess=(0.04, 2.0, 0.04, 0.30, -0.50),
        use_feller_penalty=True,
        method="L-BFGS-B",
        maxiter=300,
        weights_by_maturity=None
    ):
        """
        Calibration multi-maturité.

        data_by_maturity: list of dicts, each dict:
            {
              "T": float,
              "K_list": list[float],
              "market_prices": list[float]
            }

        weights_by_maturity: optional list of arrays same shape as market_prices per maturity
                             (e.g. inverse spread^2)
        """

        option_type = option_type.lower()

        def objective(x):
            v0, kappa, theta, sigma_v, rho = x

            # Hard invalid
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-0.999 < rho < 0.999):
                return 1e12

            # Soft Feller penalty
            penalty = 0.0
            if use_feller_penalty:
                feller = 2.0 * kappa * theta - sigma_v**2
                if feller < 0:
                    penalty += 1e6 * (feller**2)

            se_sum = 0.0
            w_sum = 0.0

            for j, slice_data in enumerate(data_by_maturity):
                T = float(slice_data["T"])
                K_list = slice_data["K_list"]
                mkt = np.asarray(slice_data["market_prices"], float)

                if weights_by_maturity is None:
                    w = np.ones_like(mkt)
                else:
                    w = np.asarray(weights_by_maturity[j], float)

                # Model prices for this maturity
                mod = np.empty_like(mkt)
                for i, K in enumerate(K_list):
                    model = HestonModel(
                        S, K, r, T, q,
                        option_type=option_type,
                        position=position,
                        option_class="vanilla",
                        v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho
                    )
                    p = model.price()
                    if not np.isfinite(p):
                        return 1e12
                    mod[i] = p

                err = mod - mkt
                se_sum += np.sum(w * err * err)
                w_sum += np.sum(w)

            return se_sum / max(w_sum, 1e-12) + penalty

        x0 = np.array(initial_guess, float)

        bounds = [
            (1e-6, 2.0),      # v0
            (1e-4, 50.0),     # kappa
            (1e-6, 2.0),      # theta
            (1e-4, 5.0),      # sigma_v
            (-0.999, 0.999)   # rho
        ]

        res = minimize(
            objective,
            x0=x0,
            method=method,
            bounds=bounds if method.upper() in ["L-BFGS-B", "TNC", "SLSQP"] else None,
            options={"maxiter": maxiter}
        )

        return res.x, res
