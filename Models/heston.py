import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import minimize, brentq

from Models.models import Model
from Models.blackscholes import BlackScholes


class HestonModel(Model):
    """
    Heston (1993)

    Vanilla:
        - Closed-form (Fourier / Laguerre)
        - Monte Carlo fallback if unstable

    Exotic (MC only):
        - Asian (fixed strike)
        - Lookback (fixed strike)
    """

    _LAG_CACHE = {}

    @staticmethod
    def _laguerre_nodes(n):
        if n not in HestonModel._LAG_CACHE:
            x, w = laggauss(n)
            HestonModel._LAG_CACHE[n] = (x.astype(float), w.astype(float))
        return HestonModel._LAG_CACHE[n]

    def __init__(
        self,
        S, K, r, T, q,
        option_type="call",        # call / put / asian_call / lookback_call ...
        position="long",           
        option_class="vanilla",    # vanilla / exotic
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7
    ):
        super().__init__(S, K, r, T, option_type, position, option_class)

        self.q = q
        self.v0 = max(v0, 1e-8)
        self.kappa = max(kappa, 1e-8)
        self.theta = max(theta, 1e-8)
        self.sigma_v = max(sigma_v, 1e-8)
        self.rho = np.clip(rho, -0.999, 0.999)

    # ---------------- Utilities ---------------- #

    def _sign(self):
        return -1.0 if self.position.lower() == "short" else 1.0

    def _r_adj(self):
        return self.r - self.q

    # ---------------- Characteristic Function ---------------- #

    def _char_func(self, u, j):
        u = np.asarray(u, dtype=np.complex128)
        i = 1j
        tau = float(self.T)
        x0 = np.log(self.S)

        u_j = 0.5 if j == 1 else -0.5
        b_j = (self.kappa - self.rho * self.sigma_v) if j == 1 else self.kappa

        a = self.kappa * self.theta
        sigma = self.sigma_v
        rho = self.rho

        eps = 1e-14 + 0j

        d = np.sqrt(
            (rho * sigma * i * u - b_j) ** 2
            - sigma**2 * (2 * u_j * i * u - u**2)
            + 0j
        )

        g = (b_j - rho * sigma * i * u - d) / (b_j - rho * sigma * i * u + d + eps)
        exp_dt = np.exp(-d * tau)

        one_minus_g = 1.0 - g
        one_minus_gexp = 1.0 - g * exp_dt

        log_term = np.log((one_minus_gexp + eps) / (one_minus_g + eps))

        C = (
            self._r_adj() * i * u * tau
            + (a / sigma**2)
            * ((b_j - rho * sigma * i * u - d) * tau - 2.0 * log_term)
        )

        D = (b_j - rho * sigma * i * u - d) * (
            (1.0 - exp_dt) / (sigma**2 * (one_minus_gexp + eps))
        )

        return np.exp(C + D * self.v0 + i * u * x0)

    # ---------------- Fourier Pricing ---------------- #

    def _Pj(self, j, n=32):
        x, w = self._laguerre_nodes(n)
        u = x + 1e-10
        phi = self._char_func(u, j)

        integrand = np.real(
            np.exp(-1j * u * np.log(self.K)) * phi / (1j * u)
        )

        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        return 0.5 + np.sum(w * np.exp(x) * integrand) / np.pi

    # ---------------- Monte Carlo ---------------- #

    def _simulate_paths(self, n_paths=40_000, n_steps=200, seed=42):
        rng = np.random.default_rng(seed)
        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = self.S
        v[:, 0] = self.v0

        for t in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2

            v_pos = np.maximum(v[:, t], 0.0)

            v[:, t + 1] = np.maximum(
                v[:, t]
                + self.kappa * (self.theta - v_pos) * dt
                + self.sigma_v * np.sqrt(v_pos) * sqrt_dt * Z2,
                0.0
            )

            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - 0.5 * v_pos) * dt
                + np.sqrt(v_pos) * sqrt_dt * Z1
            )

        return S

    def _price_mc_vanilla(self):
        paths = self._simulate_paths()
        ST = paths[:, -1]

        if self.option_type.lower() == "call":
            payoff = np.maximum(ST - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - ST, 0.0)

        return np.exp(-self.r * self.T) * payoff.mean()

    def _price_exotic_mc(self):
        paths = self._simulate_paths()
        ot = self.option_type.lower()

        if ot in ["asian_call", "call_asian"]:
            avgS = paths[:, 1:].mean(axis=1)
            payoff = np.maximum(avgS - self.K, 0.0)

        elif ot in ["asian_put", "put_asian"]:
            avgS = paths[:, 1:].mean(axis=1)
            payoff = np.maximum(self.K - avgS, 0.0)

        elif ot in ["lookback_call", "call_lookback"]:
            Smax = paths[:, 1:].max(axis=1)
            payoff = np.maximum(Smax - self.K, 0.0)

        elif ot in ["lookback_put", "put_lookback"]:
            Smin = paths[:, 1:].min(axis=1)
            payoff = np.maximum(self.K - Smin, 0.0)

        else:
            raise ValueError(f"Unsupported exotic payoff: {self.option_type}")

        return np.exp(-self.r * self.T) * payoff.mean()

    # ---------------- Public API ---------------- #

    def price(self, n=32):
        if self.option_class.lower() == "vanilla":
            try:
                P1 = self._Pj(1, n)
                P2 = self._Pj(2, n)

                if not (np.isfinite(P1) and np.isfinite(P2)):
                    raise FloatingPointError

                disc_r = np.exp(-self.r * self.T)
                disc_q = np.exp(-self.q * self.T)

                call = self.S * disc_q * P1 - self.K * disc_r * P2
                price = call if self.option_type.lower() == "call" else call - self.S * disc_q + self.K * disc_r

                return self._sign() * price

            except Exception:
                return self._sign() * self._price_mc_vanilla()

        return self._sign() * self._price_exotic_mc()

    def implied_volatility(self, price):
        if self.option_class.lower() != "vanilla":
            return np.nan

        target = price * self._sign()

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

        def objective_factory(n_quad: int):
            def obj(x):
                v0, kappa, theta, sigma_v, rho = x

                if enforce_feller and 2 * kappa * theta <= sigma_v ** 2:
                    return 1e6

                err = 0.0
                for K, T, pm in zip(K_list, T_list, P_mkt):
                    model = HestonModel(
                        S, K, r, T, q,
                        option_type=option_type,
                        position="long",
                        option_class="vanilla",
                        v0=v0, kappa=kappa, theta=theta,
                        sigma_v=sigma_v, rho=rho
                    )
                    p = model.price(n=n_quad)
                    if not np.isfinite(p) or p <= 0:
                        return 1e6

                    weight = 1.0 + 2.0 * abs(K / S - 1.0)
                    diff = weight * (p - pm)

                    if use_relative_error:
                        diff = diff / max(pm, 1e-6)

                    err += diff * diff

                return err / len(K_list)
            return obj

        # Stage 1 (coarse)
        res1 = minimize(
            objective_factory(n_coarse),
            np.array(initial_guess, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter_coarse), "ftol": 1e-9}
        )
        x1 = res1.x if res1.success else np.array(initial_guess, dtype=float)

        # Stage 2 (refine)
        res2 = minimize(
            objective_factory(n_refine),
            x1,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter_refine), "ftol": 1e-10}
        )

        return res2.x if res2.success else x1
