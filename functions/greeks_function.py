import numpy as np
import matplotlib.pyplot as plt

from functions.greeks_bs_function import Greeks_BS
from functions.greeks_heston_function import Greeks_Heston
from functions.greeks_gamma_variance_function import Greeks_VarianceGamma


class Greeks:
    """
    Unified Greeks interface for multiple option pricing models.

    This class acts as a wrapper around model-specific Greeks engines
    (Black–Scholes, Heston, Variance Gamma) and provides:
    - scalar Greeks (delta, gamma, vega, theta, rho)
    - Greeks curves as a function of spot price
    - plotting utilities for all Greeks

    Parameters
    ----------
    option_type : str
        Option type ("call" or "put").
    model : str
        Pricing model name.
        Supported values:
        - "Black-Scholes"
        - "Heston"
        - "Gamma Variance"
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (continuously compounded).
    sigma : float, optional
        Volatility (Black–Scholes, Variance Gamma).
    theta : float, optional
        Drift parameter (Variance Gamma).
    nu : float, optional
        Variance parameter (Variance Gamma).
    v0 : float, optional
        Initial variance (Heston).
    kappa : float, optional
        Mean reversion speed (Heston).
    theta_heston : float, optional
        Long-run variance (Heston).
    sigma_v : float, optional
        Volatility of variance (Heston).
    rho : float, optional
        Correlation between asset and variance (Heston).
    buy_sell : str, default "buy"
        Position side:
        - "buy"  → long position
        - "sell" → short position

    Notes
    -----
    - The sign of Greeks is handled inside each model-specific engine.
    - `option_type` and `buy_sell` are internally converted to lowercase.
    """

    def __init__(
        self,
        option_type,
        model,
        S, K, T, r,
        sigma=None,
        theta=None,
        nu=None,
        v0=None,
        kappa=None,
        theta_heston=None,
        sigma_v=None,
        rho=None,
        buy_sell="buy",
    ):
        # Core parameters
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)

        self.sigma = sigma
        self.theta_vg = theta
        self.nu = nu

        self.v0 = v0
        self.kappa = kappa
        self.theta_heston = theta_heston
        self.sigma_v = sigma_v
        self.rho_heston = rho

        self.option_type = option_type.lower()
        self.model = model
        self.buy_sell = buy_sell.lower()

    def _engine(self, S=None):
        """
        Instantiate and return the appropriate Greeks engine
        for the selected pricing model.

        Parameters
        ----------
        S : float, optional
            Spot price override (used for curve computations).

        Returns
        -------
        object
            Model-specific Greeks engine.
        """
        S = self.S if S is None else S

        if self.model == "Black-Scholes":
            return Greeks_BS(
                S, self.K, self.T, self.r,
                self.sigma, self.option_type, self.buy_sell
            )

        if self.model == "Heston":
            return Greeks_Heston(
                S, self.K, self.T, self.r,
                self.v0,
                self.kappa,
                self.theta_heston,
                self.sigma_v,
                self.rho_heston,
                self.option_type,
                self.buy_sell
            )

        if self.model == "Gamma Variance":
            return Greeks_VarianceGamma(
                S, self.K, self.r, self.T,
                self.sigma, self.theta_vg, self.nu,
                self.option_type, self.buy_sell
            )

        raise ValueError(f"Unknown model: {self.model}")

    def delta(self): return self._engine().delta()
    def gamma(self): return self._engine().gamma()
    def vega(self):  return self._engine().vega()
    def theta(self): return self._engine().theta()
    def rho(self):   return self._engine().rho()

    def _curve(self, f, points=0.3, n=100):
        """
        Compute a Greek curve as a function of spot price.

        Parameters
        ----------
        f : callable
            Function returning a Greek given spot price S.
        points : float
            Relative range around spot (e.g. 0.3 → ±30%).
        n : int
            Number of grid points.

        Returns
        -------
        tuple (S_grid, values)
            Spot grid and corresponding Greek values.
        """
        S_grid = np.linspace(self.S * (1 - points), self.S * (1 + points), n)
        values = [f(S) for S in S_grid]
        return S_grid, values

    def list_delta(self, points=0.3, n=100):
        """Return delta as a function of spot."""
        return self._curve(lambda S: self._engine(S).delta(), points, n)

    def list_gamma(self, points=0.3, n=100):
        """Return gamma as a function of spot."""
        return self._curve(lambda S: self._engine(S).gamma(), points, n)

    def list_vega(self, points=0.3, n=100):
        """Return vega as a function of spot."""
        return self._curve(lambda S: self._engine(S).vega(), points, n)

    def list_theta(self, points=0.3, n=100):
        """Return theta as a function of spot."""
        return self._curve(lambda S: self._engine(S).theta(), points, n)

    def list_rho(self, points=0.3, n=100):
        """Return rho as a function of spot."""
        return self._curve(lambda S: self._engine(S).rho(), points, n)

    # -------- Plotting --------
    def plot_all_greeks(self, points=0.3, n=100):
        """
        Plot Delta, Gamma, Vega, Theta and Rho as functions of spot price.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the five Greeks plots.
        """
        curves = [
            ("Delta", *self.list_delta(points, n), "tab:blue"),
            ("Gamma", *self.list_gamma(points, n), "tab:green"),
            ("Vega",  *self.list_vega(points, n),  "tab:red"),
            ("Theta", *self.list_theta(points, n), "tab:orange"),
            ("Rho",   *self.list_rho(points, n),   "tab:purple"),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        fig.patch.set_facecolor("#0e1117")

        for ax, (name, S, vals, color) in zip(axes, curves):
            ax.plot(S, vals, lw=2, color=color)
            ax.set_title(name, color="orange")
            ax.set_facecolor("#0e1117")
            ax.grid(True, linestyle="--", alpha=0.3)
            for side in ax.spines.values():
                side.set_color("orange")
            ax.tick_params(colors="orange")

        plt.tight_layout()
        return fig
