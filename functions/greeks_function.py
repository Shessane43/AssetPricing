import numpy as np
import matplotlib.pyplot as plt

from functions.greeks_bs_function import Greeks_BS
from functions.greeks_heston_function import Greeks_Heston
from functions.greeks_gamma_variance_function import Greeks_VarianceGamma


class Greeks:
    """
    Wrapper class for computing option Greeks across multiple models:
    - Black-Scholes
    - Heston
    - Gamma Variance (Variance Gamma)
    
    Provides:
        - Single-value Greeks: delta, gamma, vega, theta, rho
        - Greek curves vs underlying spot
        - Plotting of all Greeks
    """

    def __init__(
        self,
        option_type,
        model,
        S, K, T, r,
        sigma=None,           # For Black-Scholes and VG
        theta=None,           # Theta parameter for VG
        nu=None,              # Nu parameter for VG
        v0=None,              # Initial variance for Heston
        kappa=None,           # Heston mean reversion
        theta_heston=None,    # Heston long-term variance
        sigma_v=None,         # Heston vol-of-vol
        rho=None,             # Heston correlation
        buy_sell="buy",       # "buy"/"long" or "sell"/"short"
    ):
        # --- Core option parameters ---
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)

        # --- Model-specific parameters ---
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
        Internal engine to select the correct Greeks computation object
        depending on the model.

        Args:
            S (float): spot price to use (optional, defaults to self.S)

        Returns:
            Model-specific Greeks object (Greeks_BS, Greeks_Heston, or Greeks_VarianceGamma)
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

    # --- Single-value Greeks ---
    def delta(self): return self._engine().delta()
    def gamma(self): return self._engine().gamma()
    def vega(self):  return self._engine().vega()
    def theta(self): return self._engine().theta()
    def rho(self):   return self._engine().rho()

    # --- Internal helper to generate Greek curves ---
    def _curve(self, f, points=0.3, n=100):
        """
        Generate a curve of Greek values as spot varies around S.

        Args:
            f (callable): function returning a Greek value for a given S
            points (float): percentage range around S (default ±30%)
            n (int): number of points in curve

        Returns:
            S_grid (array): underlying spot grid
            values (array): Greek values
        """
        S_grid = np.linspace(self.S * (1 - points), self.S * (1 + points), n)
        values = [f(S) for S in S_grid]
        return S_grid, values

    # --- Greek curves ---
    def list_delta(self, points=0.3, n=100):
        return self._curve(lambda S: self._engine(S).delta(), points, n)

    def list_gamma(self, points=0.3, n=100):
        return self._curve(lambda S: self._engine(S).gamma(), points, n)

    def list_vega(self, points=0.3, n=100):
        return self._curve(lambda S: self._engine(S).vega(), points, n)

    def list_theta(self, points=0.3, n=100):
        return self._curve(lambda S: self._engine(S).theta(), points, n)

    def list_rho(self, points=0.3, n=100):
        return self._curve(lambda S: self._engine(S).rho(), points, n)

    # --- Plot all Greeks ---
    def plot_all_greeks(self, points=0.3, n=100):
        """
        Plot all main Greeks (Delta, Gamma, Vega, Theta, Rho) in one figure.
        Dark theme with colored curves.

        Args:
            points (float): percentage range around S (default ±30%)
            n (int): number of points per curve

        Returns:
            matplotlib.figure.Figure: figure containing 5 Greek plots
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
