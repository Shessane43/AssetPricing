from Models.heston import HestonModel


class Greeks_Heston:
    """
    Universal Heston Greeks via finite differences.

    - Vanilla options: finite differences on semi-closed Heston pricer.
    - Exotic options: Monte Carlo + finite differences.

    Supports all option types (call, put, asian_*, lookback_*).
    """

    def __init__(
        self,
        S, K, T, r,
        v0, kappa, theta_v, sigma_v, rho_heston,
        option_type="call",
        position="buy",
        q=0.0
    ):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)

        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta_v = float(theta_v)
        self.sigma_v = float(sigma_v)
        self.rho_heston = float(rho_heston)

        self.option_type = option_type.lower()
        self.position = position.lower()

        self.option_class = (
            "vanilla" if self.option_type in ["call", "put"] else "exotic"
        )

    def _price(self, **kwargs):
        """
        Price wrapper for finite differences.
        Automatically switches to Monte Carlo for exotic options.
        """
        model = HestonModel(
            S=kwargs.get("S", self.S),
            K=self.K,
            r=kwargs.get("r", self.r),
            T=kwargs.get("T", self.T),
            q=self.q,
            option_type=self.option_type,
            position=self.position,
            option_class=self.option_class,   
            v0=kwargs.get("v0", self.v0),
            kappa=self.kappa,
            theta=self.theta_v,
            sigma_v=kwargs.get("sigma_v", self.sigma_v),
            rho=kwargs.get("rho", self.rho_heston),
        )
        return model.price()

    def delta(self, h=None):
        """
        Delta = sensitivity to underlying price S
        """
        h = h or max(1e-4 * self.S, 1e-2)
        return (self._price(S=self.S + h) - self._price(S=self.S - h)) / (2 * h)

    def gamma(self, h=None):
        """
        Gamma = second derivative with respect to underlying S
        """
        h = h or max(1e-4 * self.S, 1e-2)
        return (
            self._price(S=self.S + h)
            - 2 * self._price()
            + self._price(S=self.S - h)
        ) / h**2

    def vega(self, h=1e-4):
        """
        Vega = sensitivity to initial variance v0.
        Note: this is NOT the implied-volatility vega.
        """
        return (self._price(v0=self.v0 + h) - self._price(v0=self.v0 - h)) / (2 * h)

    def theta(self, h=None):
        """
        Theta = sensitivity to time to maturity T
        """
        h = h or 1 / 365
        Tp = self.T + h
        Tm = max(1e-6, self.T - h)
        dPdT = (self._price(T=Tp) - self._price(T=Tm)) / (2 * h)
        return -dPdT

    def rho(self, h=1e-4):
        """
        Rho = sensitivity to risk-free rate r
        """
        return (self._price(r=self.r + h) - self._price(r=self.r - h)) / (2 * h)
