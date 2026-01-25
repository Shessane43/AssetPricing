from Models.heston import HestonModel


class Greeks_Heston:

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

    def _price(self, **kwargs):
        return HestonModel(
            S=kwargs.get("S", self.S),
            K=self.K,
            r=kwargs.get("r", self.r),
            T=kwargs.get("T", self.T),
            q=self.q,
            option_type=self.option_type,
            position=self.position,
            option_class="vanilla",
            v0=kwargs.get("v0", self.v0),
            kappa=self.kappa,
            theta=self.theta_v,
            sigma_v=kwargs.get("sigma_v", self.sigma_v),
            rho=kwargs.get("rho", self.rho_heston),
        ).price()

    def delta(self, h=1e-4):
        return (self._price(S=self.S + h) - self._price(S=self.S - h)) / (2 * h)

    def gamma(self, h=1e-4):
        return (
            self._price(S=self.S + h)
            - 2 * self._price()
            + self._price(S=self.S - h)
        ) / h**2

    def vega(self, h=1e-4):
        return (self._price(v0=self.v0 + h) - self._price(v0=self.v0 - h)) / (2 * h)

    def theta(self, h=1e-4):
        return -(self._price(T=self.T + h) - self._price(T=self.T - h)) / (2 * h)

    def rho(self, h=1e-4):
        return (self._price(r=self.r + h) - self._price(r=self.r - h)) / (2 * h)
