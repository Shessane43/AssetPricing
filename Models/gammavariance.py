import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model


class VarianceGamma(Model):
    """
    Variance Gamma model (Madan–Carr–Chang)
    """

    def __init__(
        self,
        S, K, r, T,
        sigma,
        theta,
        nu,
        option_type="call",
        position="long"   # conservé pour compatibilité, mais ignoré
    ):
        super().__init__(S, K, r, T, option_type, position, "vanilla")

        self.sigma = max(float(sigma), 1e-8)
        self.nu = max(float(nu), 1e-8)
        self.theta = float(theta)

    def char_func(self, u):
        u = np.asarray(u, dtype=np.complex128)
        i = 1j

        drift = np.log(self.S) + self.r * self.T

        denom = (
            1
            - i * u * self.theta * self.nu
            + 0.5 * self.sigma**2 * self.nu * u**2
        )

        return np.exp(i * u * drift) * denom ** (-self.T / self.nu)

    def price(self, n=64):
        if self.option_type not in ["call", "put"]:
            raise ValueError(
                "Variance Gamma pricing is only defined for vanilla call / put."
            )

        x, w = laggauss(n)
        u = x + 1e-10
        lnK = np.log(self.K)

        phi = self.char_func(u - 1j)

        integrand = np.real(
            np.exp(-1j * u * lnK) * phi / (1j * u)
        )

        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "put":
            call_price = call_price - self.S + self.K * np.exp(-self.r * self.T)

        # sécurité numérique
        return max(float(call_price), 0.0)
