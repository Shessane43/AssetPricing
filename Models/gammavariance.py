import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model


class VarianceGamma(Model):
    """
    Variance Gamma (VG) model for European option pricing.

    This class implements the Variance Gamma model introduced by
    Madan, Carr, and Chang. The VG process is a pure-jump Lévy process
    obtained by time-changing a Brownian motion with drift using a
    Gamma process. It allows for skewness and excess kurtosis in
    asset returns, going beyond the Black-Scholes framework.
    """

    def __init__(
        self,
        S, K, r, T,
        sigma,
        theta,
        nu,
        option_type="call",
        position="long"
    ):
        """
        Initialize a Variance Gamma option pricing model.

        Parameters
        ----------
        S : float
            Current price of the underlying asset.
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (continuously compounded).
        T : float
            Time to maturity (in years).
        sigma : float
            Volatility parameter of the Brownian component.
        theta : float
            Drift parameter of the Brownian motion.
        nu : float
            Variance rate of the Gamma subordinator.
        option_type : str, optional
            Type of the option ("call" or "put"), default is "call".
        position : str, optional
            Position in the option ("long" or "short"), default is "long".

        Notes
        -----
        Lower bounds are imposed on `sigma` and `nu` to avoid numerical
        instabilities in the characteristic function.
        """
        super().__init__(S, K, r, T, option_type, position, "vanilla")

        self.sigma = max(float(sigma), 1e-8)
        self.nu = max(float(nu), 1e-8)
        self.theta = float(theta)

    def char_func(self, u):
        """
        Compute the characteristic function of log(S_T) under the
        Variance Gamma model.

        Parameters
        ----------
        u : array_like or complex
            Argument of the characteristic function.

        Returns
        -------
        numpy.ndarray
            Value of the characteristic function evaluated at `u`.

        Notes
        -----
        The characteristic function is given by:

            φ(u) = exp(iu (ln S + rT))
                   * [1 - iθνu + 0.5σ²νu²]^(-T/ν)

        where σ, θ, and ν are the VG parameters.
        """
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
        """
        Compute the price of a European option under the Variance Gamma model
        using Fourier inversion with Gauss–Laguerre quadrature.

        Parameters
        ----------
        n : int, optional
            Number of Gauss–Laguerre quadrature points (default is 64).

        Returns
        -------
        float
            Discounted option price, floored at zero.

        Raises
        ------
        ValueError
            If the option type is not "call" or "put".

        Notes
        -----
        The option price is computed via the Carr–Madan Fourier pricing
        approach, where the integral is evaluated using Gauss–Laguerre
        quadrature for numerical stability and efficiency.
        """
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

        integrand = np.nan_to_num(
            integrand, nan=0.0, posinf=0.0, neginf=0.0
        )

        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "put":
            call_price = call_price - self.S + self.K * np.exp(-self.r * self.T)

        return max(float(call_price), 0.0)
