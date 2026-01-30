import numpy as np
from numpy.polynomial.laguerre import laggauss
from Models.models import Model


class MertonJumpDiffusion(Model):
    """
    Merton Jump-Diffusion model for European option pricing.

    This class implements the Merton (1976) jump-diffusion model, in which
    the underlying asset price follows a diffusion process augmented by
    a compound Poisson jump component with normally distributed jump sizes.
    The model captures sudden large moves in asset prices while retaining
    a continuous Brownian component.
    """

    def __init__(
        self, S, K, r, T,
        sigma, lambd, mu_j, sigma_j,
        option_type="call"
    ):
        """
        Initialize a Merton Jump-Diffusion option pricing model.

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
            Volatility of the diffusion component.
        lambd : float
            Intensity (arrival rate) of the Poisson jump process.
        mu_j : float
            Mean of the jump size (log-jump).
        sigma_j : float
            Standard deviation of the jump size (log-jump).
        option_type : str, optional
            Type of the option ("call" or "put"), default is "call".

        Notes
        -----
        Lower bounds are imposed on volatility parameters to ensure
        numerical stability.
        """
        super().__init__(S, K, r, T, option_type, position="long", option_class="vanilla")

        self.sigma = max(float(sigma), 1e-8)
        self.lambd = max(float(lambd), 0.0)
        self.mu_j = float(mu_j)
        self.sigma_j = max(float(sigma_j), 1e-8)

    def char_func(self, u):
        """
        Compute the characteristic function of log(S_T) under the
        Merton Jump-Diffusion model.

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
        The characteristic function combines:
            - a Brownian diffusion component,
            - a compound Poisson jump component with normal jump sizes.
        """
        drift = self.r - self.lambd * (
            np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        )

        jump = np.exp(
            1j * u * self.mu_j
            - 0.5 * self.sigma_j**2 * u**2
        )

        return np.exp(
            1j * u * (np.log(self.S) + drift * self.T)
            - 0.5 * self.sigma**2 * u**2 * self.T
            + self.lambd * self.T * (jump - 1)
        )

    def price(self, n=64):
        """
        Compute the price of a European option under the Merton
        Jump-Diffusion model using Fourier inversion and
        Gauss–Laguerre quadrature.

        Parameters
        ----------
        n : int, optional
            Number of Gauss–Laguerre quadrature points (default is 64).

        Returns
        -------
        float
            Discounted option price, floored at zero.

        Notes
        -----
        The pricing formula is evaluated via Fourier inversion of the
        characteristic function. Put prices are obtained using
        put-call parity.
        """
        if self.option_type not in ["call", "put"]:
            return np.nan

        x, w = laggauss(n)
        lnK = np.log(self.K)

        phi = self.char_func(x - 1j)

        integrand = np.real(
            np.exp(-1j * x * lnK) * phi / (1j * x)
        )

        integrand = np.nan_to_num(integrand, nan=0.0)

        call_price = np.exp(-self.r * self.T) * np.sum(w * integrand) / np.pi

        if self.option_type == "put":
            price = call_price - self.S + self.K * np.exp(-self.r * self.T)
        else:
            price = call_price

        return max(price, 0.0)
