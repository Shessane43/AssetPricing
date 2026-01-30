import numpy as np
from scipy.stats import norm
from Models.models import Model

class Bachelier(Model):
    """
    Bachelier Model for European Option Pricing.

    This class implements the Bachelier (normal) model for pricing European
    vanilla options. In the Bachelier framework, the underlying asset price
    follows a normal diffusion rather than a lognormal one, making it suitable
    for assets that can take negative values (e.g. interest rates).

    The option price is computed using the closed-form Bachelier formula
    for European call and put options.
    """

    def __init__(self, S, K, r, T, sigma, option_type="call"):
        """
        Initialize a Bachelier option pricing model.

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
            Volatility of the underlying asset under the Bachelier model.
        option_type : str, optional
            Type of the option, either "call" or "put" (default is "call").

        Notes
        -----
        A lower bound is imposed on the volatility to avoid numerical issues
        when sigma is close to zero.
        """
        super().__init__(S, K, r, T, option_type, position="long", option_class="vanilla")
        self.sigma = max(float(sigma), 1e-8)

    def price(self):
        """
        Compute the price of the European option using the Bachelier formula.

        Returns
        -------
        float
            The discounted option price. The price is floored at zero
            to avoid negative option values.

        Notes
        -----
        The pricing formula is given by:

            C = exp(-rT) * [ (S - K) * Φ(d) + σ√T * φ(d) ]

        where:
            d = (S - K) / (σ√T),
            Φ is the standard normal CDF,
            φ is the standard normal PDF.

        The put price is obtained via put-call parity.
        """
        d = (self.S - self.K) / (self.sigma * np.sqrt(self.T))

        call_price = np.exp(-self.r * self.T) * (
            (self.S - self.K) * norm.cdf(d)
            + self.sigma * np.sqrt(self.T) * norm.pdf(d)
        )

        if self.option_type == "put":
            price = (call_price - self.S + self.K * np.exp(-self.r * self.T)) * self.S
        else:
            price = call_price * self.S

        return max(price, 0.0)
