from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all option pricing models.

    This class defines the common interface shared by all pricing models
    in the library. Any concrete model must inherit from this class and
    implement at least the `price` method.

    Attributes
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (continuously compounded).
    T : float
        Time to maturity (in years).
    option_type : str
        Type of the option payoff (e.g. "call", "put").
    position : str
        Position type ("buy"/"long" or "sell"/"short").
    option_class : str
        Option class (e.g. "vanilla", "exotic").
    """

    def __init__(
        self,
        S,
        K,
        r,
        T,
        option_type="call",
        position="buy",
        option_class="vanille"
    ):
        """
        Initialize the base option pricing model.

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
        option_type : str, optional
            Type of the option payoff ("call" or "put").
        position : str, optional
            Position type ("buy"/"long" or "sell"/"short").
        option_class : str, optional
            Option class ("vanilla" or "exotic").

        Notes
        -----
        All string parameters are converted to lowercase to ensure
        consistency across models.
        """
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.option_class = option_class.lower()

    @abstractmethod
    def price(self, **kwargs):
        """
        Compute the option price.

        Returns
        -------
        float
            Option price according to the chosen model.

        Notes
        -----
        This method must be implemented by all subclasses.
        """
        pass
