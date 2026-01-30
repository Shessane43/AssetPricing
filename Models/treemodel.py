import numpy as np


class TrinomialTree:
    """
    Recombining Trinomial Tree for option pricing
    (Boyle / Kamrad–Ritchken style).

    This class implements a recombining trinomial lattice for pricing
    European and American vanilla options. The model supports a continuous
    dividend yield and converges faster than the binomial tree for a
    comparable number of steps.

    Features
    --------
    - European and American exercise styles
    - Call and put options
    - Continuous dividend yield
    - Recombining lattice for computational efficiency
    """

    def __init__(
        self,
        S,
        K,
        r,
        T,
        sigma,
        option_type="call",
        exercise="european",
        n_steps=200,
        q=0.0,
    ):
        """
        Initialize the trinomial tree model.

        Parameters
        ----------
        S : float
            Initial price of the underlying asset.
        K : float
            Strike price of the option.
        r : float
            Risk-free interest rate (continuously compounded).
        T : float
            Time to maturity (in years).
        sigma : float
            Volatility of the underlying asset.
        option_type : str, optional
            Type of the option ("call" or "put"), default is "call".
        exercise : str, optional
            Exercise style ("european" or "american"), default is "european".
        n_steps : int, optional
            Number of time steps in the tree, default is 200.
        q : float, optional
            Continuous dividend yield, default is 0.0.

        Raises
        ------
        ValueError
            If invalid parameters are provided.
        """
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.q = float(q)
        self.T = float(T)
        self.sigma = float(sigma)
        self.n_steps = int(n_steps)

        self.option_type = option_type.lower()
        self.exercise = exercise.lower()

        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")

        self.dt = self.T / self.n_steps
        self.df = np.exp(-self.r * self.dt)

        # Recombining step size
        self.u = np.exp(self.sigma * np.sqrt(3.0 * self.dt))

        m1 = np.exp((self.r - self.q) * self.dt)  # E[S_{t+dt} / S_t]
        a = self.u
        b = 1.0 / self.u

        if self.sigma == 0.0:
            # Deterministic growth (no uncertainty)
            self.pu, self.pm, self.pd = 0.0, 1.0, 0.0
        else:
            # Drift of log-price
            nu = self.r - self.q - 0.5 * self.sigma**2

            self.pu = 1.0 / 6.0 + (nu * np.sqrt(self.dt)) / (2.0 * self.sigma * np.sqrt(3.0))
            self.pm = 2.0 / 3.0
            self.pd = 1.0 - self.pu - self.pm

            # Safety checks to avoid negative or invalid probabilities
            eps = 1e-12
            self.pu = max(eps, min(1.0 - eps, self.pu))
            self.pd = max(eps, min(1.0 - eps, self.pd))
            self.pm = max(eps, min(1.0 - eps, self.pm))

            s = self.pu + self.pm + self.pd
            self.pu, self.pm, self.pd = self.pu / s, self.pm / s, self.pd / s

    def payoff(self, S):
        """
        Compute the option payoff at maturity.

        Parameters
        ----------
        S : float or numpy.ndarray
            Underlying asset price(s).

        Returns
        -------
        float or numpy.ndarray
            Payoff of the option.

        Raises
        ------
        ValueError
            If the option type is invalid.
        """
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        elif self.option_type == "put":
            return np.maximum(self.K - S, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def price(self):
        """
        Compute the option price using the trinomial tree.

        Returns
        -------
        float
            Option price at time 0.

        Notes
        -----
        The method proceeds by backward induction. For American options,
        early exercise is checked at each node.
        """
        n = self.n_steps

        # Terminal asset prices
        k = np.arange(-n, n + 1)
        S_T = self.S * (self.u ** k)

        # Terminal payoffs
        V = self.payoff(S_T)

        # Backward induction
        for j in range(n, 0, -1):
            V = self.df * (
                self.pu * V[2:] +
                self.pm * V[1:-1] +
                self.pd * V[:-2]
            )

            if self.exercise == "american":
                k_prev = np.arange(-(j - 1), (j - 1) + 1)
                S_prev = self.S * (self.u ** k_prev)
                V = np.maximum(V, self.payoff(S_prev))

        return float(V[0])
