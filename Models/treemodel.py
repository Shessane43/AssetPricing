import numpy as np


class TrinomialTree:
    """
    Recombining Trinomial Tree (Boyle / Kamrad-Ritchken style)
    - European / American
    - With dividend yield q (continuous)
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

        m1 = np.exp((self.r - self.q) * self.dt)  # E[S_{t+dt}/S_t]
        a = self.u
        b = 1.0 / self.u

      
        if self.sigma == 0.0:
            # Deterministic growth
            self.pu, self.pm, self.pd = 0.0, 1.0, 0.0
        else:
            nu = (self.r - self.q - 0.5 * self.sigma**2)  # drift of log-price
            self.pu = 1.0 / 6.0 + (nu * np.sqrt(self.dt)) / (2.0 * self.sigma * np.sqrt(3.0))
            self.pm = 2.0 / 3.0
            self.pd = 1.0 - self.pu - self.pm

            # Safety (avoid negative probs due to extreme params)
            eps = 1e-12
            self.pu = max(eps, min(1.0 - eps, self.pu))
            self.pd = max(eps, min(1.0 - eps, self.pd))
            self.pm = max(eps, min(1.0 - eps, self.pm))
            s = self.pu + self.pm + self.pd
            self.pu, self.pm, self.pd = self.pu / s, self.pm / s, self.pd / s

    def payoff(self, S):
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        elif self.option_type == "put":
            return np.maximum(self.K - S, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def price(self):
        n = self.n_steps

        k = np.arange(-n, n + 1)
        S_T = self.S * (self.u ** k)

        V = self.payoff(S_T)

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
