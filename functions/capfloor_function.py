# capfloor_function.py
import numpy as np
from scipy.stats import norm

class CapFloor:
    def __init__(self, nominal, strike, maturity, frequency, vol, option_type="Cap"):
        """
        Parameters
        ----------
        nominal : float
            Notional amount of the cap or floor.
        strike : float
            Strike rate of the cap/floor.
        maturity : float
            Maturity of the contract in years.
        frequency : int
            Number of payment periods per year.
        vol : float
            Black implied volatility.
        option_type : str, optional
            Type of the option: "Cap" or "Floor" (default is "Cap").
        """
        self.nominal = nominal
        self.strike = strike
        self.maturity = maturity
        self.frequency = frequency
        self.vol = vol
        self.option_type = option_type

    def price_classic(self, forward_rate=0.03):
        """
        Prices the cap or floor using the Black 76 approximation.

        Parameters
        ----------
        forward_rate : float, optional
            Forward interest rate of the underlying (default is 0.03).

        Returns
        -------
        float
            Present value of the cap or floor.
        """
        P0T = 1 / (1 + forward_rate * self.frequency)
        T = self.maturity / self.frequency

        if self.vol == 0:
            # Degenerate case
            if self.option_type == "Cap":
                return max(forward_rate - self.strike, 0) * self.nominal
            else:
                return max(self.strike - forward_rate, 0) * self.nominal
        
        d1 = (np.log(forward_rate / self.strike) + 0.5 * self.vol**2 * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)

        if self.option_type == "Cap":
            price = P0T * (forward_rate * norm.cdf(d1) - self.strike * norm.cdf(d2)) * self.nominal
        else:
            price = P0T * (self.strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1)) * self.nominal

        return price
