class FRAFuture:
    def __init__(self, nominal, forward_rate, strike, product_type="FRA", start=None, end=None):
        """
        Parameters
        ----------
        nominal : float
            Notional amount of the FRA or Future.
        forward_rate : float
            Forward interest rate of the underlying.
        strike : float
            Strike rate of the FRA or Future.
        product_type : str, optional
            Type of the product: "FRA" or "Future" (default is "FRA").
        start : float, optional
            Start time of the FRA period in years (required for FRA).
        end : float, optional
            End time of the FRA period in years (required for FRA).
        """
        self.nominal = nominal
        self.forward_rate = forward_rate
        self.strike = strike
        self.product_type = product_type  

        if product_type == "FRA":
            if start is None or end is None:
                raise ValueError("FRA requires start and end dates")
            self.start = start
            self.end = end
            self.delta = end - start
        else:
            self.start = None
            self.end = None
            self.delta = 0

    def price_classic(self):
        """
        Computes the classic price of a FRA or an interest rate future.

        Returns
        -------
        float
            Present value of the FRA or Future.
            Positive value corresponds to a long position when the forward rate
            is above the strike.
        """
        if self.product_type == "FRA":
            PV = (
                self.nominal
                * (self.forward_rate - self.strike)
                * self.delta
                / (1 + self.delta * self.forward_rate)
            )
        elif self.product_type == "Future":
            PV = self.nominal * (self.forward_rate - self.strike)
        else:
            raise ValueError("product_type must be 'FRA' or 'Future'")
        return PV
