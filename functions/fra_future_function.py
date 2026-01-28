class FRAFuture:
    def __init__(self, nominal, forward_rate, strike, product_type="FRA", start=None, end=None):
        """
        FRA/Future pricing class.
        """
        self.nominal = nominal
        self.forward_rate = forward_rate
        self.strike = strike
        self.product_type = product_type  # "FRA" ou "Future"

        if product_type == "FRA":
            if start is None or end is None:
                raise ValueError("FRA requires start and end dates")
            self.start = start
            self.end = end
            self.delta = end - start
        else:
            # Future n'a pas besoin de start/end
            self.start = None
            self.end = None
            self.delta = 0

    def price_classic(self):
        """Compute the classic price of FRA or Future."""
        if self.product_type == "FRA":
            # FRA PV
            PV = self.nominal * (self.forward_rate - self.strike) * self.delta / (1 + self.delta * self.forward_rate)
        elif self.product_type == "Future":
            # Future PV approximé (Mark-to-market)
            PV = self.nominal * (self.forward_rate - self.strike)
        else:
            raise ValueError("product_type must be 'FRA' or 'Future'")
        return PV

    def metrics(self, dr=1e-4):
        """Compute Delta. Vega n'existe plus pour FRA/Future."""
        price = self.price_classic()

        # Delta wrt forward rate
        fra_up = FRAFuture(self.nominal, self.forward_rate + dr, self.strike, self.product_type, self.start, self.end)
        fra_down = FRAFuture(self.nominal, self.forward_rate - dr, self.strike, self.product_type, self.start, self.end)
        delta = (fra_up.price_classic() - fra_down.price_classic()) / (2 * dr)

        vega = 0.0  # plus de volatilité pour FRA/Future
        return price, delta, vega
