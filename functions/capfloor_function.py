# capfloor_function.py
import numpy as np
from scipy.stats import norm

class CapFloor:
    def __init__(self, nominal, strike, maturity, frequency, vol, option_type="Cap"):
        self.nominal = nominal
        self.strike = strike
        self.maturity = maturity
        self.frequency = frequency
        self.vol = vol
        self.option_type = option_type

    def price_classic(self, forward_rate=0.03):
        """
        Price using Black 76 approximation for Cap/Floor.
        forward_rate: forward rate for the underlying.
        """
        P0T = 1 / (1 + forward_rate*self.frequency)
        T = self.maturity / self.frequency
        if self.vol == 0:
            # Degenerate case
            if self.option_type=="Cap":
                return max(forward_rate - self.strike, 0) * self.nominal
            else:
                return max(self.strike - forward_rate, 0) * self.nominal
        
        d1 = (np.log(forward_rate/self.strike) + 0.5*self.vol**2*T) / (self.vol*np.sqrt(T))
        d2 = d1 - self.vol*np.sqrt(T)
        if self.option_type=="Cap":
            price = P0T * (forward_rate*norm.cdf(d1) - self.strike*norm.cdf(d2)) * self.nominal
        else:
            price = P0T * (self.strike*norm.cdf(-d2) - forward_rate*norm.cdf(-d1)) * self.nominal
        return price
    
    def capfloor_metrics(capfloor, forward_rate=0.03, dr=1e-4, dvol=1e-4):
        price = capfloor.price_classic(forward_rate)
    
        # Delta
        price_up = capfloor.price_classic(forward_rate + dr)
        price_down = capfloor.price_classic(forward_rate - dr)
        delta = (price_up - price_down) / (2*dr)
    
        # Vega
        vol_up = CapFloor(capfloor.nominal, capfloor.strike, capfloor.maturity, capfloor.frequency, capfloor.vol + dvol, capfloor.option_type)
        vol_down = CapFloor(capfloor.nominal, capfloor.strike, capfloor.maturity, capfloor.frequency, capfloor.vol - dvol, capfloor.option_type)
        vega = (vol_up.price_classic(forward_rate) - vol_down.price_classic(forward_rate)) / (2*dvol)
    
        return price, delta, vega
