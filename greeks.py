import numpy as np
from scipy.stats import norm
from options import Options
from greeks_BS import Greeks_BS
from greeks_heston import Greeks_Heston

class Greeks(Options):
    def __init__(self, S, K, T, r, sigma, option_type, model, v = None, buyer = True):
        
        super().__init__(S, K, T, r, sigma, option_type, model, v, buyer)

    def delta(self):
        if self.model == "BS":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buyer)
            return greeks_bs.delta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buyer)
            return greeks_heston.delta()
    

    def gamma(self):
        if self.model == "BS":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buyer)
            return greeks_bs.gamma()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buyer)
            return greeks_heston.gamma()
        
    def vega(self):
        if self.model == "BS":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buyer)
            return greeks_bs.vega()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buyer)
            return greeks_heston.vega()
        
    def theta(self):
        if self.model == "BS":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buyer)
            return greeks_bs.theta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buyer)
            return greeks_heston.theta()
    
    def rho(self):
        if self.model == "BS":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buyer)
            return greeks_bs.rho()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buyer)
            return greeks_heston.rho()