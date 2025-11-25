from Models.heston import HestonLewis

class Greeks_Heston:

    def __init__(self, S, K, r, T, v0, kappa, theta, sigma_v, rho, option_type="call", buy_sell=True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r

        # Paramètres Heston
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0

        self.option_type = option_type
        self.buy_sell = buy_sell

    def delta(self, h=1e-4):
        C_plus = HestonLewis(self.S+h, self.K, self.r, self.T, self.v0,
                             self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        C_minus = HestonLewis(self.S-h, self.K, self.r, self.T, self.v0,
                              self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else: 
            return - (C_plus - C_minus) / (2*h)

    def gamma(self, h=1e-4):
        C_plus = HestonLewis(self.S+h, self.K, self.r, self.T, self.v0,
                             self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        C_mid = HestonLewis(self.S, self.K, self.r, self.T, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        C_minus = HestonLewis(self.S-h, self.K, self.r, self.T, self.v0,
                              self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        if self.buy_sell:    
            return (C_plus - 2*C_mid + C_minus) / h**2
        else:
            return - (C_plus - 2*C_mid + C_minus) / h**2

    def vega(self, h=1e-4):
        C_minus = HestonLewis(self.S, self.K, self.r, self.T, self.v0,
                              self.kappa, self.theta, self.sigma_v - h, self.rho, option_type=self.option_type).price()
        C_plus = HestonLewis(self.S, self.K, self.r, self.T, self.v0,
                             self.kappa, self.theta, self.sigma_v + h, self.rho, option_type=self.option_type).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else:
            return - (C_plus - C_minus) / (2*h)

    def theta(self, h=1e-4):
        C_minus = HestonLewis(self.S, self.K, self.r, self.T - h, self.v0,
                              self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        C_plus = HestonLewis(self.S, self.K, self.r, self.T + h, self.v0,
                             self.kappa, self.theta, self.sigma_v, self.rho, option_type=self.option_type).price()
        if self.buy_sell:
            return - (C_plus - C_minus) / (2*h)
        else:
            return (C_plus - C_minus) / (2*h)

    def rho(self, h=1e-4):
        C_minus = Greeks_Heston(self.S, self.K, self.T, self.r - h,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buy_sell).price()
        C_plus = Greeks_Heston(self.S, self.K, self.T, self.r + h,
                              self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                              self.option_type, self.buy_sell).price()
        if self.buy_sell:
            return (C_plus - C_minus) / (2*h)
        else:
            return - (C_plus - C_minus) / (2*h)