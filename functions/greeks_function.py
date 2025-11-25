import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functions.greeks_gamma_variance_function import Greeks_VarianceGamma
from functions.greeks_bs_function import Greeks_BS
from functions.greeks_heston_function import Greeks_Heston

class Greeks():
    def __init__(self, option_type, model, S, K, T, r, sigma=None, theta=None, nu=None, v0=None, kappa=None, theta_heston=None, sigma_v=None, rho=None, buy_sell = True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta_heston
        self.sigma_v = sigma_v
        self.rho = rho
        self.option_type = option_type
        self.model = model
        if buy_sell == "Buy":
            self.buy_sell = True
        else:
            self.buy_sell = False

    def delta(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.delta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.delta()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.delta()
    
    def lists_delta(self, points = 0.3, number = 100, h=1e-4):
        delta_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta, self.nu, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho,
                            self.buy_sell)
            delta_list.append(greeks.delta())
        return S_values, delta_list
    
    def gamma(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.gamma()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.gamma()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.gamma()
        
    def lists_gamma(self, points = 0.3, number= 100, h=1e-4):
        gamma_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta, self.nu, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho,
                            self.buy_sell)
            gamma_list.append(greeks.gamma())
        return S_values, gamma_list
    
    def vega(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.vega()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.vega()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.vega()

    def lists_vega(self, points = 0.3, number= 100, h=1e-4):
        vega_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta, self.nu, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho,
                            self.buy_sell)
            vega_list.append(greeks.vega())
        return S_values, vega_list
    
    def theta(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.theta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.theta()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.theta()
    
    def lists_theta(self, points = 0.3, number= 100, h=1e-4):
        theta_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta, self.nu, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho,
                            self.buy_sell)
            theta_list.append(greeks.theta())
        return S_values, theta_list

    def rho(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.rho()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.rho()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.rho()
    
    def lists_rho(self, points = 0.3, number= 100, h=1e-4):
        rho_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta, self.nu, self.v0,
                            self.kappa, self.theta, self.sigma_v, self.rho,
                            self.buy_sell)
            rho_list.append(greeks.rho())
        return S_values, rho_list
    
    def plot_all_greeks(self, points=0.3, number=100, h=1e-4):

        S_delta, delta_values = self.lists_delta(points, number, h)
        S_gamma, gamma_values = self.lists_gamma(points, number, h)
        S_vega, vega_values = self.lists_vega(points, number, h)
        S_theta, theta_values = self.lists_theta(points, number, h)
        S_rho, rho_values = self.lists_rho(points, number, h)

        plt.figure(figsize=(12, 8))

        plt.plot(S_delta, delta_values, label="Delta")
        plt.plot(S_gamma, gamma_values, label="Gamma")
        plt.plot(S_vega, vega_values, label="Vega")
        plt.plot(S_theta, theta_values, label="Theta")
        plt.plot(S_rho, rho_values, label="Rho")

        plt.title("Greeks vs Underlying Price S")
        plt.xlabel("Underlying Price S")
        plt.ylabel("Greeks")
        plt.grid(True)
        plt.legend()
        plt.show()
