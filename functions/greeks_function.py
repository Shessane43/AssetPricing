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
        self.theta_classic = theta
        self.nu = nu
        self.v0 = v0
        self.kappa = kappa
        self.theta_heston = theta_heston
        self.sigma_v = sigma_v
        self.rho_value = rho
        self.option_type = option_type
        self.model = model
        self.buy_sell = buy_sell

    def delta(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.delta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta_heston, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.delta()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta_classic, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.delta()
    
    def lists_delta(self, points = 0.3, number = 100, h=1e-4):
        delta_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta_classic, self.nu, self.v0,
                            self.kappa, self.theta_heston, self.sigma_v, self.rho_value,
                            self.buy_sell)
            delta_list.append(greeks.delta())
        return S_values, delta_list
    
    def gamma(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.gamma()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta_heston, self.sigma_v, self.rho, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.gamma()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta_classic, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.gamma()
        
    def lists_gamma(self, points = 0.3, number= 100, h=1e-4):
        gamma_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta_classic, self.nu, self.v0,
                            self.kappa, self.theta_heston, self.sigma_v, self.rho_value,
                            self.buy_sell)
            gamma_list.append(greeks.gamma())
        return S_values, gamma_list
    
    def vega(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.vega()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                          self.kappa, self.theta_heston, self.sigma_v, self.rho_value, self.v0,
                                          self.option_type, self.buy_sell)
            return greeks_heston.vega()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                             self.sigma, self.theta_classic, self.nu,
                                             self.option_type, self.buy_sell)
            return greeks_vg.vega()

    def lists_vega(self, points = 0.3, number= 100, h=1e-4):
        vega_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta_classic, self.nu, self.v0,
                            self.kappa, self.theta_heston, self.sigma_v, self.rho_value,
                            self.buy_sell)
            vega_list.append(greeks.vega())
        return S_values, vega_list
    
    def theta(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.theta()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                        self.kappa, self.theta_heston, self.sigma_v, self.rho_value, self.v0,
                                        self.option_type, self.buy_sell)
            return greeks_heston.theta()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                            self.sigma, self.theta_classic, self.nu,
                                            self.option_type, self.buy_sell)
            return greeks_vg.theta()


    
    def lists_theta(self, points=0.3, number=100, h=1e-4):
        theta_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        original_S = self.S
        for S in S_values:
            self.S = S
            theta_list.append(self.theta())
        self.S = original_S
        return S_values, theta_list


    def rho(self):
        if self.model == "Black-Scholes":
            greeks_bs = Greeks_BS(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.buy_sell)
            return greeks_bs.rho()
        elif self.model == "Heston":
            greeks_heston = Greeks_Heston(self.S, self.K, self.T, self.r,
                                        self.kappa, self.theta_heston, self.sigma_v, self.rho_value, self.v0,
                                        self.option_type, self.buy_sell)
            return greeks_heston.rho()
        elif self.model == "Gamma Variance":
            greeks_vg = Greeks_VarianceGamma(self.S, self.K, self.r, self.T,
                                            self.sigma, self.theta_classic, self.nu,
                                            self.option_type, self.buy_sell)
            return greeks_vg.rho()

    
    def lists_rho(self, points = 0.3, number= 100, h=1e-4):
        rho_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            greeks = Greeks(self.option_type, self.model, S, self.K, self.T, self.r,
                            self.sigma, self.theta_classic, self.nu, self.v0,
                            self.kappa, self.theta_heston, self.sigma_v, self.rho_value,
                            self.buy_sell)
            rho_list.append(greeks.rho())
        return S_values, rho_list
    
    def plot_all_greeks(self, points=0.3, number=100, h=1e-4):
        greeks_list = [
            ("Delta", self.lists_delta(points, number, h)[1], self.lists_delta(points, number, h)[0], "blue"),
            ("Gamma", self.lists_gamma(points, number, h)[1], self.lists_gamma(points, number, h)[0], "green"),
            ("Vega", self.lists_vega(points, number, h)[1], self.lists_vega(points, number, h)[0], "red"),
            ("Theta", self.lists_theta(points, number, h)[1], self.lists_theta(points, number, h)[0], "orange"),
            ("Rho", self.lists_rho(points, number, h)[1], self.lists_rho(points, number, h)[0], "purple")
        ]

        fig, axes = plt.subplots(1, 5, figsize=(20,4))  # 1 ligne, 5 colonnes

        fig.patch.set_facecolor('black')
        
        for ax, (name, values, S_vals, color) in zip(axes, greeks_list):
            ax.set_facecolor('black')
            ax.plot(S_vals, values, color=color, linewidth=2)
            
            # Style axes
            for side in ["bottom", "top", "left", "right"]:
                ax.spines[side].set_color("orange")
            ax.tick_params(axis="x", colors="orange")
            ax.tick_params(axis="y", colors="orange")
            
            ax.set_title(name, color="orange")
            ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        plt.tight_layout()
        return fig




