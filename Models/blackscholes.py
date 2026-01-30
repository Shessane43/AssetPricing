import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Models.models import Model
import streamlit as st


class BlackScholes(Model):
    def __init__(self, S, K, r, sigma, T, q, option_type, buy_sell, option_class):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q 
        self.option_type = option_type.lower()
        self.buy_sell = buy_sell.lower()
        self.option_class = option_class.lower()  
     
    def price(self):
        if self.option_class != "vanilla":
            raise Exception("Black-Scholes applies only to European vanilla options.")

             

        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == "call":
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return -price if self.buy_sell == "sell" else price

    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        if self.option_type == "call":
            delta = np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            delta = np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
        return -delta if self.buy_sell == "short" else delta
    
    def lists_delta(self, points = 0.3, number = 100, h=1e-4):
        delta_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            modele = BlackScholes(S, self.K, self.r, self.sigma, self.T, self.q, self.option_type, self.buy_sell, self.option_class)
            greeks = modele.delta()
            delta_list.append(greeks)
        return S_values, delta_list

    def gamma(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        gamma = (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S * self.sigma * np.sqrt(self.T))
        return -gamma if self.buy_sell == "short" else gamma
    
    def lists_gamma(self, points = 0.3, number= 100, h=1e-4):
        gamma_list = []
        S_values = np.linspace(self.S - points*self.S, self.S + points*self.S, number)
        for S in S_values:
            modele = BlackScholes(S, self.K, self.r, self.sigma, self.T,self.q, self.option_type, self.buy_sell, self.option_class)
            greeks = modele.gamma()
            gamma_list.append(greeks)
        return S_values, gamma_list
    
    def vega(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
        return -vega if self.buy_sell == "short" else vega
    
    def lists_vega(self, points = 0.3, number= 100, h=1e-4):
        vega_list = []
        sigma_values = np.linspace(self.sigma - points*self.sigma, self.sigma + points*self.sigma, number)
        for sigma in sigma_values:
            modele = BlackScholes(self.S, self.K, self.r, sigma, self.T,self.q, self.option_type, self.buy_sell, self.option_class)
            greeks = modele.vega()
            vega_list.append(greeks)
        return sigma_values, vega_list
    
    def theta(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == "call":
            theta = (- (self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
                     + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1))
        else:
            theta = (- (self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                     - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        return -theta if self.buy_sell == "short" else theta
    
    def lists_theta(self, points = 0.3, number= 100, h=1e-4):
        theta_list = []
        T_values = np.linspace(self.T - points*self.T, self.T + points*self.T, number)
        for T in T_values:
            modele = BlackScholes(self.S, self.K, self.r, self.sigma, T,self.q,self.option_type, self.buy_sell, self.option_class)
            greeks = modele.theta()
            theta_list.append(greeks)
        return T_values, theta_list
    
    def rho(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return -rho if self.buy_sell == "short" else rho
    
    def lists_rho(self, points = 0.3, number= 100, h=1e-4):
        rho_list = []
        r_values = np.linspace(self.r - points*self.r, self.r + points*self.r, number)
        for r in r_values:
            modele = BlackScholes(self.S, self.K, r, self.sigma, self.T,self.q ,self.option_type, self.buy_sell, self.option_class)
            greeks = modele.rho()
            rho_list.append(greeks)
        return r_values, rho_list
    
    def plot_all_greeks(self, points=0.3, number=100, h=1e-4):

        S_delta, delta_values = self.lists_delta(points, number, h)
        S_gamma, gamma_values = self.lists_gamma(points, number, h)
        sigma_vega, vega_values = self.lists_vega(points, number, h)
        T_theta, theta_values = self.lists_theta(points, number, h)
        r_rho, rho_values = self.lists_rho(points, number, h)

        plt.figure(figsize=(12, 8))

        plt.plot(S_delta, delta_values, label="Delta")
        plt.plot(S_gamma, gamma_values, label="Gamma")
        plt.plot(sigma_vega, vega_values, label="Vega")
        plt.plot(T_theta, theta_values, label="Theta")
        plt.plot(r_rho, rho_values, label="Rho")

        plt.title("Greeks")
        plt.ylabel("Greeks")
        plt.grid(True)
        plt.legend()
        plt.show()
