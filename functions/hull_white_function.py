import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions.bond_function import Bond
from functions.swap_function import Swap

# ---------------- Hull-White Class ----------------
class HullWhite:
    def __init__(self, r0=0.03, alpha=0.1, sigma=0.01):
        self.r0 = r0
        self.alpha = alpha
        self.sigma = sigma

    # Analytic ZCB price (simplified)
    def zero_coupon_bond(self, T):
        B = (1 - np.exp(-self.alpha * T)) / self.alpha
        A = np.exp((B - T) * (self.alpha**2 * self.r0 - 0.5 * self.sigma**2)/self.alpha**2 - self.sigma**2 * B**2 / (4*self.alpha))
        P = A * np.exp(-B * self.r0)
        return P

    # Analytic coupon bond price
    def coupon_bond(self, nominal, coupon_rate, t0, maturity, frequency, r=None):
        if r is None:
            r = self.r0
        times = np.arange(1/frequency, maturity + 1/frequency, 1/frequency)
        price = 0
        for tau in times:
            price += coupon_rate * nominal / frequency * self.zero_coupon_bond(tau - t0)
        price += nominal * self.zero_coupon_bond(maturity - t0)
        return price

    # Analytic swap price
    def swap(self, nominal, fixed_rate, t0, maturity, frequency, r=None):
        if r is None:
            r = self.r0
        times = np.arange(1/frequency, maturity + 1/frequency, 1/frequency)
        fixed_leg = sum(fixed_rate * nominal / frequency * self.zero_coupon_bond(t - t0) for t in times)
        floating_leg = nominal - nominal * self.zero_coupon_bond(maturity - t0)
        return fixed_leg - floating_leg

    # Monte Carlo short-rate simulation with optional seed
    def simulate_short_rate(self, T, N, M, r0=None, seed=None):
        if r0 is None:
            r0 = self.r0
        if seed is not None:
            np.random.seed(seed)
        dt = T / M
        r = np.zeros((N, M+1))
        r[:,0] = r0
        for i in range(1, M+1):
            # Mean reversion towards r0 (important for sensitives)
            dr = self.alpha * (r0 - r[:,i-1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn(N)
            r[:,i] = r[:,i-1] + dr
        return r

    # Monte Carlo coupon bond
    def mc_coupon_bond(self, nominal, coupon_rate, T, frequency=1, N=1000, M=100, r0=None, seed=None):
        if r0 is None:
            r0 = self.r0
        dt = T / M
        r_paths = self.simulate_short_rate(T, N, M, r0, seed)
        pv_paths = np.zeros_like(r_paths)
        for i in range(N):
            discount_factors = np.exp(-np.cumsum(r_paths[i,:])*dt)
            pv = np.zeros(M+1)
            for tau in np.arange(1/frequency, T + 1/frequency, 1/frequency):
                idx = int(tau/dt)
                if idx > M:
                    idx = M
                pv[:idx+1] += coupon_rate * nominal / frequency * discount_factors[:idx+1]
            pv[-1] += nominal * discount_factors[-1]
            pv_paths[i,:] = pv
        return pv_paths.mean(axis=0), pv_paths

    # Monte Carlo swap
    def mc_swap(self, nominal, fixed_rate, T, frequency=1, N=1000, M=100, r0=None, seed=None):
        if r0 is None:
            r0 = self.r0
        dt = T / M
        r_paths = self.simulate_short_rate(T, N, M, r0, seed)
        pv_paths = np.zeros_like(r_paths)
        for i in range(N):
            discount_factors = np.exp(-np.cumsum(r_paths[i,:])*dt)
            pv = np.zeros(M+1)
            for tau in np.arange(1/frequency, T + 1/frequency, 1/frequency):
                idx = int(tau/dt)
                if idx > M:
                    idx = M
                pv[:idx+1] += fixed_rate * nominal / frequency * discount_factors[:idx+1]
            pv -= nominal - nominal * discount_factors  # approximate floating leg
            pv_paths[i,:] = pv
        return pv_paths.mean(axis=0), pv_paths