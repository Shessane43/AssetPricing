import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions.bond_function import Bond
from functions.swap_function import Swap

# ---------------- Hull-White Class ----------------
class HullWhite:
    """
    Hull-White short-rate model for interest rate derivatives.
    Provides analytic bond pricing and Monte Carlo simulation.
    """

    def __init__(self, r0=0.03, alpha=0.1, sigma=0.01):
        """
        Initialize Hull-White parameters.

        Parameters
        ----------
        r0 : float
            Initial short rate.
        alpha : float
            Mean-reversion speed.
        sigma : float
            Volatility of the short rate.
        """
        self.r0 = r0
        self.alpha = alpha
        self.sigma = sigma

    # ---------------- Analytic ZCB price ----------------
    def zero_coupon_bond(self, T):
        """
        Analytic zero-coupon bond price P(0,T) under Hull-White.

        Parameters
        ----------
        T : float
            Time to maturity.

        Returns
        -------
        float
            Price of the zero-coupon bond.
        """
        B = (1 - np.exp(-self.alpha * T)) / self.alpha
        A = np.exp(
            (B - T) * (self.alpha**2 * self.r0 - 0.5 * self.sigma**2) / self.alpha**2
            - self.sigma**2 * B**2 / (4 * self.alpha)
        )
        P = A * np.exp(-B * self.r0)
        return P

    # ---------------- Analytic coupon bond price ----------------
    def coupon_bond(self, nominal, coupon_rate, t0, maturity, frequency, r=None):
        """
        Price a coupon bond using analytic zero-coupon bonds.

        Parameters
        ----------
        nominal : float
            Face value of the bond.
        coupon_rate : float
            Annual coupon rate.
        t0 : float
            Current time.
        maturity : float
            Maturity of the bond.
        frequency : int
            Coupons per year.
        r : float, optional
            Short rate override. Defaults to r0.

        Returns
        -------
        float
            Price of the coupon bond.
        """
        if r is None:
            r = self.r0
        times = np.arange(1 / frequency, maturity + 1 / frequency, 1 / frequency)
        price = 0
        for tau in times:
            price += coupon_rate * nominal / frequency * self.zero_coupon_bond(tau - t0)
        price += nominal * self.zero_coupon_bond(maturity - t0)
        return price

    # ---------------- Analytic swap price ----------------
    def swap(self, nominal, fixed_rate, t0, maturity, frequency, r=None):
        """
        Price a fixed-for-floating interest rate swap analytically.

        Parameters
        ----------
        nominal : float
            Notional amount.
        fixed_rate : float
            Fixed rate of the swap.
        t0 : float
            Current time.
        maturity : float
            Swap maturity.
        frequency : int
            Payment frequency per year.
        r : float, optional
            Short rate override. Defaults to r0.

        Returns
        -------
        float
            Net present value of the swap (fixed - floating).
        """
        if r is None:
            r = self.r0
        times = np.arange(1 / frequency, maturity + 1 / frequency, 1 / frequency)
        fixed_leg = sum(fixed_rate * nominal / frequency * self.zero_coupon_bond(t - t0) for t in times)
        floating_leg = nominal - nominal * self.zero_coupon_bond(maturity - t0)
        return fixed_leg - floating_leg

    # ---------------- Monte Carlo short-rate simulation ----------------
    def simulate_short_rate(self, T, N, M, r0=None, seed=None):
        """
        Simulate short-rate paths under Hull-White using Euler discretization.

        Parameters
        ----------
        T : float
            Time horizon.
        N : int
            Number of Monte Carlo paths.
        M : int
            Number of time steps.
        r0 : float, optional
            Initial rate. Defaults to self.r0.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of simulated short-rate paths with shape (N, M+1).
        """
        if r0 is None:
            r0 = self.r0
        if seed is not None:
            np.random.seed(seed)
        dt = T / M
        r = np.zeros((N, M + 1))
        r[:, 0] = r0
        for i in range(1, M + 1):
            dr = self.alpha * (r0 - r[:, i - 1]) * dt + self.sigma * np.sqrt(dt) * np.random.randn(N)
            r[:, i] = r[:, i - 1] + dr
        return r

    # ---------------- Monte Carlo coupon bond ----------------
    def mc_coupon_bond(self, nominal, coupon_rate, T, frequency=1, N=1000, M=100, r0=None, seed=None):
        """
        Monte Carlo estimation of coupon bond price.

        Returns average price and all path prices.

        Parameters
        ----------
        nominal : float
            Face value.
        coupon_rate : float
            Annual coupon rate.
        T : float
            Maturity.
        frequency : int
            Coupon frequency.
        N : int
            Number of MC paths.
        M : int
            Number of time steps.
        r0 : float, optional
            Initial short rate.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple
            (mean price, array of all path prices)
        """
        if r0 is None:
            r0 = self.r0
        dt = T / M
        r_paths = self.simulate_short_rate(T, N, M, r0, seed)
        pv_paths = np.zeros(N)
        times = np.arange(1 / frequency, T + 1 / frequency, 1 / frequency)
        for i in range(N):
            discount_factors = np.exp(-np.cumsum(r_paths[i, :]) * dt)
            pv = 0
            for tau in times:
                idx = min(int(round(tau / dt)), M)
                pv += coupon_rate * nominal / frequency * discount_factors[idx]
            pv += nominal * discount_factors[M]
            pv_paths[i] = pv
        return pv_paths.mean(), pv_paths

    # ---------------- Monte Carlo swap ----------------
    def mc_swap(self, nominal, fixed_rate, T, frequency=1, N=1000, M=100, r0=None, seed=None):
        """
        Monte Carlo estimation of swap value.

        Returns average swap value and all path values.

        Parameters
        ----------
        nominal : float
            Notional amount.
        fixed_rate : float
            Fixed leg rate.
        T : float
            Maturity.
        frequency : int
            Payment frequency per year.
        N : int
            Number of MC paths.
        M : int
            Number of time steps.
        r0 : float, optional
            Initial short rate.
        seed : int, optional
            Random seed.

        Returns
        -------
        tuple
            (mean swap value, array of all path swap values)
        """
        if r0 is None:
            r0 = self.r0
        dt = T / M
        r_paths = self.simulate_short_rate(T, N, M, r0, seed)
        pv_paths = np.zeros(N)
        times = np.arange(1 / frequency, T + 1 / frequency, 1 / frequency)
        for i in range(N):
            discount_factors = np.exp(-np.cumsum(r_paths[i, :]) * dt)
            pv = 0
            # Fixed leg
            for tau in times:
                idx = min(int(round(tau / dt)), M)
                pv += fixed_rate * nominal / frequency * discount_factors[idx]
            # Floating leg
            pv -= nominal - nominal * discount_factors[M]
            pv_paths[i] = pv
        return pv_paths.mean(), pv_paths
