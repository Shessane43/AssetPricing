import matplotlib.pyplot as plt
import numpy as np

class Swap:
    def __init__(self, nominal, fixed_rate, zero_coupon_rate, maturity, frequency=1):
        """
        Parameters
        ----------
        nominal : float
            Swap notional.
        fixed_rate : float
            Fixed rate of the swap.
        zero_coupon_rate : float
            Flat zero-coupon interest rate used for discounting.
        maturity : float
            Swap maturity in years.
        frequency : int, optional
            Number of payment periods per year (default is 1).
        """
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.zero_coupon_rate = zero_coupon_rate
        self.maturity = maturity
        self.frequency = frequency
        self.n_periods = int(maturity * frequency)
    
    def discount_factor(self, t):
        """
        Computes the discount factor for a given time.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor at time t.
        """
        r = self.zero_coupon_rate
        f = self.frequency
        return 1 / (1 + r / f) ** t
    
    def price_from(self, start_period):
        """
        Computes the swap value starting from a given period.

        Parameters
        ----------
        start_period : int
            Starting period index (0 = today).

        Returns
        -------
        float
            Net present value (NPV) of the remaining swap cash flows.
            Positive value corresponds to a receive-fixed position.
        """
        r = self.zero_coupon_rate
        f = self.frequency
        C = self.fixed_rate
        nominal = self.nominal
        
        remaining_periods = self.n_periods - start_period
        
        if remaining_periods <= 0:
            return 0.0
        
        fixed_leg = 0.0
        for i in range(1, remaining_periods + 1):
            t = i / f
            fixed_leg += C * nominal / f * self.discount_factor(t)

        floating_leg = nominal - nominal * self.discount_factor(remaining_periods / f)
        
        return fixed_leg - floating_leg

    def plot_value_evolution(self):
        """
        Plots the remaining net present value of the swap over time.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure object.
        """
        times = np.arange(0, self.n_periods + 1) / self.frequency
        values = [self.price_from(p) for p in range(self.n_periods + 1)]
        
        fig, ax = plt.subplots(figsize=(10,5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.plot(
            times,
            values,
            marker="o",
            color="orange",
            lw=2,
            label="Remaining value"
        )

        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("orange")

        ax.tick_params(colors="orange")
        ax.set_xlabel("Time (years)", color="orange")
        ax.set_ylabel("Remaining net value of the swap", color="orange")
        ax.set_title("Evolution of the swap value over time", color="orange")
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        legend = ax.legend(facecolor="black", edgecolor="orange")
        for text in legend.get_texts():
            text.set_color("orange")

        return fig
    
    def duration(self, dr=0.0001):
        """
        Approximates the swap duration using finite differences.

        Parameters
        ----------
        dr : float, optional
            Interest rate shift used for finite differences (default is 1bp).

        Returns
        -------
        float
            Swap duration.
        """
        pv_plus = Swap(
            self.nominal, self.fixed_rate, self.zero_coupon_rate + dr,
            self.maturity, self.frequency
        ).price_from(0)

        pv_minus = Swap(
            self.nominal, self.fixed_rate, self.zero_coupon_rate - dr,
            self.maturity, self.frequency
        ).price_from(0)

        pv = self.price_from(0)

        return (pv_minus - pv_plus) / (2 * dr * pv)

    def convexity(self, dr=0.0001):
        """
        Approximates the swap convexity using finite differences.

        Parameters
        ----------
        dr : float, optional
            Interest rate shift used for finite differences (default is 1bp).

        Returns
        -------
        float
            Swap convexity.
        """
        pv_plus = Swap(
            self.nominal, self.fixed_rate, self.zero_coupon_rate + dr,
            self.maturity, self.frequency
        ).price_from(0)

        pv_minus = Swap(
            self.nominal, self.fixed_rate, self.zero_coupon_rate - dr,
            self.maturity, self.frequency
        ).price_from(0)

        pv = self.price_from(0)

        return (pv_plus + pv_minus - 2 * pv) / (pv * dr ** 2)

    def pv01(self):
        """
        Computes the PV01 / DV01 of the swap.

        Returns
        -------
        float
            Change in swap value for a +1bp increase in interest rates.
            Positive for a receive-fixed position.
        """
        pv_shift = Swap(
            self.nominal, self.fixed_rate, self.zero_coupon_rate + 0.0001,
            self.maturity, self.frequency
        ).price_from(0)

        return self.price_from(0) - pv_shift
