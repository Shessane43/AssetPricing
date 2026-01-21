import matplotlib.pyplot as plt
import numpy as np

class Swap:
    def __init__(self, nominal, fixed_rate, zero_coupon_rate, maturity, frequency=1):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.zero_coupon_rate = zero_coupon_rate
        self.maturity = maturity
        self.frequency = frequency
        self.n_periods = int(maturity * frequency)
    
    def discount_factor(self, t):
        r = self.zero_coupon_rate
        f = self.frequency
        return 1 / (1 + r/f) ** t
    
    def price_from(self, start_period):
        """
        Price the swap starting from a given period (0 = today).
        Returns the NPV of remaining cash flows.
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