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
        
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(times, values, marker='o')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Remaining net value of the swap')
        ax.set_title('Evolution of the swap value over time')
        ax.grid(True)

        return fig