import numpy as np
import matplotlib.pyplot as plt

class Bond:
    def __init__(self, nominal, coupon_rate, rate, maturity, frequency=1):
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.rate = rate
        self.maturity = maturity
        self.frequency = frequency
    
    def price(self):
        price = 0.0
        coupon = self.nominal * self.coupon_rate / self.frequency
        n_periods = int(self.maturity * self.frequency)
        
        for i in range(1, n_periods + 1):
            price += coupon / (1 + self.rate / self.frequency) ** i
        
        price += self.nominal / (1 + self.rate / self.frequency) ** n_periods
        return price
    
    def cash_flows(self):
        coupon = self.nominal * self.coupon_rate / self.frequency
        n_periods = int(self.maturity * self.frequency)
        times = np.arange(1/self.frequency, self.maturity + 1/self.frequency, 1/self.frequency)
        flows = [coupon] * n_periods
        flows[-1] += self.nominal
        return times, flows
    
    def plot_value_evolution(self):
        times, flows = self.cash_flows()
        remaining_value = []
        
        for t in times:
            idx = times >= t
            value = sum([f / (1 + self.rate / self.frequency) ** ((time - t) * self.frequency)
                         for f, time in zip(np.array(flows)[idx], times[idx])])
            remaining_value.append(value)
        
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(times, remaining_value, marker='o')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Present value of remaining cash flows')
        ax.set_title('Evolution of the bond value')
        ax.grid(True)

        return fig
