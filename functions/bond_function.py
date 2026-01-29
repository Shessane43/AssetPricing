import numpy as np
import matplotlib.pyplot as plt

class Bond:
    def __init__(self, nominal, coupon_rate, rate, maturity, frequency=1):
        """
        Parameters
        ----------
        nominal : float
            Bond face value.
        coupon_rate : float
            Annual coupon rate.
        rate : float
            Annual yield to maturity.
        maturity : float
            Bond maturity in years.
        frequency : int, optional
            Number of coupon payments per year (default is 1).
        """
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.rate = rate
        self.maturity = maturity
        self.frequency = frequency
    
    def price(self):
        """
        Computes the bond price.

        Returns
        -------
        float
            Present value of all future cash flows.
        """
        price = 0.0
        coupon = self.nominal * self.coupon_rate / self.frequency
        n_periods = int(self.maturity * self.frequency)
        
        for i in range(1, n_periods + 1):
            price += coupon / (1 + self.rate / self.frequency) ** i
        
        price += self.nominal / (1 + self.rate / self.frequency) ** n_periods
        return price
    
    def cash_flows(self):
        """
        Returns the bond cash flow schedule.

        Returns
        -------
        times : numpy.ndarray
            Payment times in years.
        flows : list of float
            Cash flow amounts at each payment date.
        """
        coupon = self.nominal * self.coupon_rate / self.frequency
        n_periods = int(self.maturity * self.frequency)
        times = np.arange(1/self.frequency, self.maturity + 1/self.frequency, 1/self.frequency)
        flows = [coupon] * n_periods
        flows[-1] += self.nominal
        return times, flows
    
    def plot_value_evolution(self):
        """
        Plots the remaining present value of the bond over time.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure object.
        """
        times, flows = self.cash_flows()
        remaining_value = []
        
        for t in times:
            idx = times >= t
            value = sum([
                f / (1 + self.rate / self.frequency) ** ((time - t) * self.frequency)
                for f, time in zip(np.array(flows)[idx], times[idx])
            ])
            remaining_value.append(value)

        fig, ax = plt.subplots(figsize=(10,5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.plot(
            times,
            remaining_value,
            marker="o",
            color="orange",
            lw=2,
            label="Remaining value"
        )

        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color("orange")

        ax.tick_params(colors="orange")
        ax.set_xlabel("Time (years)", color="orange")
        ax.set_ylabel("Present value of remaining cash flows", color="orange")
        ax.set_title("Evolution of the bond value", color="orange")
        ax.grid(True, linestyle="--", color="orange", alpha=0.3)

        legend = ax.legend(facecolor="black", edgecolor="orange")
        for text in legend.get_texts():
            text.set_color("orange")

        return fig

    def duration(self, dr=0.0001):
        """
        Approximates the Macaulay duration using finite differences.

        Parameters
        ----------
        dr : float, optional
            Yield shift used for finite differences (default is 1bp).

        Returns
        -------
        float
            Macaulay duration in years.
        """
        pv_plus = Bond(self.nominal, self.coupon_rate, self.rate + dr, self.maturity, self.frequency).price()
        pv_minus = Bond(self.nominal, self.coupon_rate, self.rate - dr, self.maturity, self.frequency).price()
        pv = self.price()
        return (pv_minus - pv_plus) / (2 * dr * pv)

    def convexity(self, dr=0.0001):
        """
        Approximates the bond convexity using finite differences.

        Parameters
        ----------
        dr : float, optional
            Yield shift used for finite differences (default is 1bp).

        Returns
        -------
        float
            Bond convexity.
        """
        pv_plus = Bond(self.nominal, self.coupon_rate, self.rate + dr, self.maturity, self.frequency).price()
        pv_minus = Bond(self.nominal, self.coupon_rate, self.rate - dr, self.maturity, self.frequency).price()
        pv = self.price()
        return (pv_plus + pv_minus - 2 * pv) / (pv * dr ** 2)

    def pv01(self):
        """
        Computes the PV01 / DV01 of the bond.

        Returns
        -------
        float
            Price change for a +1bp increase in yield.
            Positive for a long bond position.
        """
        pv_shift = Bond(self.nominal, self.coupon_rate, self.rate + 0.0001, self.maturity, self.frequency).price()
        return self.price() - pv_shift
