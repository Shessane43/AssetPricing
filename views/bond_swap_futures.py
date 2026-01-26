import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions.bond_function import Bond
from functions.swap_function import Swap
from functions.hull_white_function import HullWhite  

def app():
    # -------------------- Model selection --------------------
    pricing_model = st.selectbox(
        "Select pricing model",
        ["Classic (Closed-form)", "Hull-White"]
    )

    # Hull-White parameters (only if model selected)
    if pricing_model == "Hull-White":
        st.subheader("Hull-White Model Parameters")
        r0 = st.number_input("Initial short rate r0 (%)", value=3.0, step=0.1) / 100
        alpha = st.number_input("Mean reversion α", value=0.1, step=0.01)
        sigma = st.number_input("Volatility σ", value=0.01, step=0.001)
        hw_model = HullWhite(r0=r0, alpha=alpha, sigma=sigma)

        show_hw_info = st.checkbox("Show Hull-White model explanation")
        if show_hw_info:
            st.markdown("""
            ### Hull-White Model

            The **Hull-White model** is a short-rate stochastic model defined by the SDE:

            $$
            dr_t = \\alpha (\\theta_t - r_t) dt + \\sigma dW_t
            $$

            Where:
            - $r_t$ is the short rate at time $t$  
            - $\\alpha$ is the mean reversion speed  
            - $\\theta_t$ is the time-dependent long-term mean  
            - $\\sigma$ is the volatility of the short rate  
            - $dW_t$ is a standard Wiener process  

            This model allows:
            - **Mean reversion** of rates
            - **Stochastic volatility**
            - **Analytical formulas** for Zero-Coupon Bonds and Swaps
            - Monte Carlo simulations for complex derivatives (Caps, Floors, Swaptions)
            """)

    # -------------------- Instrument selection --------------------
    instrument_type = st.radio("Which instrument would you like to price?", ["Bond", "Swap"])

    # -------------------- Bond --------------------
    if instrument_type == "Bond":
        st.subheader("Bond Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, step=0.1) / 100
        rate = st.number_input("Interest Rate (%) (Not used in Hull-White)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5, step=1)
        frequency = st.number_input("Coupon Payments per Year", value=1, step=1)

        if st.button("Calculate Bond Price"):
            if pricing_model == "Classic (Closed-form)":
                bond = Bond(nominal, coupon_rate, rate, maturity, frequency)
                price = bond.price()
                duration = bond.duration()
                convexity = bond.convexity()
                pv01 = bond.pv01()

            if pricing_model == "Hull-White":
                # Monte Carlo price at t=0
                mean_pv, pv_paths = hw_model.mc_coupon_bond(nominal, coupon_rate, maturity, frequency, N=5000, M=100)
                price = mean_pv[0]  # t=0 price

                dr = 0.0001
                # Sensitivities to r0
                price_up = hw_model.mc_coupon_bond(nominal, coupon_rate, maturity, frequency, N=5000, M=100, r0=r0+dr)[0][0]
                price_down = hw_model.mc_coupon_bond(nominal, coupon_rate, maturity, frequency, N=5000, M=100, r0=r0-dr)[0][0]

                duration = (price_down - price_up) / (2 * dr * price)
                convexity = (price_up + price_down - 2 * price) / (price * dr**2)
                pv01 = hw_model.mc_coupon_bond(nominal, coupon_rate, maturity, frequency, N=5000, M=100, r0=r0+0.0001)[0][0] - price

            # Display Key Metrics
            st.subheader("Key Bond Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bond Price (€)", f"{price:.2f}")
            col2.metric("Duration (yrs)", f"{duration:.4f}")
            col3.metric("Convexity", f"{convexity:.4f}")
            col4.metric("PV01 (€)", f"{pv01:.2f}")

            # Graph Monte Carlo PV
            st.subheader("Remaining Value Evolution")
            if pricing_model == "Classic (Closed-form)":
                fig = bond.plot_value_evolution()
                st.pyplot(fig)

            if pricing_model == "Hull-White":
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10,5))
                for path in pv_paths[:100]:
                    ax.plot(np.linspace(0, maturity, len(path)), path, color='orange', alpha=0.2)
                ax.plot(np.linspace(0, maturity, len(mean_pv)), mean_pv, color='red', lw=2, label="Mean PV")
                ax.set_title("Monte Carlo Simulation of Bond Value (Hull-White)", color='orange')
                ax.set_xlabel("Time (years)", color='orange')
                ax.set_ylabel("Bond PV (€)", color='orange')
                ax.tick_params(colors='orange')
                ax.spines['bottom'].set_color('orange')
                ax.spines['top'].set_color('orange')
                ax.spines['right'].set_color('orange')
                ax.spines['left'].set_color('orange')
                ax.legend(facecolor='black', edgecolor='orange', labelcolor='orange')
                st.pyplot(fig)

    # -------------------- Swap --------------------
    elif instrument_type == "Swap":
        st.subheader("Swap Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        fixed_rate = st.number_input("Fixed Rate (%)", value=5.0, step=0.1) / 100
        zero_coupon_rate = st.number_input("Zero-Coupon Rate (%) (Not used in Hull-White)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5, step=1)
        frequency = st.number_input("Payments per Year", value=1, step=1)

        if st.button("Calculate Swap Price"):
            if pricing_model == "Classic (Closed-form)":
                swap = Swap(nominal, fixed_rate, zero_coupon_rate, maturity, frequency)
                value = swap.price_from(0)
                duration = swap.duration()
                convexity = swap.convexity()
                pv01 = swap.pv01()

            if pricing_model == "Hull-White":
                # Monte Carlo price at t=0
                mean_pv, pv_paths = hw_model.mc_swap(nominal, fixed_rate, maturity, frequency, N=5000, M=100)
                value = mean_pv[0]  # t=0 value

                dr = 0.0001
                # Sensitivities to r0
                price_up = hw_model.mc_swap(nominal, fixed_rate, maturity, frequency, N=5000, M=100, r0=r0+dr)[0][0]
                price_down = hw_model.mc_swap(nominal, fixed_rate, maturity, frequency, N=5000, M=100, r0=r0-dr)[0][0]

                duration = (price_down - price_up) / (2 * dr * value)
                convexity = (price_up + price_down - 2 * value) / (value * dr**2)
                pv01 = hw_model.mc_swap(nominal, fixed_rate, maturity, frequency, N=5000, M=100, r0=r0+0.0001)[0][0] - value

            # Display Key Metrics
            st.subheader("Key Swap Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Net Swap Value (€)", f"{value:.2f}")
            col2.metric("Duration (yrs)", f"{duration:.4f}")
            col3.metric("Convexity", f"{convexity:.4f}")
            col4.metric("PV01 (€)", f"{pv01:.2f}")

            # Graph Monte Carlo PV
            st.subheader("Remaining Value Evolution")
            if pricing_model == "Classic (Closed-form)":
                fig = swap.plot_value_evolution()
                st.pyplot(fig)

            if pricing_model == "Hull-White":
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10,5))
                for path in pv_paths[:100]:
                    ax.plot(np.linspace(0, maturity, len(path)), path, color='orange', alpha=0.2)
                ax.plot(np.linspace(0, maturity, len(mean_pv)), mean_pv, color='red', lw=2, label="Mean PV")
                ax.set_title("Monte Carlo Simulation of Swap Value (Hull-White)", color='orange')
                ax.set_xlabel("Time (years)", color='orange')
                ax.set_ylabel("Swap PV (€)", color='orange')
                ax.tick_params(colors='orange')
                ax.spines['bottom'].set_color('orange')
                ax.spines['top'].set_color('orange')
                ax.spines['right'].set_color('orange')
                ax.spines['left'].set_color('orange')
                ax.legend(facecolor='black', edgecolor='orange', labelcolor='orange')
                st.pyplot(fig)
