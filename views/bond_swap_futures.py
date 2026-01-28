import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions.bond_function import Bond
from functions.swap_function import Swap
from functions.capfloor_function import CapFloor
from functions.fra_future_function import FRAFuture
from functions.hull_white_function import HullWhite

def app():
    # -------------------- Model selection --------------------
    pricing_model = st.selectbox(
        "Select pricing model",
        ["Classic (Closed-form)", "Hull-White"]
    )

    # Hull-White parameters
    if pricing_model == "Hull-White":
        st.subheader("Hull-White Model Parameters")
        r0 = st.number_input("Initial short rate r0 (%)", value=3.0, step=0.1) / 100
        alpha = st.number_input("Mean reversion α", value=0.1, step=0.01)
        sigma = st.number_input("Volatility σ", value=0.01, step=0.001)
        hw_model = HullWhite(r0=r0, alpha=alpha, sigma=sigma)

        if st.checkbox("Show Hull-White explanation"):
            st.markdown("""
            ### Hull-White Model
            $$
            dr_t = \\alpha (\\theta_t - r_t) dt + \\sigma dW_t
            $$
            - Short rate: $r_t$  
            - Mean reversion: $\\alpha$  
            - Volatility: $\\sigma$  
            - Monte Carlo simulation possible for bonds, swaps, caps/floors, FRA/Future
            """)

    # -------------------- Instrument selection --------------------
    instrument_category = st.radio(
        "Select instrument category",
        ["Bond", "Swap", "Cap / Floor", "FRA / Future"]
    )

    # -------------------- Bond --------------------
    if instrument_category == "Bond":
        st.subheader("Bond Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, step=0.1) / 100
        rate = st.number_input("Interest Rate (%) (Not used in HW)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5.0, step=1.0)
        frequency = st.number_input("Coupon Payments per Year", value=3.0, step=1.0)

        if st.button("Calculate Bond Price"):
            if pricing_model == "Classic (Closed-form)":
                bond = Bond(nominal, coupon_rate, rate, maturity, frequency)
                price = bond.price()
                duration = bond.duration()
                convexity = bond.convexity()
                pv01 = bond.pv01()

                st.subheader("Key Bond Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price (€)", f"{price:.2f}")
                col2.metric("Duration (yrs)", f"{duration:.4f}")
                col3.metric("Convexity", f"{convexity:.4f}")
                col4.metric("PV01 (€)", f"{pv01:.2f}")
            else:
                mean_pv, path_pv= hw_model.mc_coupon_bond(nominal, coupon_rate, maturity, frequency, N=5000, M=100)
                price = mean_pv
                st.metric("Price (€)", f"{price:.2f}")
                st.markdown("**Note:** Duration, Convexity, and PV01 are not available under Hull-White Monte Carlo pricing.")



           

    # -------------------- Swap --------------------
    elif instrument_category == "Swap":
        st.subheader("Swap Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        fixed_rate = st.number_input("Fixed Rate (%)", value=5.0, step=0.1) / 100
        zero_coupon_rate = st.number_input("Zero-Coupon Rate (%)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5.0, step=1.0)
        frequency = st.number_input("Payments per Year", value=1.0, step=1.0)

        if st.button("Calculate Swap Price"):
            if pricing_model == "Classic (Closed-form)":
                swap = Swap(nominal, fixed_rate, zero_coupon_rate, maturity, frequency)
                value = swap.price_from(0)
                duration = swap.duration()
                convexity = swap.convexity()
                pv01 = swap.pv01()
                st.subheader("Key Swap Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Net Swap Value (€)", f"{value:.2f}")
                col2.metric("Duration (yrs)", f"{duration:.4f}")
                col3.metric("Convexity", f"{convexity:.4f}")
                col4.metric("PV01 (€)", f"{pv01:.2f}")
            else:
                mean_pv , path_pv= hw_model.mc_swap(nominal, fixed_rate, maturity, frequency, N=5000, M=100)
                price = mean_pv
                st.metric("Price (€)", f"{price:.2f}")
                st.markdown("**Note:** Duration, Convexity, and PV01 are not available under Hull-White Monte Carlo pricing.")




    # -------------------- Cap / Floor --------------------
    elif instrument_category == "Cap / Floor":
        if pricing_model == "Hull-White":
            st.warning("Hull-White pricing is not available for Cap/Floor. Please select Classic (Closed-form).")
            st.stop()
        option_type = st.radio("Select type", ["Cap", "Floor"])
        st.subheader(f"{option_type} Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        strike = st.number_input("Strike Rate (%)", value=5.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5.0, step=1.0)
        frequency = st.number_input("Payments per Year", value=1.0, step=1.0)
        vol = st.number_input("Volatility σ (%)", value=20.0, step=0.1) / 100

        if st.button("Calculate Cap/Floor Price"):
            capfloor = CapFloor(nominal, strike, maturity, frequency, vol, option_type)
            price = capfloor.price_classic()

            dr = 1e-4
            dvol = 1e-4
            cap_up = CapFloor(nominal, strike, maturity, frequency, vol + dr, option_type)
            cap_down = CapFloor(nominal, strike, maturity, frequency, vol - dr, option_type)
            vega = (cap_up.price_classic() - cap_down.price_classic()) / (2 * dr)
            delta = (capfloor.price_classic(strike + dr) - capfloor.price_classic(strike - dr)) / (2 * dr)

            st.subheader(f"{option_type} Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Price (€)", f"{price:.2f}")
            col2.metric("Delta (€)", f"{delta:.4f}")
            col3.metric("Vega (€)", f"{vega:.4f}")

    # -------------------- FRA / Future --------------------
    elif instrument_category == "FRA / Future":
        if pricing_model == "Hull-White":
            st.warning("Hull-White pricing is not available for FRA/Future. Please select Classic (Closed-form).")
            st.stop()
        product_type = st.radio("Select product", ["FRA", "Future"])
        st.subheader(f"{product_type} Parameters")

        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        forward_rate = st.number_input("Forward Rate (%)", value=3.0, step=0.01) / 100
        strike = st.number_input("Strike Rate (%)", value=1.0, step=0.01) / 100

        # Start/end uniquement pour FRA
        if product_type == "FRA":
            start = st.number_input("Start (years)", value=1.0, step=0.25)
            end = st.number_input("End (years)", value=1.25, step=0.25)

        if st.button(f"Calculate {product_type} Price"):
            if product_type == "FRA":
                fra_future = FRAFuture(
                    nominal=nominal,
                    forward_rate=forward_rate,
                    strike=strike,
                    product_type=product_type,
                    start=start,
                    end=end
                )
            else:  # Future n’a pas de start/end et pas de vol
                fra_future = FRAFuture(
                    nominal=nominal,
                    forward_rate=forward_rate,
                    strike=strike,
                    product_type=product_type
                )

            price, delta, vega = fra_future.metrics()

            st.subheader(f"{product_type} Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Price (€)", f"{price:.2f}")
            col2.metric("Delta (€)", f"{delta:.4f}")
            col3.metric("Vega (€)", f"{vega:.4f}")
