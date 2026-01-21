import streamlit as st
from functions.bond_function import Bond
from functions.swap_function import Swap

def app():
    st.title("Pricing Tool: Bond or Swap")
    
    # Select instrument
    instrument_type = st.radio("Which instrument would you like to price?", ["Bond", "Swap"])
    
    if instrument_type == "Bond":
        st.subheader("Bond Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, step=0.1) / 100
        rate = st.number_input("Interest Rate (%)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5, step=1)
        frequency = st.number_input("Coupon Payments per Year", value=1, step=1)
        
        if st.button("Calculate Bond Price"):
            bond = Bond(nominal, coupon_rate, rate, maturity, frequency)
            price = bond.price()
            st.write(f"Bond Price: {price:.2f} €")
            st.write("Remaining value evolution:")
            fig = bond.plot_value_evolution()
            st.pyplot(fig)

    
    elif instrument_type == "Swap":
        st.subheader("Swap Parameters")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        fixed_rate = st.number_input("Fixed Rate (%)", value=5.0, step=0.1) / 100
        zero_coupon_rate = st.number_input("Zero-Coupon Rate (%)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturity (years)", value=5, step=1)
        frequency = st.number_input("Payments per Year", value=1, step=1)
        
        if st.button("Calculate Swap Price"):
            swap = Swap(nominal, fixed_rate, zero_coupon_rate, maturity, frequency)
            value = swap.price_from(0)
            st.write(f"Net Swap Value: {value:.2f} €")
            st.write("Remaining value evolution:")
            fig = swap.plot_value_evolution()
            st.pyplot(fig)
