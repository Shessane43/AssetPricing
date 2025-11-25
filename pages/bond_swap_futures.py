import streamlit as st
from functions.bond_function import Bond
from functions.swap_function import Swap

def app():
    st.title("Pricing Tool: Bond or Swap")
    
    # Choix du produit
    instrument_type = st.radio("Quel instrument voulez-vous pricer ?", ["Bond", "Swap"])
    
    if instrument_type == "Bond":
        st.subheader("Paramètres du Bond")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        coupon_rate = st.number_input("Taux du coupon (%)", value=5.0, step=0.1) / 100
        rate = st.number_input("Taux d'actualisation (%)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturité (années)", value=5, step=1)
        frequency = st.number_input("Fréquence des coupons par an", value=1, step=1)
        
        if st.button("Calculer le prix du Bond"):
            bond = Bond(nominal, coupon_rate, rate, maturity, frequency)
            price = bond.price()
            st.write(f"Prix du Bond : {price:.2f} €")
            st.write("Évolution de la valeur restante :")
            fig = bond.plot_value_evolution()
            st.pyplot(fig)

    
    elif instrument_type == "Swap":
        st.subheader("Paramètres du Swap")
        nominal = st.number_input("Nominal (€)", value=1000.0, step=100.0)
        fixed_rate = st.number_input("Taux fixe (%)", value=5.0, step=0.1) / 100
        zero_coupon_rate = st.number_input("Taux zéro-coupon (%)", value=3.0, step=0.1) / 100
        maturity = st.number_input("Maturité (années)", value=5, step=1)
        frequency = st.number_input("Fréquence des paiements par an", value=1, step=1)
        
        if st.button("Calculer le prix du Swap"):
            swap = Swap(nominal, fixed_rate, zero_coupon_rate, maturity, frequency)
            value = swap.price_from(0)
            st.write(f"Valeur nette du Swap : {value:.2f} €")
            st.write("Évolution de la valeur restante :")
            fig = swap.plot_value_evolution()
            st.pyplot(fig)

