import streamlit as st
from pages import accueil, data , greeks, pricing, vol, bond_swap_futures, structured, portfolio

st.set_page_config(page_title="Asset Pricing App", layout="wide")
st.title("Volatility & Option Pricing")

tabs = st.tabs(["Accueil", "Data","Pricing", "Greeks","Implied Volatility","Bond & Swap & Futures",'Structured Products',"My Own Portfolio"])

with tabs[0]:
    accueil.app()  
with tabs[1]:
    data.app()
with tabs[2]:
    pricing.app()
with tabs[3]:
    greeks.app()
with tabs[4]:
    vol.app()
with tabs[5]:
    bond_swap_futures.app()
with tabs[6]:
    structured.app()
with tabs[7]:
    portfolio.app()


