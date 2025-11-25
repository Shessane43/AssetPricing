import streamlit as st
from pages import accueil, data , greeks, pricing, vol

st.set_page_config(page_title="Asset Pricing App", layout="wide")
st.title("Asset Pricing & Option Greeks")

tabs = st.tabs(["Accueil", "Data", "Vol","Pricing", "Greeks"])

with tabs[0]:
    accueil.app()  
with tabs[1]:
    data.app()
with tabs[2]:
    vol.app()
with tabs[3]:
    pricing.app()
with tabs[4]:
    greeks.app()

