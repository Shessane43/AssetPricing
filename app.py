import streamlit as st
from pages import accueil, data , greeks, pricing

st.set_page_config(page_title="Asset Pricing App", layout="wide")
st.title("Asset Pricing & Option Greeks")

tabs = st.tabs(["Accueil", "Data", "Pricing", "Greeks"])

with tabs[0]:
    accueil.app()  
with tabs[1]:
    data.app()
with tabs[2]:
    greeks.app()
with tabs[3]:
    pricing.app()

