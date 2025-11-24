import streamlit as st

# Import des pages
from pages import accueil, pricing, greeks

st.set_page_config(page_title="Asset Pricing App", layout="wide")

st.title("Asset Pricing & Option Greeks")

# Création des onglets
tabs = st.tabs(["Accueil", "Pricing", "Greeks"])

with tabs[0]:
    accueil.app()

with tabs[1]:
    pricing.app()

with tabs[2]:
    greeks.app()
