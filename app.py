import streamlit as st
from pages import accueil, data, greeks, pricing, vol, bond_swap_futures, structured, portfolio, parametre

# Page configuration
st.set_page_config(
    page_title="Asset Pricing App",
    layout="wide"
)

# Page selection: simple selectbox in the main area (not sidebar)
page = st.selectbox(
    "Select a page:",
    [
        "Home", 
        "Parameters & Payoff", 
        "Data",
        "Pricing",
        "Greeks",
        "Implied Volatility Surface",
        "Bond, Swap & Futures",
        "Structured Products",
        "My Portfolio"
    ]
)

# Display pages
if page == "Home":
    accueil.app()
elif page == "Parameters & Payoff":
    parametre.app()
elif page == "Data":
    data.app()
elif page == "Pricing":
    pricing.app()
elif page == "Greeks":
    greeks.app()
elif page == "Implied Volatility Surface":
    vol.app()
elif page == "Bond, Swap & Futures":
    bond_swap_futures.app()
elif page == "Structured Products":
    structured.app()
elif page == "My Portfolio":
    portfolio.app()
