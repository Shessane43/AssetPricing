
import streamlit as st

st.set_page_config(page_title="Asset Pricing App", layout="wide")

if "ticker" not in st.session_state:
    st.session_state["ticker"] = "AAPL"

if "S" not in st.session_state:
    st.session_state["S"] = None

from pages import (
    accueil, data, greeks, pricing, vol,
    bond_swap_futures, structured, portfolio,
    parametre, vol_simulation
)

st.markdown('<h1 class="title">Asset Pricing Application</h1>', unsafe_allow_html=True)
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
        "Volatility Simulation",
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
elif page == "Volatility Simulation":
    vol_simulation.app()
elif page == "Bond, Swap & Futures":
    bond_swap_futures.app()
elif page == "Structured Products":
    structured.app()
elif page == "My Portfolio":
    portfolio.app()
