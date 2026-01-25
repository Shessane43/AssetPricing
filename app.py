
import streamlit as st

st.set_page_config(page_title="Asset Pricing App", layout="wide")
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* Titres */
h1, h2, h3 {
    color: #e6edf3;
    font-weight: 700;
}

/* Selectbox, inputs */
.stSelectbox, .stNumberInput {
    background-color: #161b22;
    border-radius: 10px;
}

/* Radio buttons */
.stRadio > div {
    background-color: #161b22;
    padding: 15px;
    border-radius: 12px;
}

/* Boutons */
.stButton button {
    background: linear-gradient(90deg, #f97316, #fb923c);
    color: black;
    border-radius: 12px;
    font-weight: 700;
    padding: 10px 18px;
    border: none;
}
.stButton button:hover {
    background: linear-gradient(90deg, #fb923c, #f97316);
}

/* Info / success boxes */
.stAlert {
    border-radius: 12px;
}

/* Supprimer padding moche */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

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
