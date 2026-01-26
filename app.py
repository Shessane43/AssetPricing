import streamlit as st

st.set_page_config(
    page_title="Asset Pricing Application",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- GLOBAL STYLE --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    color: #e6edf3;
    font-weight: 700;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION DEFAULTS --------------------
st.session_state.setdefault("ticker", "AAPL")

# -------------------- IMPORT PAGES --------------------
from views import (
    accueil,
    data,
    pricing,
    greeks,
    vol,
    vol_simulation,
    bond_swap_futures,
    structured,
    portfolio,
    parametre,
)

# -------------------- NAV STRUCTURE --------------------
SECTIONS = {
    "Home": [],
    "Derivatives": [
        "Parameters & Payoff",
        "Pricing",
        "Greeks",
    ],
    "Market": [
        "Data",
        "Implied Volatility Surface",
        "Volatility Simulation",
    ],
    "Fixed Income": [
        "Bond & Swap",
    ],
    "Structured Products": [
        "Structured Products",
    ],
    "Portfolio": [
        "My Portfolio",
    ],
}

st.markdown("<h1 style='text-align:center'>Asset Pricing Application</h1>", unsafe_allow_html=True)

main_section = st.selectbox("Main Section", list(SECTIONS.keys()))

sub_section = None
if SECTIONS[main_section]:
    sub_section = st.selectbox("Sub Section", SECTIONS[main_section])

# -------------------- ROUTING --------------------
if main_section == "Home":
    accueil.app()

elif main_section == "Derivatives":
    if sub_section == "Parameters & Payoff":
        parametre.app()
    elif sub_section == "Pricing":
        pricing.app()
    elif sub_section == "Greeks":
        greeks.app()

elif main_section == "Market":
    if sub_section == "Data":
        data.app()
    elif sub_section == "Implied Volatility Surface":
        vol.app()
    elif sub_section == "Volatility Simulation":
        vol_simulation.app()

elif main_section == "Fixed Income":
    bond_swap_futures.app()

elif main_section == "Structured Products":
    structured.app()

elif main_section == "Portfolio":
    portfolio.app()
