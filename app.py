import streamlit as st

st.set_page_config(
    page_title="Asset Pricing Application",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- GLOBAL STYLE --------------------
st.markdown("""
<div style="text-align:center;margin-top:10px;margin-bottom:10px">
  <div style="font-size:52px;font-weight:800;color:#e6edf3;">Asset Pricing Application</div>
  <div style="color:#9ca3af;font-size:18px;margin-top:6px;">
    Pricing • Greeks • Volatility • Portfolio
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:0;height:1px;background:rgba(255,255,255,0.08);margin:18px 0 24px 0'/>", unsafe_allow_html=True)


# -------------------- SESSION DEFAULTS --------------------
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

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
# -------------------- HANDLE NAV FROM HOME --------------------
if "__nav_target__" in st.session_state:
    target_main, target_sub = st.session_state.pop("__nav_target__")

    st.session_state["main_section"] = target_main
    if target_sub is not None:
        st.session_state["sub_section"] = target_sub

main_section = st.selectbox(
    "Main Section",
    list(SECTIONS.keys()),
    key="main_section"
)

sub_section = None
if SECTIONS[main_section]:
    sub_section = st.selectbox(
        "Sub Section",
        SECTIONS[main_section],
        key="sub_section"
    )


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
