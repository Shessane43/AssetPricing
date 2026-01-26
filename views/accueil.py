import streamlit as st

def app():

    # -------------------- CSS --------------------
    st.markdown("""
    <style>
    .card-btn button {
        width: 100%;
        height: 180px;
        background: linear-gradient(135deg, #f97316, #fb923c);
        border: none;
        border-radius: 22px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.35);
        color: #111827;
        text-align: left;
        padding: 28px;
        font-size: 16px;
        transition: all 0.25s ease;
        cursor: pointer;
    }

    .card-btn button:hover {
        transform: translateY(-8px) scale(1.02);
    }

    .card-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .card-text {
        font-size: 15px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- HERO --------------------
    st.markdown("""
    <div style="text-align:center;margin:40px 0 60px 0">
        <h1 style="font-size:56px;font-weight:800">Asset Pricing Application</h1>
        <p style="color:#9da5b4;font-size:18px">
            Pricing • Greeks • Volatility • Portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    # ---------------- DERIVATIVES ----------------
    with col1:
        with st.container():
            st.markdown('<div class="card-btn">', unsafe_allow_html=True)
            if st.button("🧮 Derivatives\n\nPricing, Greeks, payoff diagrams\nBlack-Scholes, Heston, Variance Gamma"):
                st.session_state.page = "pricing"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- MARKET ----------------
    with col2:
        with st.container():
            st.markdown('<div class="card-btn">', unsafe_allow_html=True)
            if st.button("📊 Market Analysis\n\nMarket data, implied volatility surfaces,\nvolatility simulations"):
                st.session_state.page = "data"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PORTFOLIO ----------------
    with col3:
        with st.container():
            st.markdown('<div class="card-btn">', unsafe_allow_html=True)
            if st.button("💼 Portfolio\n\nPositions, valuation,\nrisk tracking"):
                st.session_state.page = "portfolio"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
