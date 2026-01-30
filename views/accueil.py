import streamlit as st

def _go(main, sub=None):
    st.session_state["__nav_target__"] = (main, sub)
    st.rerun()

def app():
    st.markdown("""
    <style>
    .home-wrap {max-width:1100px;margin:0 auto;}
    .qs {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px 20px;
        margin: 0 0 22px 0;
    }
    .qs b{color:#e6edf3;}
    .muted{color:#9ca3af;}
    .card {
        background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 22px;
        padding: 22px;
        height: 210px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
        transition: all 0.22s ease;
    }
    .card:hover{
        transform: translateY(-6px);
        border: 1px solid rgba(249,115,22,0.30);
        box-shadow: 0 26px 60px rgba(0,0,0,0.55);
    }
    .card-title{font-size:20px;font-weight:750;margin-bottom:10px;}
    .card-desc{font-size:14.5px;line-height:1.55;color:#d1d5db;margin-bottom:10px;}
    .card-list{font-size:13.5px;line-height:1.6;color:#9ca3af;}
    .cta button{
        width:100%;
        border-radius: 16px;
        padding: 14px 14px;
        border: 1px solid rgba(249,115,22,0.25);
        background: rgba(249,115,22,0.10);
        color:#e6edf3;
        font-weight:650;
        transition: all 0.2s ease;
    }
    .cta button:hover{
        background: rgba(249,115,22,0.18);
        border: 1px solid rgba(249,115,22,0.40);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="home-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="qs">
      <b>Quick start</b><br>
      <span class="muted">
      1) Start with <b>Derivatives → Parameters & Payoff</b> to set the underlying + option specs.<br>
      2) Go to <b>Pricing</b> to compare models (BS / Heston / VG / Tree).<br>
      3) Use <b>Greeks</b> for sensitivities. Then explore <b>Market → Implied Volatility Surface</b>.
      </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, _ = st.columns([1, 1, 1], gap="large")

    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-title"> Derivatives</div>
          <div class="card-desc">
            Everything for option setup, payoff visualization, pricing and Greeks.
          </div>
          <div class="card-list">
            • Parameters & Payoff (entry point)<br>
            • Pricing (BS / Heston (Exotic and Vanilla) / VG / Tree / Bachelier / Merton Jump Diffusion)<br>
            • Greeks (curves + metrics)
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cta">', unsafe_allow_html=True)
        if st.button("Open Parameters & Payoff"):
            _go("Derivatives", "Parameters & Payoff")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div class="card-title"> Market</div>
          <div class="card-desc">
            Market data + implied volatility curves & surfaces.
          </div>
          <div class="card-list">
            • Data (spot / chains)<br>
            • IV Surface (2D slice + 3D surface)<br>
            • Volatility Simulation
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cta">', unsafe_allow_html=True)
        if st.button("Open Implied Volatility Surface", key="go_iv"):
            _go("Market", "Implied Volatility Surface")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
          <div class="card-title"> Portfolio</div>
          <div class="card-desc">
            Aggregate positions, valuation and risk tracking at portfolio level.
          </div>
          <div class="card-list">
            • Positions overview<br>
            • Portfolio valuation<br>
            • Risk tracking
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cta">', unsafe_allow_html=True)
        if st.button("Open My Portfolio", key="go_ptf"):
            _go("Portfolio", "My Portfolio")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    with c4:
      st.markdown("""
      <div class="card">
        <div class="card-title"> Fixed Income</div>
        <div class="card-desc">
          Pricing and analysis of interest rate products.
        </div>
        <div class="card-list">
          • Zero-coupon & coupon bonds<br>
          • Yield curves (bootstrapping)<br>
          • Duration, convexity & sensitivities
        </div>
      </div>
      """, unsafe_allow_html=True)

      st.markdown('<div class="cta">', unsafe_allow_html=True)
      if st.button("Open Fixed Income", key="go_fi"):
          _go("Fixed Income", "Pricing")
      st.markdown('</div>', unsafe_allow_html=True)
    with c5:
      st.markdown("""
      <div class="card">
        <div class="card-title"> Structured Products</div>
        <div class="card-desc">
          Design and pricing of structured payoff products.
        </div>
        <div class="card-list">
          • Autocalls / Phoenix / Athena<br>
          • Capital protected notes<br>
          • Payoff decomposition & Greeks
        </div>
      </div>
      """, unsafe_allow_html=True)

      st.markdown('<div class="cta">', unsafe_allow_html=True)
      if st.button("Open Structured Products", key="go_sp"):
          _go("Structured Products", "Overview")
      st.markdown('</div>', unsafe_allow_html=True)
