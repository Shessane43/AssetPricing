import streamlit as st
import pandas as pd
from Models.heston import HestonModel
from functions.greeks_function import Greeks
from functions.vol_function import (
    get_all_option_maturities,
    get_market_prices_yahoo,
    generate_vol_curve,
)

def _ensure_heston_calibrated(ticker, S, r, q, option_type):
    if "heston_calibrated" in st.session_state:
        return True

    st.warning("Heston not calibrated yet.")

    maturities = get_all_option_maturities(ticker)
    if not maturities:
        st.error("No option maturities available for calibration.")
        return False

    maturity_choice = st.selectbox(
        "Choose maturity used for Heston calibration",
        maturities,
        index=min(2, len(maturities) - 1),
        key="heston_calib_maturity_choice",
    )

    T_days = (pd.to_datetime(maturity_choice) - pd.Timestamp.today()).days
    calls, puts, maturity_real = get_market_prices_yahoo(ticker, T_days=T_days)

    if not calls and not puts:
        st.error("No option chain available for the selected maturity.")
        return False

    strikes, vols, T_years, prices = generate_vol_curve(S, maturity_real, r, q, calls, puts, option_type)

    if len(strikes) < 6:
        st.error("Not enough strikes to calibrate Heston.")
        return False

    prices_list = [prices[k] for k in strikes]

    with st.expander("Calibration settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            v0 = st.number_input("v0", value=0.04, step=0.01, format="%.4f")
            kappa = st.number_input("kappa", value=2.0, step=0.1, format="%.4f")
            theta = st.number_input("theta", value=0.04, step=0.01, format="%.4f")
        with col2:
            sigma_v = st.number_input("sigma_v", value=0.30, step=0.05, format="%.4f")
            rho = st.number_input("rho", value=-0.50, step=0.05, format="%.4f")

        initial_guess = [v0, kappa, theta, sigma_v, rho]

    if st.button("Calibrate Heston now", type="primary"):
        params = HestonModel.calibrate(
            S=S,
            r=r,
            T=T_years,
            q=q,
            option_type=option_type,
            
            position="buy",
            K_list=strikes,
            market_prices=prices_list,
            initial_guess=initial_guess,
        )

        st.session_state["heston_calibrated"] = {
            "v0": float(params[0]),
            "kappa": float(params[1]),
            "theta": float(params[2]),
            "sigma_v": float(params[3]),
            "rho": float(params[4]),
        }
        st.success(f"Saved in session: {st.session_state['heston_calibrated']}")
        return True

    st.info("Click the button to calibrate and compute Greeks.")
    return False


def app():
    required_keys = ["ticker", "S", "K", "r", "sigma", "T", "q", "option_type", "buy_sell"]
    if not all(k in st.session_state for k in required_keys):
        st.warning("Please set parameters first.")
        return

    ticker = st.session_state["ticker"]
    S = float(st.session_state["S"])
    K = float(st.session_state["K"])
    T = float(st.session_state["T"])
    r = float(st.session_state["r"])
    sigma = float(st.session_state["sigma"])
    q = float(st.session_state["q"])
    option_type = st.session_state["option_type"].lower()
    position = st.session_state["buy_sell"].lower()

    st.subheader("Greeks")

    model_name = st.radio(
        "Greek model",
        ["Black-Scholes", "Heston", "Gamma Variance"],
        horizontal=True,
    )

    st.markdown(
        f"""
        **Ticker**: **{ticker}**

        **Spot (S)**: {S:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Strike (K)**: {K:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Maturity (T)**: {T} year(s)

        **Risk-free rate (r)**: {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Volatility (σ)**: {sigma:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
        **Dividend (q)**: {q:.2%}

        **Option**: **{option_type.capitalize()}** &nbsp;&nbsp;|&nbsp;&nbsp;
        **Position**: **{position.capitalize()}** &nbsp;&nbsp;|&nbsp;&nbsp;
        **Model**: **{model_name}**
        """
    )

    if model_name == "Heston":
        ok = _ensure_heston_calibrated(ticker, S, r, q, option_type)
        if not ok:
            return

        calib = st.session_state["heston_calibrated"]

        greeks = Greeks(
            option_type=option_type,
            model="Heston",
            S=S, K=K, T=T, r=r,
            v0=calib["v0"],
            kappa=calib["kappa"],
            theta_heston=calib["theta"],
            sigma_v=calib["sigma_v"],
            rho=calib["rho"],
            buy_sell=position ,
        )

    elif model_name == "Black-Scholes":
        greeks = Greeks(
            option_type=option_type,
            model="Black-Scholes",
            S=S, K=K, T=T, r=r,
            sigma=sigma,
            buy_sell=position,
        )
    elif model_name == "Gamma Variance":

        st.subheader("Gamma Variance parameters")

        col1, col2 = st.columns(2)
        with col1:
            theta_vg = st.number_input(
                "Theta (VG drift)",
                value=0.0,
                step=0.05,
                format="%.4f"
            )
        with col2:
            nu_vg = st.number_input(
                "Nu (VG variance rate)",
                value=0.2,
                step=0.05,
                format="%.4f"
            )
        greeks = Greeks(
            option_type=option_type,
            model="Gamma Variance",
            S=S, K=K, T=T, r=r,
            sigma=sigma,
            theta=theta_vg,
            nu=nu_vg,
            buy_sell=position,
        )
    else:
        st.info("Gamma Variance: please provide (theta, nu) inputs here if you want it.")
        return

    fig = greeks.plot_all_greeks()
    st.subheader("Greek curves")
    st.pyplot(fig)

    greek_values = {
        "Delta": greeks.delta(),
        "Gamma": greeks.gamma(),
        "Vega": greeks.vega(),
        "Theta": greeks.theta(),
        "Rho": greeks.rho(),
    }

    cols = st.columns(5)
    for col, (name, val) in zip(cols, greek_values.items()):
        col.metric(name, f"{float(val):.4f}")
