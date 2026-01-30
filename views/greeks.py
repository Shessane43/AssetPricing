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

    with st.expander("Heston calibration", expanded=False):

        st.warning("Heston not calibrated yet.")

        maturities = get_all_option_maturities(ticker)
        if not maturities:
            st.error("No option maturities available.")
            return False

        maturity_choice = st.selectbox(
            "Calibration maturity",
            maturities,
            index=min(2, len(maturities) - 1),
        )

        T_days = (pd.to_datetime(maturity_choice) - pd.Timestamp.today()).days
        calls, puts, maturity_real = get_market_prices_yahoo(ticker, T_days)

        strikes, vols, T_years, prices = generate_vol_curve(
            S, maturity_real, r, q, calls, puts, option_type
        )

        if len(strikes) < 6:
            st.error("Not enough strikes to calibrate Heston.")
            return False

        col1, col2 = st.columns(2)
        with col1:
            v0 = st.number_input("v0", value=0.04, format="%.4f")
            kappa = st.number_input("kappa", value=2.0, format="%.4f")
            theta = st.number_input("theta", value=0.04, format="%.4f")
        with col2:
            sigma_v = st.number_input("sigma_v", value=0.30, format="%.4f")
            rho = st.number_input("rho", value=-0.50, format="%.4f")

        if st.button("Calibrate Heston", type="primary"):

            params = HestonModel.calibrate(
                S=S,
                r=r,
                q=q,
                option_type="call",
                K_list=strikes,
                T_list=[T_years] * len(strikes),
                market_prices=[prices[k] for k in strikes],
                initial_guess=[v0, kappa, theta, sigma_v, rho],
            )

            st.session_state["heston_calibrated"] = dict(
                v0=float(params[0]),
                kappa=float(params[1]),
                theta=float(params[2]),
                sigma_v=float(params[3]),
                rho=float(params[4]),
            )

            st.success("Heston calibrated ✔")
            return True

    return False

def app():

    required = ["ticker", "S", "K", "T", "r", "sigma", "q", "option_type", "buy_sell"]
    if not all(k in st.session_state for k in required):
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

    is_vanilla = option_type in ["call", "put"]
    allow_curves = True

    st.subheader("Greeks")

    model_name = st.radio(
        "Greek model",
        ["Black-Scholes", "Heston", "Gamma Variance"],
        horizontal=True,
    )

    st.markdown(
        f"""
        **Ticker**: **{ticker}**

        **Spot (S)**: {S:.4f} | **Strike (K)**: {K:.4f} | **Maturity (T)**: {T}

        **r**: {r:.2%} | **σ**: {sigma:.2%} | **q**: {q:.2%}

        **Option**: {option_type.capitalize()} | **Position**: {position.capitalize()} | **Model**: {model_name}
        """
    )

    if model_name == "Black-Scholes":
        if not is_vanilla:
            st.warning("Black-Scholes Greeks only available for vanilla options.")
            return

        greeks = Greeks(
            option_type=option_type,
            model="Black-Scholes",
            S=S, K=K, T=T, r=r,
            sigma=sigma,
            buy_sell=position,
        )

    elif model_name == "Gamma Variance":
        if not is_vanilla:
            st.warning("Variance Gamma Greeks only available for vanilla options.")
            return

        col1, col2 = st.columns(2)
        with col1:
            theta_vg = st.number_input("Theta (VG)", value=0.0, format="%.4f")
        with col2:
            nu_vg = st.number_input("Nu (VG)", value=0.2, format="%.4f")

        greeks = Greeks(
            option_type=option_type,
            model="Gamma Variance",
            S=S, K=K, T=T, r=r,
            sigma=sigma,
            theta=theta_vg,
            nu=nu_vg,
            buy_sell=position,
        )

    elif model_name == "Heston":

        calibrated = _ensure_heston_calibrated(ticker, S, r, q, option_type)
        if not calibrated:
            st.info("Please calibrate Heston to compute Greeks.")
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
            buy_sell=position,
        )

        if not is_vanilla:
            st.info(
                "Exotic Greeks under Heston are computed by Monte Carlo finite differences. "
                "Only point Greeks are shown."
            )
            allow_curves = False

    if allow_curves:
        st.subheader("Greek curves")
        fig = greeks.plot_all_greeks()
        st.pyplot(fig)

    st.subheader("Point Greeks")

    greek_vals = {
        "Delta": greeks.delta(),
        "Gamma": greeks.gamma(),
        "Vega": greeks.vega(),
        "Theta": greeks.theta(),
        "Rho": greeks.rho(),
    }

    cols = st.columns(5)
    for col, (k, v) in zip(cols, greek_vals.items()):
        col.metric(k, f"{float(v):.4f}")
