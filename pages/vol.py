import streamlit as st
import numpy as np
import pandas as pd

from Models.heston import HestonModel
from functions.vol_function import (
    get_market_prices_yahoo,
    get_all_option_maturities,
    generate_vol_curve,
    plot_vol_curve,
    plot_vol_surface,
)

def app():

    if "ticker" not in st.session_state or not st.session_state["ticker"]:
        st.info("Select a ticker in Parameters & Payoff to view volatility.")
        return

    required = ["S", "r", "q", "option_type", "K"]
    if any(k not in st.session_state for k in required):
        st.info("Please set parameters in Parameters & Payoff first.")
        return

    ticker = st.session_state["ticker"]
    S = st.session_state["S"]
    K0 = st.session_state["K"]
    r = st.session_state["r"]
    q = st.session_state["q"]
    option_type = st.session_state["option_type"].lower()

    st.title("Implied Volatility")

    with st.sidebar:
        model_choice = st.radio(
            "Model",
            ["Black-Scholes (Market IV)", "Heston (Model IV)"]
        )

    maturities = get_all_option_maturities(ticker)
    if not maturities:
        st.warning("No option maturities available.")
        return

    maturity_choice = st.selectbox("Maturity for 2D slice", maturities, index=0)
    T_days_choice = (pd.to_datetime(maturity_choice) - pd.Timestamp.today()).days

    calls, puts, maturity_real = get_market_prices_yahoo(
        ticker,
        T_days=T_days_choice
    )

    if not calls and not puts:
        st.warning("No option chain available for selected maturity.")
        return

    strikes_2d, vols_2d, T_2d, prices_2d = generate_vol_curve(
        S, maturity_real, r, q, calls, puts, option_type
    )

    fig_2d = plot_vol_curve(
        strikes_2d,
        vols_2d,
        maturity=maturity_real,
        K=K0
    )
    st.pyplot(fig_2d)

    market_surface = {}

    for T_date in maturities:
        T_days = (pd.to_datetime(T_date) - pd.Timestamp.today()).days
        if T_days <= 7:
            continue

        calls, puts, maturity = get_market_prices_yahoo(ticker, T_days=T_days)
        if not calls and not puts:
            continue

        strikes, vols, T_years, prices = generate_vol_curve(
            S, maturity, r, q, calls, puts, option_type
        )

        if len(strikes) >= 6 and np.sum(np.isfinite(vols)) >= 6:
            market_surface[maturity] = {
                "strikes": strikes,
                "vols": vols,
                "T": T_years,
                "prices": prices
            }

    if not market_surface:
        st.warning("Not enough market data to build volatility surface.")
        return

    if model_choice == "Black-Scholes (Market IV)":
        fig_3d = plot_vol_surface(market_surface)
        st.plotly_chart(fig_3d, use_container_width=True)
        return

    if st.button("Calibrate Heston", type="primary"):

        maturity_calib = list(market_surface.keys())[0]
        data = market_surface[maturity_calib]

        strikes = data["strikes"]
        prices = [data["prices"][k] for k in strikes]
        T_calib = data["T"]

        initial_guess = [0.04, 2.0, 0.04, 0.30, -0.50]

        params = HestonModel.calibrate(
            S=S,
            r=r,
            T=T_calib,
            q=q,
            option_type=option_type,
            position="buy",
            K_list=strikes,
            market_prices=prices,
            initial_guess=initial_guess,
        )

        st.session_state["heston_calibrated"] = {
            "v0": float(params[0]),
            "kappa": float(params[1]),
            "theta": float(params[2]),
            "sigma_v": float(params[3]),
            "rho": float(params[4]),
        }

        st.success("Heston calibration completed.")

    if "heston_calibrated" not in st.session_state:
        st.info("Calibrate the Heston model to generate the IV surface.")
        return

    calib = st.session_state["heston_calibrated"]

    cols = st.columns(5)
    for col, (k, v) in zip(cols, calib.items()):
        col.metric(k, f"{v:.4f}")

    vol_surface_heston = {}

    for maturity, data in market_surface.items():
        T = data["T"]
        strikes = data["strikes"]

        vols_model = []
        for K in strikes:
            model = HestonModel(
                S, K, r, T, q,
                option_type=option_type,
                position="buy",
                option_class="vanilla",
                **calib
            )
            iv = model.implied_volatility(model.price())
            vols_model.append(iv)

        vol_surface_heston[maturity] = {
            "strikes": strikes,
            "vols": vols_model,
            "T": T
        }

    fig_3d = plot_vol_surface(vol_surface_heston)
    st.plotly_chart(fig_3d, use_container_width=True)
