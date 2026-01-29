import streamlit as st
import numpy as np
import pandas as pd

from Models.heston import HestonModel
from functions.vol_function import (
    get_market_prices_yahoo,
    get_all_option_maturities,
    generate_vol_curve,
    clean_iv_points,
    plot_vol_curve,
    plot_iv_surface_KT,
    generate_heston_iv_curve,
    select_calibration_points
)


def app():

    # =========================
    # CHECK SESSION
    # =========================
    if "ticker" not in st.session_state or not st.session_state["ticker"]:
        st.info("Select a ticker in Parameters & Payoff to view volatility.")
        return

    required = ["S", "r", "q", "option_type", "K"]
    if any(k not in st.session_state for k in required):
        st.info("Please set parameters in Parameters & Payoff first.")
        return

    ticker = st.session_state["ticker"]
    S = float(st.session_state["S"])
    K0 = float(st.session_state["K"])
    r = float(st.session_state["r"])
    sigma = st.session_state["sigma"]
    q = float(st.session_state["q"])
    option_type = st.session_state["option_type"].lower()
    buy_sell = st.session_state["buy_sell"]

    st.title("Implied Volatility")

    # =========================
    # MODEL SELECTION
    # =========================
    model_choice = st.radio(
        "Model",
        ["Black-Scholes (Market IV)", "Heston (Model IV)"],
        horizontal=True
    )

    st.markdown(
        f"""
        **Ticker**: **{ticker}**

        **Spot (S)**: {S:.4f}  
        **Reference strike (K)**: {K0:.4f}

        **Risk-free rate (r)**: {r:.2%}  
        **Dividend yield (q)**: {q:.2%}

        **Option type**: **{option_type}**  
        **Position**: **{buy_sell}**
        """
    )

    st.title("Implied Volatility")

    st.radio(
        "Model",
        ["Black-Scholes", "Heston"],
        horizontal=True,
        key="iv_model"
    )
    model_choice = st.session_state.iv_model


    # =========================
    # MARKET DATA
    # =========================
    maturities = get_all_option_maturities(ticker)
    if not maturities:
        st.warning("No option maturities available.")
        return

    maturity_choice = st.selectbox("Maturity for 2D smile", maturities)

    T_days = (pd.to_datetime(maturity_choice) - pd.Timestamp.today()).days
    calls, puts, maturity_real = get_market_prices_yahoo(ticker, T_days)

    if not calls and not puts:
        st.warning("No option chain available for selected maturity.")
        return

    strikes_2d, vols_2d, T_2d, prices_2d = generate_vol_curve(
        S, maturity_real, r, q, calls, puts, option_type
    )

    # =========================
    # 2D MARKET IV
    # =========================
    if model_choice == "Black-Scholes (Market IV)":
        fig_2d = plot_vol_curve(
            strikes_2d,
            vols_2d,
            maturity=maturity_real,
            K=K0,
            title="Market Implied Volatility Curve"
        )
        st.pyplot(fig_2d)

    # =========================
    # BUILD MARKET SURFACE
    # =========================
    market_surface = {}
    for T_date in maturities:
        T_days_i = (pd.to_datetime(T_date) - pd.Timestamp.today()).days
        if T_days_i < 7:
            continue

        calls_m, puts_m, maturity_m = get_market_prices_yahoo(ticker, T_days_i)
        if not calls_m and not puts_m:
            continue

        strikes, vols, T, prices = generate_vol_curve(
            S, maturity_m, r, q, calls_m, puts_m, option_type
        )
        strikes, vols = clean_iv_points(S, strikes, vols, T, r, q)

        if len(strikes) >= 6:
            market_surface[maturity_m] = {
                "strikes": strikes.tolist(),
                "vols": vols.tolist(),
                "T": float(T),
                "prices": {float(k): float(v) for k, v in prices.items()}
            }

    if not market_surface:
        st.warning("Not enough data to build volatility surface.")
        return

    # =========================
    # MARKET IV SURFACE
    # =========================
    if model_choice == "Black-Scholes (Market IV)":
        fig_3d = plot_vol_surface(market_surface)
        st.plotly_chart(fig_3d, use_container_width=True)
        return

    
    # =========================
    # HESTON MULTI-MATURITY CALIBRATION
    # =========================

    if st.button("Calibrate Heston (Multi-maturity)", type="primary"):

        data_by_maturity = []

        for maturity, data in market_surface.items():
            T = float(data["T"])
            K_list = list(data["strikes"])
            prices = [float(data["prices"][k]) for k in K_list]

            data_by_maturity.append({
                "T": T,
                "K_list": K_list,
                "market_prices": prices
            })

        initial_guess = [0.04, 2.0, 0.04, 0.30, -0.50]

        params, res = HestonModel.calibrate_multi_maturity(
            S=S,
            r=r,
            q=q,
            option_type=option_type,
            position="buy",
            data_by_maturity=data_by_maturity,
            initial_guess=initial_guess,
            use_feller_penalty=True
        )

        st.session_state["heston_calibrated"] = {
            "params": params,
            "res": res
        }
    # =========================
    # LOAD HESTON PARAMETERS
    # =========================

    if "heston_calibrated" not in st.session_state:
        st.info("Calibrate Heston to display model IV.")
        return

    params = st.session_state["heston_calibrated"]["params"]

    calib = {
        "v0": float(params[0]),
        "kappa": float(params[1]),
        "theta": float(params[2]),
        "sigma_v": float(params[3]),
        "rho": float(params[4]),
    }
    # =========================
    # 2D HESTON IV
    # =========================
    vols_heston_2d = []
    for K in strikes_2d:
        model = HestonModel(
            S=S,
            K=K,
            r=r,
            T=T_2d,
            q=q,
            option_type=option_type,
            position="buy",
            option_class="vanilla",
            **calib
        )
        iv = model.implied_volatility(model.price())
        vols_heston_2d.append(iv)

    fig_2d = plot_vol_curve(
        strikes_2d,
        vols_heston_2d,
        maturity=maturity_real,
        K=K0,
        title="Heston Implied Volatility Curve"
    )
    st.pyplot(fig_2d)

    # =========================
    # 3D HESTON SURFACE
    # =========================
    vol_surface_heston = {}

    vols_heston_2d = generate_heston_iv_curve(
        S=S,
        r=r,
        q=q,
        option_type=option_type,
        strikes=strikes_2d,
        T=T_2d,
        heston_params=calib,
        HestonModel=HestonModel
    )

    mask = np.isfinite(vols_heston_2d)
    if np.sum(mask) >= 4:
        fig_heston_2d = plot_vol_curve(
            S=S,
            strikes=np.asarray(strikes_2d)[mask],
            vols=vols_heston_2d[mask],
            maturity=maturity_real,
            K=K0,
            T=T_2d,
            r=r,
            q=q,
            title="Heston Model Implied Volatility Smile"
        )
        st.pyplot(fig_heston_2d)
    else:
        st.warning("Not enough valid Heston IV points for 2D smile.")


    heston_surface = {}
    for maturity, data in market_surface.items():
        T = float(data["T"])
        strikes = np.asarray(data["strikes"], dtype=float)

        vols_model = generate_heston_iv_curve(
            S=S,
            r=r,
            q=q,
            option_type=option_type,
            strikes=strikes,
            T=T,
            heston_params=calib,
            HestonModel=HestonModel
        )

        heston_surface[maturity] = {
            "strikes": strikes.tolist(),
            "vols": vols_model.tolist(),
            "T": T
        }

    fig_heston_3d = plot_iv_surface_KT(
        heston_surface,
        title="Heston Model Implied Volatility Surface (K, T)"
    )
    st.plotly_chart(fig_heston_3d, use_container_width=True)
