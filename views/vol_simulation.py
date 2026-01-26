import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions.vol_simulation_function import simulate_log_sv

def plot_vol_paths(vol_paths, T, title="Volatility Paths"):
    """
    Plot volatility paths with dark-orange theme.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    time_grid = np.linspace(0, T, vol_paths.shape[1])
    
    # Plot up to 10 sample paths
    for i in range(min(10, vol_paths.shape[0])):
        ax.plot(time_grid, vol_paths[i,:], color="orange", lw=1, alpha=0.7)
    
    # Plot the mean path
    avg_path = vol_paths.mean(axis=0)
    ax.plot(time_grid, avg_path, color="red", lw=2, label="Mean path")

    # Axes and style
    ax.set_xlabel("Time (years)", color="orange")
    ax.set_ylabel("Volatility σ(t)", color="orange")
    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color("orange")
    ax.tick_params(colors="orange")
    ax.grid(True, linestyle="--", color="orange", alpha=0.3)
    ax.set_title(title, color="orange")
    ax.legend(facecolor="black", edgecolor="orange", labelcolor="orange")
    
    return fig

def app():
    st.title("Volatility Simulation (log-SV)")

    # --- User inputs ---
    sigma0 = st.number_input("Initial Volatility σ₀", value=0.2, step=0.01)
    kappa = st.number_input("Mean reversion κ", value=2.0, step=0.1)
    theta = st.number_input("Long-term variance θ", value=0.04, step=0.01)
    sigma_v = st.number_input("Volatility of volatility σ_v", value=0.3, step=0.01)
    T = st.number_input("Maturity (years)", value=1.0, step=0.1)
    N = st.number_input("Time steps", value=252)
    M = st.number_input("Number of paths", value=50)

    if st.button("Simulate Volatility"):
        vol_paths = simulate_log_sv(
            sigma0=sigma0, kappa=kappa, theta=theta,
            sigma_v=sigma_v, T=T, N=N, M=M
        )
        
        # Save simulated paths in session_state for pricing
        st.session_state["simulated_sigma_paths"] = vol_paths
        st.session_state["vol_T"] = T

        # --- Plots ---
        fig1 = plot_vol_paths(vol_paths, T, title="Sample Volatility Paths (log-SV)")
        st.pyplot(fig1)

        fig2 = plot_vol_paths(vol_paths.mean(axis=0).reshape(1,-1), T, title="Average Volatility Path")
        st.pyplot(fig2)

    # --- Model explanation ---
    if st.checkbox("Show Model Explanation"):
        st.markdown("""
        ### Log-Stochastic Volatility Model (log-SV)

        The model jointly describes the dynamics of the **asset price**, its **stochastic volatility**,
        and their **correlation (leverage effect)**.

        **Asset price dynamics**:

        $$
        \\frac{dS_t}{S_t} = \\mu \\, dt + \\sigma_t \\, dW_t^S
        $$

        where:
        - $S_t$ : asset price
        - $\\mu$ : drift
        - $\\sigma_t$ : instantaneous volatility
        - $W_t^S$ : standard Brownian motion

        **Stochastic volatility dynamics (log-variance)**:

        $$
        d \\ln(\\sigma_t^2)
        = \\kappa ( \\ln(\\theta) - \\ln(\\sigma_t^2) ) \\, dt
        + \\sigma_v \\, dW_t^v
        $$

        where:
        - $\\kappa$ : mean reversion speed
        - $\\theta$ : long-term variance
        - $\\sigma_v$ : volatility of volatility
        - $W_t^v$ : standard Brownian motion

        **Correlation (leverage effect)**:

        $$
        d\\langle W_t^S, W_t^v \\rangle = \\rho \\, dt
        $$

        with $\\rho \\in [-1,1]$.

        **Euler discretization** for simulation:

        $$
        \\ln(\\sigma^2_{t+\\Delta t})
        = \\ln(\\sigma^2_t)
        + \\kappa ( \\ln(\\theta) - \\ln(\\sigma^2_t) ) \\Delta t
        + \\sigma_v \\sqrt{\\Delta t} \\, \\varepsilon_t
        $$

        where $\\varepsilon_t \\sim \\mathcal N(0,1)$.
        """)

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
