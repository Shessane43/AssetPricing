import numpy as np

def simulate_log_sv(S0=100, sigma0=0.2, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5, T=1, N=252, M=50, seed=42):
    """
    Simulates stochastic volatility paths using a log-SV model (Euler discretization).

    The model:
        sigma_t = σ_t : volatility
        dln(σ^2) = κ(θ - ln(σ^2)) dt + σ_v dW

    Parameters
    ----------
    S0 : float, optional
        Initial spot (not used in this function, included for consistency), default is 100.
    sigma0 : float, optional
        Initial volatility, default is 0.2.
    kappa : float, optional
        Mean-reversion speed, default is 2.0.
    theta : float, optional
        Long-term mean of log-volatility, default is 0.04.
    sigma_v : float, optional
        Volatility of volatility, default is 0.3.
    rho : float, optional
        Correlation with underlying (not used in this function), default is -0.5.
    T : float, optional
        Time horizon in years, default is 1.
    N : int, optional
        Number of time steps, default is 252.
    M : int, optional
        Number of simulated paths, default is 50.
    seed : int, optional
        Random seed for reproducibility, default is 42.

    Returns
    -------
    vol_paths : numpy.ndarray of shape (M, N+1)
        Simulated volatility paths. Each row corresponds to one trajectory and each column to a time step.
    """
    np.random.seed(seed)
    dt = T / N
    vol_paths = np.zeros((M, N + 1))
    vol_paths[:, 0] = sigma0

    for i in range(M):
        ln_sigma2 = np.log(sigma0 ** 2)
        for t in range(1, N + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            ln_sigma2 += kappa * (np.log(theta) - ln_sigma2) * dt + sigma_v * dW
            vol_paths[i, t] = np.sqrt(np.exp(ln_sigma2))
    
    return vol_paths



def plot_vol_paths(vol_paths, T, title="Volatility Paths"):
    """
    Plots simulated volatility paths using a dark-orange theme.

    Parameters
    ----------
    vol_paths : np.ndarray
        Array of simulated volatility paths (shape: [num_paths, num_time_steps]).
    T : float
        Time horizon (in years) for the simulation.
    title : str, optional
        Title of the plot, default is "Volatility Paths".

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
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

    # Axes styling
    ax.set_xlabel("Time (years)", color="orange")
    ax.set_ylabel("Volatility σ(t)", color="orange")
    for side in ("bottom", "top", "left", "right"):
        ax.spines[side].set_color("orange")
    ax.tick_params(colors="orange")
    ax.grid(True, linestyle="--", color="orange", alpha=0.3)
    ax.set_title(title, color="orange")
    ax.legend(facecolor="black", edgecolor="orange", labelcolor="orange")
    
    return fig
