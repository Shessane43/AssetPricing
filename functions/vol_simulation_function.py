import numpy as np

def simulate_log_sv(S0=100, sigma0=0.2, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5, T=1, N=252, M=50, seed=42):
    """
    Simule des trajectoires de volatilité stochastique selon un modèle log-SV (discretisation Euler).
    
    sigma_t = σ_t : volatilité
    dln(σ^2) = κ(θ - ln(σ^2))dt + σ_v dW
    
    Args:
        S0 : Spot initial (non utilisé ici, juste pour cohérence)
        sigma0 : volatilité initiale
        kappa : vitesse de retour à la moyenne
        theta : niveau moyen log-volatilité
        sigma_v : volatilité de la volatilité
        rho : corrélation (non utilisée si seulement σ)
        T : maturité en années
        N : nombre de pas de temps
        M : nombre de trajectoires
    Returns:
        vol_paths : np.array(M, N+1)
    """
    np.random.seed(seed)
    dt = T/N
    vol_paths = np.zeros((M, N+1))
    vol_paths[:,0] = sigma0

    for i in range(M):
        ln_sigma2 = np.log(sigma0**2)
        for t in range(1, N+1):
            dW = np.random.normal(0, np.sqrt(dt))
            ln_sigma2 += kappa*(np.log(theta) - ln_sigma2)*dt + sigma_v*dW
            vol_paths[i,t] = np.sqrt(np.exp(ln_sigma2))
    
    return vol_paths
