import numpy as np
import matplotlib.pyplot as plt

def generate_vol_curve(vol_implicite_func, S, T, r, type_option, strikes=None):
    """
    Génère une courbe de volatilité implicite pour une série de strikes.
    
    Returns :
    - strikes : array des strikes
    - vols : array des vols implicites correspondantes
    """
    if strikes is None:
        strikes = np.arange(int(S*0.8), int(S*1.2)+1, 5)
    
    vols = []
    for K in strikes:
        vol = vol_implicite_func(S=S, K=K, T=T, r=r, type_option=type_option)
        vols.append(vol)
    
    return np.array(strikes), np.array(vols)


def plot_vol_curve(strikes, vols, title="Courbe de volatilité implicite"):
    """
    Trace la courbe de volatilité implicite.
    """
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(strikes, vols, marker="o", color="orange", lw=2)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Vol implicite")
    ax.set_title(title)
    ax.grid(True)
    return fig
