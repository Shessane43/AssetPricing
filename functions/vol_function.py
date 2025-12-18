import numpy as np
from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

def implied_volatility(
    S, K, T, r, q, market_price, option_type,
    sigma0=0.2, tol=1e-6, max_iter=100
):
    """
    Calcule la volatilité implicite d'une option vanille européenne
    via Newton-Raphson, avec protections contre overflow et valeurs extrêmes.
    """

    # Protection contre valeurs invalides
    if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
        return None

    # Limiter sigma initiale
    sigma0 = max(min(sigma0, 5.0), 1e-4)

    def bs_price(sigma):
        sigma = max(min(sigma, 5.0), 1e-8)  # sigma toujours positif et limité
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        except FloatingPointError:
            return None

        if option_type.lower() == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def vega(sigma):
        sigma = max(min(sigma, 5.0), 1e-8)
        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        except FloatingPointError:
            return 1e-8  # éviter division par zéro
        return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    try:
        vol = newton(
            func=lambda sigma: bs_price(sigma) - market_price,
            x0=sigma0,
            fprime=vega,
            tol=tol,
            maxiter=max_iter
        )
        # Vérifier que vol est dans un intervalle raisonnable
        if vol <= 0 or vol > 5:
            return None
        return vol
    except (RuntimeError, OverflowError, FloatingPointError):
        return None


def get_market_prices_yahoo(ticker, T_days=None):
    stock = yf.Ticker(ticker)
    dates = stock.options
    if not dates:
        return {}

    # maturité la plus proche de T_days
    if T_days is not None:
        maturity = min(
            dates,
            key=lambda x: abs((pd.to_datetime(x) - pd.Timestamp.today()).days - T_days)
        )
    else:
        maturity = dates[0]

    # récup chaîne
    chain = stock.option_chain(maturity)
    df_calls = chain.calls
    df_puts = chain.puts

    # filtrer prix marché valides et options liquides (volume > 0)
    df_calls = df_calls[(df_calls["lastPrice"] > 0) & (df_calls["volume"] > 0)]
    df_puts = df_puts[(df_puts["lastPrice"] > 0) & (df_puts["volume"] > 0)]

    # retourner les prix et la maturité réelle
    return (
        {row['strike']: row['lastPrice'] for _, row in df_calls.iterrows()},
        {row['strike']: row['lastPrice'] for _, row in df_puts.iterrows()},
        maturity
    )

def parity_call_put(S, K, T, r, q, price, option_type_wanted):
    if option_type_wanted.lower() == "call":
        # C = P + S*e^(-qT) - K*e^(-rT)
        price_corrected = price + (S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        # P = C + K*e^(-rT) - S*e^(-qT)
        price_corrected = price + (K * np.exp(-r * T) - S * np.exp(-q * T))
    return price_corrected

def generate_vol_curve(S, T, r, q, market_prices_calls, market_prices_puts, option_type):
    """
    Génère la courbe de volatilité implicite pour plusieurs strikes
    """
    T = (pd.to_datetime(T) - pd.Timestamp.today()).days / 365.0
    corrected_prices = {}
    if option_type.lower() == "put":
        for K in market_prices_calls:
            corrected_price = parity_call_put(S, K, T, r, q, market_prices_calls[K], "put")
            corrected_prices[K] = corrected_price
    else:
        for K in market_prices_puts:
            corrected_price = parity_call_put(S, K, T, r, q, market_prices_puts[K], "call")
            corrected_prices[K] = corrected_price

    # maintenant on a 2 disctionnaire avec des K et des options du même type
    # On va fusionner les deux dictionnaires en prenant les prix moyens pour les strikes communs

    market_prices = {}
    if option_type.lower() == "put":
        market_prices.update(market_prices_puts)
    else:
        market_prices.update(market_prices_calls)

    for K, price in corrected_prices.items():
        if K in market_prices:
            market_prices[K] = 0.5 * (market_prices[K] + price)
        else:
            market_prices[K] = price
    
    # on récupère les stirkes et on calcule les vols
    strikes = sorted(market_prices.keys())
    vols = []
    for K in strikes:
        price = market_prices[K]
        vol = implied_volatility(S, K, T, r, q, price, option_type)
        vols.append(vol)
    return strikes, vols

def smooth_vol_curve(strikes, vols, num_points=100, smoothing_factor=1):
    """
    Lisser une courbe de volatilité implicite avec une smoothing spline.
    
    Args:
        strikes (list or np.array): liste des strikes
        vols (list or np.array): volatilités implicites correspondantes
        num_points (int): nombre de points à générer pour la courbe lisse
        smoothing_factor (float or None): paramètre de lissage pour UnivariateSpline.
            Si None, le spline interpole exactement.
    
    Returns:
        tuple: (strikes_smooth, vols_smooth) 
            - strikes_smooth: np.array de strikes lissés
            - vols_smooth: np.array de vols lissés
    """
    
    strikes = np.array(strikes)
    vols = np.array(vols)
    
    mask = vols != None
    strikes = strikes[mask]
    vols = vols[mask]
    
    # Spline cubique
    spline = UnivariateSpline(strikes, vols, s=smoothing_factor, k=3)
    
    strikes_smooth = np.linspace(strikes.min(), strikes.max(), num_points)
    vols_smooth = spline(strikes_smooth)
    
    return strikes_smooth, vols_smooth

def plot_vol_curve(strikes, vols, maturity=None, K=None, title="Courbe de volatilité implicite", smoothing_factor=1, num_points=100):
    """
    Trace la courbe de volatilité implicite avec smoothing spline et marque le strike choisi.

    Args:
        strikes: liste ou np.array de strikes
        vols: liste ou np.array de volatilités correspondantes
        maturity: date réelle de maturité pour le titre
        K_user: strike choisi par l'utilisateur pour afficher la ligne verticale et la vol estimée
        title: titre de la figure
        smoothing_factor: paramètre de lissage pour la spline
        num_points: nombre de points générés pour la spline
    
    Returns:
        fig: figure matplotlib
    """

    # --- Points réels ---
    strikes = np.array(strikes)
    vols = np.array(vols)
    mask = vols != None
    strikes_plot = strikes[mask]
    vols_plot = vols[mask]

    # --- Spline lissée via la fonction existante ---
    from copy import deepcopy
    strikes_smooth, vols_smooth = smooth_vol_curve(deepcopy(strikes_plot), deepcopy(vols_plot), num_points=num_points, smoothing_factor=smoothing_factor)

    # --- Création du plot ---
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12,6))

    # Points réels en croix
    ax.scatter(strikes_plot, vols_plot, marker='x', color='orange', label='Points réels', s=80)

    # Spline lissée en trait plein
    ax.plot(strikes_smooth, vols_smooth, '-', color='blue', lw=2, label='Spline lissée')

    # Ligne verticale et estimation pour K
    if K is not None:
        # estimation via la spline
        vol_at_K = np.interp(K, strikes_smooth, vols_smooth)
        ax.axvline(K, color='red', linestyle='--', lw=2)
        ax.scatter(K, vol_at_K, color='red', s=100, zorder=5)
        ax.legend(title=f"Vol estimée au strike {K}: {vol_at_K:.4f}")
    else:
        ax.legend()

    # Titres et axes
    full_title = title
    if maturity is not None:
        full_title += f" - Maturité: {maturity}"
    ax.set_title(full_title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Volatilité implicite")
    ax.grid(True)

    # Ajustement des limites
    if vols_plot.size > 0:
        ax.set_ylim(0, max(vols_plot)*1.2)
    if strikes_plot.size > 0:
        buffer = (max(strikes_plot) - min(strikes_plot)) * 0.1
        ax.set_xlim(min(strikes_plot)-buffer, max(strikes_plot)+buffer)

    return fig

def get_all_option_maturities(ticker):
    """
    Récupère toutes les maturités disponibles pour un ticker sur Yahoo Finance.

    Args:
        ticker (str): symbole de l'actif

    Returns:
        list: liste de dates de maturité (format 'YYYY-MM-DD')
    """
    stock = yf.Ticker(ticker)
    maturities = stock.options  # liste des dates de maturité
    if not maturities:
        print(f"Aucune maturité disponible pour {ticker}")
        return []
    return maturities


def generate_vol_curves_multiple_maturities(S, maturities, r, q, option_type, ticker):
    """
    Génère les courbes de volatilité implicite pour plusieurs maturités
    en utilisant get_market_prices_yahoo.

    Args:
        S (float): prix spot
        maturities (list of str): liste de dates de maturité 'YYYY-MM-DD'
        r (float): taux sans risque
        q (float): dividende
        option_type (str): 'call' ou 'put'
        ticker (str): symbole du titre

    Returns:
        dict: {maturity_date: {'strikes': [...], 'vols': [...]}}
    """
    vol_curves = {}

    for T_date in maturities:
        # récupérer les prix de marché via la fonction existante
        market_prices_calls, market_prices_puts, real_maturity = get_market_prices_yahoo(ticker, T_days=(pd.to_datetime(T_date) - pd.Timestamp.today()).days)
        if not market_prices_calls and not market_prices_puts:
            print(f"Aucune option disponible pour la maturité {T_date}")
            continue

        # convertir T en années
        T_years = (pd.to_datetime(real_maturity) - pd.Timestamp.today()).days / 365.0

        # utiliser generate_vol_curve pour calculer strikes et vols
        strikes, vols = generate_vol_curve(S, real_maturity, r, q, market_prices_calls, market_prices_puts, option_type)

        vol_curves[real_maturity] = {"strikes": strikes, "vols": vols}

    return vol_curves

def plot_vol_surface(vol_curves):
    """
    vol_curves : dict {maturity: {"strikes": [...], "vols": [...]}}

    Surface 3D interactive Plotly :
        X = Strike
        Z = Maturité (années)
        Y = Volatilité implicite
    """
    # --- Préparer les données brutes ---
    all_strikes = []
    all_vols = []
    all_maturities = []

    for maturity, data in vol_curves.items():
        strikes = np.array(data["strikes"])
        vols = np.array(data["vols"])
        # maturité en années
        T_years = (pd.to_datetime(maturity) - pd.Timestamp.today()).days / 365.0
        maturities = np.full_like(strikes, T_years, dtype=float)

        # filtrer vols valides
        mask = vols != None
        all_strikes.extend(strikes[mask])
        all_vols.extend(vols[mask])
        all_maturities.extend(maturities[mask])

    # --- Créer une grille régulière pour surface ---
    strikes_grid = np.linspace(min(all_strikes), max(all_strikes), 50)
    maturities_grid = np.linspace(min(all_maturities), max(all_maturities), 50)
    X, Z = np.meshgrid(strikes_grid, maturities_grid)  # X=strike, Z=maturité
    # interpolation des vols
    Y = griddata(
        points=(all_strikes, all_maturities),
        values=all_vols,
        xi=(X, Z),
        method='linear'
    )

    # --- Créer la surface 3D ---
    fig = go.Figure(data=[go.Surface(x=X, y=Z, z=Y, colorscale='Viridis')])
    fig.update_layout(
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturité (années)',
            zaxis_title='Volatilité implicite'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700
    )
    return fig
