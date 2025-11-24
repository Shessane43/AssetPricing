import streamlit as st
import yfinance as yf
import numpy as np 
import matplotlib.pyplot as plt

# Exemple : quelques tickers connus
ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "BABA", "ORCL"]

# --- CSS pour customiser le selectbox ---
st.markdown("""
    <style>
    /* Fond de la liste déroulante */
    div.stSelectbox > div > div > div {
        background-color: #e6f2ff;  /* bleu clair */
        color: #003366;             /* texte bleu foncé */
        border-radius: 5px;
    }
    /* Texte de l'élément sélectionné */
    div.stSelectbox > div > div > div > span {
        color: #003366;
        font-weight: bold;
    }
    /* Fond de l'option survolée */
    div.stSelectbox div[role="listbox"] div[role="option"]:hover {
        background-color: #99ccff !important;
        color: #000000 !important;
    }
    /* Fond de l'option sélectionnée */
    div.stSelectbox div[role="listbox"] div[aria-selected="true"] {
        background-color: #3399ff !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

def app():
    st.title("Asset Pricing & Option Greeks")
    st.subheader("Bienvenue dans l'application")

    st.markdown("---")
    st.write("Veuillez choisir votre action et les paramètres de l'option :")

    # --- Sélection ticker ---
    suggestions = [t for t in ALL_TICKERS]
    if suggestions:
        st.session_state.ticker = st.selectbox("Sélectionnez le ticker", suggestions)
        
        # Récupérer le dernier prix via yfinance
        ticker_data = yf.Ticker(st.session_state.ticker)
        last_price = ticker_data.history(period="1d")['Close'].iloc[-1]
        st.session_state.S = last_price
        st.info(f"Prix actuel de {st.session_state.ticker} : {last_price:.2f} $")
    else:
        st.warning("Aucun ticker ne correspond à votre saisie.")


    st.session_state.option_buysell = st.selectbox("Position", ["Buy", "Sell"])

    # --- Classe d'option ---
    st.session_state.option_class = st.selectbox("Classe d'option", ["Vanille", "Exotique"])

    # --- Type d'option ---
    if st.session_state.option_class == "Vanille":
        st.session_state.option_type = st.selectbox("Type d'option", ["Call", "Put"])
    else:
        st.session_state.option_type = st.selectbox("Type d'option exotique", ["Asian", "Lookback"])

    # --- Paramètres financiers ---
    st.session_state.K = st.number_input("Strike (K)", value=st.session_state.S)
    st.session_state.T = st.number_input("Maturité (T, années)", value=1.0)
    st.session_state.r = st.number_input("Taux sans risque (r)", value=0.02)
    st.session_state.sigma = st.number_input("Volatilité (σ)", value=0.2)
    st.session_state.q = st.number_input("Dividende (q)", value=0.0)


    # Vérifie si les paramètres existent
    required_keys = ['option_type', 'option_class', 'S', 'K']
    if not all(k in st.session_state for k in required_keys):
        st.info("Veuillez d'abord définir vos paramètres dans l'onglet Accueil.")
        return

    S0 = st.session_state.S
    K = st.session_state.K
    option_type = st.session_state.option_type
    option_class = st.session_state.option_class
    buy_sell = st.session_state.option_buysell
    

    # Range de prix pour le payoff
    S_range = np.linspace(0.5*S0, 1.5*S0, 100)
    payoff = np.zeros_like(S_range)

    if option_class == "Vanille":
        if option_type == "Call":
            payoff = np.maximum(S_range - K, 0)
        elif option_type == "Put":
            payoff = np.maximum(K - S_range, 0)
    else:
        # Exotiques simples
        if option_type == "Asian":
            # Payoff approximé par moyenne des prix
            avg_price = (S_range + S0)/2
            if option_type == "Call":
                payoff = np.maximum(avg_price - K, 0)
            else:
                payoff = np.maximum(K - avg_price, 0)
        elif option_type == "Lookback":
            # Payoff max(S_t - K) ou max(K - S_t) sur tout le range
            if option_type == "Call":
                payoff = np.maximum(S_range - K, 0)  # approximation : payoff à chaque S
                # ligne horizontale au max pour simuler vrai lookback
                payoff = np.full_like(S_range, np.max(payoff))
            else:
                payoff = np.maximum(K - S_range, 0)
                payoff = np.full_like(S_range, np.max(payoff))


    # Si l'utilisateur vend l'option
    if buy_sell == "Sell":
        payoff = -payoff

    # Plot design
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor('black')  # fond de la figure
    ax.set_facecolor('black')         # fond des axes

    # Courbe du payoff
    ax.plot(S_range, payoff, label=f"{buy_sell} {option_type}", color='orange', linewidth=2)

    # Strike
    ax.axvline(K, color='red', linestyle='--', label="Strike K", linewidth=2)

    # Axes
    ax.spines['bottom'].set_color('orange')
    ax.spines['top'].set_color('orange')
    ax.spines['left'].set_color('orange')
    ax.spines['right'].set_color('orange')

    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')

    # Labels
    ax.set_xlabel("Prix du sous-jacent à maturité", color='orange', fontsize=12)
    ax.set_ylabel("Payoff", color='orange', fontsize=12)

    # Grille
    ax.grid(True, color='orange', linestyle='--', alpha=0.3)

    # Légende
    legend = ax.legend(facecolor='black', edgecolor='orange')
    for text in legend.get_texts():
        text.set_color('orange')

    st.pyplot(fig)
