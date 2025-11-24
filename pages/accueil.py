import streamlit as st
import yfinance as yf

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

    # --- Classe d'option ---
    st.session_state.option_class = st.selectbox("Classe d'option", ["Vanille", "Exotique"])

    # --- Type d'option ---
    if st.session_state.option_class == "Vanille":
        st.session_state.option_type = st.selectbox("Type d'option", ["Call", "Put"])
    else:
        st.session_state.option_type = st.selectbox("Type d'option exotique", ["Asian", "Barrier", "Lookback"])

    # --- Paramètres financiers ---
    st.session_state.K = st.number_input("Strike (K)", value=st.session_state.S)
    st.session_state.T = st.number_input("Maturité (T, années)", value=1.0)
    st.session_state.r = st.number_input("Taux sans risque (r)", value=0.02)
    st.session_state.sigma = st.number_input("Volatilité (σ)", value=0.2)
    st.session_state.q = st.number_input("Dividende (q)", value=0.0)
