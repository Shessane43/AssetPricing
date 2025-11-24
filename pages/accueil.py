import streamlit as st

def app():
    st.title("📈 Asset Pricing & Option Greeks")
    st.subheader("Bienvenue dans l'application")

    st.write("""
    Cette application permet de :
    - Calculer le prix d'options européennes (call et put) via le modèle de Black-Scholes.
    - Visualiser les Greeks (Delta, Gamma, Vega, Theta, Rho) associés aux options.
    - Tracer des graphiques des Greeks et des prix selon différents paramètres.
    - Consulter des données de marché (spots, taux, volatilité implicite).
    """)

    st.markdown("---")
    st.write("👉 Commencez par l'onglet **Pricing** pour calculer le prix d'une option, puis explorez les autres onglets pour voir les Greeks et les graphiques.")
