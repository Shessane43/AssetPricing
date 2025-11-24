import streamlit as st

st.title("Application Streamlit avec Onglets")

# Création des onglets
tabs = st.tabs(["Accueil", "Pricing", "Greeks", "Plots", "Market Data"])

with tabs[0]:
    st.header("Accueil")
    st.write("Contenu de l'onglet Accueil")

with tabs[1]:
    st.header("Pricing")
    st.write("Contenu de l'onglet Pricing")

with tabs[2]:
    st.header("Greeks")
    st.write("Contenu de l'onglet Greeks")

with tabs[3]:
    st.header("Plots")
    st.write("Contenu de l'onglet Plots")

with tabs[4]:
    st.header("Market Data")
    st.write("Contenu de l'onglet Market Data")