from functions.data_function import show_data_page
import streamlit as st

def app():
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    show_data_page()
