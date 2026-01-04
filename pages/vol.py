import streamlit as st
from functions.vol_function import (
    get_market_prices_yahoo,
    generate_vol_curve,
    plot_vol_curve,
    get_all_option_maturities,
    generate_vol_curves_multiple_maturities,
    plot_vol_surface
)

def app():

    # ---------- 1. Récupérer les paramètres depuis la session ----------
    ticker = st.session_state.get("ticker")
    S = st.session_state.get("S")
    K = st.session_state.get("K")
    T = st.session_state.get("T")
    r = st.session_state.get("r")
    q = st.session_state.get("q")
    option_type = st.session_state.get("option_type")

    with st.container(border=True):
        st.markdown(
            f"""
            **Ticker** : **{ticker}**

            **Spot (S)** : {S:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Strike (K)** : {K:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Maturité (T)** : {T} an(s)

            **Taux sans risque (r)** : {r:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Dividendes (q)** : {q:.2%} &nbsp;&nbsp;|&nbsp;&nbsp;
            **Type d'option** : **{option_type}**
            """
        )

    market_prices_calls, market_prices_puts, maturity = get_market_prices_yahoo(ticker, T_days=int(T*365))
    if not market_prices_calls and not market_prices_puts:
        st.error("Impossible de récupérer les prix de marché pour ce ticker/maturité. Vérifiez le ticker ou la connexion.")
        return

    # ici c'est important de prendre la maturité du vraie produit trouvé sur le marché
    strikes, vols = generate_vol_curve(S, maturity, r, q, market_prices_calls, market_prices_puts, option_type)

    fig = plot_vol_curve(
        strikes,
        vols,
        maturity=maturity,
        K=K,
    )
    st.subheader("Courbe de volatilité implicite")
    st.pyplot(fig)

    all_matu = get_all_option_maturities(ticker)
    vol_curves = generate_vol_curves_multiple_maturities(S, all_matu, r, q, option_type, ticker)
    fig_3d = plot_vol_surface(vol_curves)
    st.subheader("Surface de volatilité implicite")

    st.plotly_chart(fig_3d, width='stretch')