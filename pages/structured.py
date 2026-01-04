import streamlit as st
from functions.structured_function import straddle_price, strangle_price, bull_spread_price, bear_spread_price,plot_structured_payoff, butterfly_spread_price, collar_price

def app():

    # --- Paramètres de base ---
    required_keys = ["S", "r", "q", "T"]
    if not all(k in st.session_state for k in required_keys):
        st.error("Paramètres manquants. Retournez à l'accueil.")
        return

    S = st.session_state["S"]
    r = st.session_state["r"]
    q = st.session_state["q"]
    T = st.session_state["T"]

    # --- Sélection du produit structuré ---
    st.subheader("Choisir le type de produit structuré")
    product_type = st.selectbox(
        "Produit structuré",
        ["Straddle", "Strangle", "Bull Spread", "Bear Spread","Butterfly Spread", "Collar"]
    )

    # --- Paramètres selon produit ---
    if product_type == "Straddle":
        st.write("Long Call + Short Put sur le même strike et maturité. Sert à parier sur une forte volatilité indépendamment de la direction du marché.")
        K = st.number_input("Strike", value=S)
        sigma = st.number_input("Volatilité", value=0.2)

    elif product_type == "Strangle":
        st.write("Long Call + Short Put avec strikes différents. Sert à profiter d’un mouvement important du sous-jacent tout en coûtant moins qu’un straddle.")
        K_put = st.number_input("Strike Put", value=S*0.95)
        K_call = st.number_input("Strike Call", value=S*1.05)
        sigma_put = st.number_input("Volatilité Put", value=0.2)
        sigma_call = st.number_input("Volatilité Call", value=0.2)
    elif product_type == "Bull Spread":
        st.write("Long Call strike bas + Short Call strike haut. Sert à parier sur une hausse modérée du sous-jacent avec risque limité.")
        K_low = st.number_input("Strike Long Call", value=S*0.95)
        K_high = st.number_input("Strike Short Call", value=S*1.05)
        sigma = st.number_input("Volatilité", value=0.2)

    elif product_type == "Bear Spread":
        st.write("Long Put strike haut + Short Put strike bas. Sert à parier sur une baisse modérée du sous-jacent avec risque limité.")
        K_high = st.number_input("Strike Long Put", value=S*1.05)
        K_low = st.number_input("Strike Short Put", value=S*0.95)
        sigma = st.number_input("Volatilité", value=0.2)

    elif product_type == "Butterfly Spread":
        st.write("Long Call strike bas + Short 2 Calls strike milieu + Long Call strike haut. Sert à parier sur un faible mouvement autour du strike médian.")
        K_low = st.number_input("Strike Low", value=S*0.95)
        K_mid = st.number_input("Strike Middle", value=S)
        K_high = st.number_input("Strike High", value=S*1.05)
        sigma = st.number_input("Volatilité", value=0.2)
        
    elif product_type == "Collar":
        st.write("Long Put strike bas + Short Call strike haut. Sert à limiter la perte tout en plafonnant le gain.")
        K_put = st.number_input("Strike Put (protection)", value=S*0.95)
        K_call = st.number_input("Strike Call (cap)", value=S*1.05)
        sigma_put = st.number_input("Volatilité Put", value=0.2)
        sigma_call = st.number_input("Volatilité Call", value=0.2)
        

    # --- Calcul du prix ---
    if st.button("Calculer le prix du produit structuré"):
        try:
            if product_type == "Straddle":
                total_price = straddle_price(S, K, T, r, q, sigma)
            elif product_type == "Strangle":
                total_price = strangle_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call)
            elif product_type == "Bull Spread":
                total_price = bull_spread_price(S, K_low, K_high, T, r, q, sigma)
            elif product_type == "Bear Spread":
                total_price = bear_spread_price(S, K_high, K_low, T, r, q, sigma)
            elif product_type == "Butterfly Spread":
                total_price = butterfly_spread_price(S, K_low, K_mid, K_high, T, r, q, sigma)
            elif product_type == "Collar":
                st.info(
                "ATTENTION : Le prix d’un collar peut être négatif si la prime du Call vendu est supérieure "
                "à la prime du Put acheté. Cela reflète que vous recevez de l’argent net pour créer la position.")
                total_price = collar_price(S, K_put, K_call, T, r, q, sigma_put, sigma_call)

            st.success(f" Prix total du {product_type} : **{total_price:.4f}**")
        except Exception as e:
            st.error(f"Erreur : {e}")


    if st.button("Afficher le payoff"):
        # Re-créer la liste products comme dans le calcul du prix
        products = []
        if product_type == "Straddle":
            products.append({"option_type":"Call", "K":K, "weight":1})
            products.append({"option_type":"Put", "K":K, "weight":1})
        elif product_type == "Strangle":
            products.append({"option_type":"Call", "K":K_call, "weight":1})
            products.append({"option_type":"Put", "K":K_put, "weight":1})
        elif product_type == "Bull Spread":
            products.append({"option_type":"Call", "K":K_low, "weight":1})
            products.append({"option_type":"Call", "K":K_high, "weight":-1})
        elif product_type == "Bear Spread":
            products.append({"option_type":"Put", "K":K_high, "weight":1})
            products.append({"option_type":"Put", "K":K_low, "weight":-1})
        elif product_type == "Butterfly Spread":
            products.append({"option_type":"Call", "K":K_low, "weight":1, "sigma":sigma})
            products.append({"option_type":"Call", "K":K_mid, "weight":-2, "sigma":sigma})
            products.append({"option_type":"Call", "K":K_high, "weight":1, "sigma":sigma})
        elif product_type == "Collar":
            products.append({"option_type":"Put", "K":K_put, "weight":1, "sigma":sigma_put})
            products.append({"option_type":"Call", "K":K_call, "weight":-1, "sigma":sigma_call})

        fig = plot_structured_payoff(products, S)
        st.pyplot(fig)