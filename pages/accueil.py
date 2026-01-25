import streamlit as st

def app():
    st.markdown(
        """
        <style>
        .title {
            color: white;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0px; /* reduced space under the title */
            margin-top: 5px;    /* space above the title */
        }
        hr {
            margin-top: 5px;    /* reduced space between title and line */
            margin-bottom: 10px; /* space below the line before the cards */
        }
        .subtitle {
            color: white;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .card {
            background-color: #FF6600;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 4px 4px 15px rgba(0,0,0,0.15);
            margin: 5px 0px;
        }
        .card p {
            color: white;
            margin: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Horizontal line with reduced space
    st.markdown("<hr>", unsafe_allow_html=True)

    # Feature columns
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="subtitle">Option Simulation</h3>', unsafe_allow_html=True)
        st.markdown('<p>Select a ticker, set the option parameters, and visualize the payoff.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="subtitle">Market Analysis</h3>', unsafe_allow_html=True)
        st.markdown('<p>Access market data, stock prices, greeks, and implied volatility.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="subtitle">Portfolio</h3>', unsafe_allow_html=True)
        st.markdown('<p>Add your positions and track total value and associated risks.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Footer or instructions
    st.info("Select a page from the navigation menu to get started.")
