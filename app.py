import streamlit as st


# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="ML Explorer Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("🤖 Machine Learning Explorer Dashboard")
st.markdown("## Interactive Introduction to ML Paradigms")

# Introduction text
st.markdown("""
            Welcome! this dashboard demonstrates three types of machine learning:
            1. **Supervised Learning** - Learning from labeled examples
            2. **Unsupervised Learning** - Finding patterns in unlabeled data
            3. **Reinforcement Learning** - Learning through trial and error

           👈 **Select a page from the sidebar** to explore each type!
            """)

# Create three columns for information cards
col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
            **📊 Supervised Learning**

            - Classification (Iris dataset)
            - Regression (Housing prices)
            - Train/test split demo
            """)

with col2:
    st.info("""
            **🔍 Unsupervised Learning**

            - K-Means clustering
            - PCA dimensionality reduction
            - Pattern discovery
            """)

with col3:
    st.info("""
            **🎮 Reinforcement Learning**

            - Q-learning grid world
            - Agent training
            - Reward-based learning
            """)

# Footer

st.markdown("---")
st.markdown("**Built with:** Streamlit + scikit-learn + Plotly")
