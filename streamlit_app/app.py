import streamlit as st

# Set page config
st.set_page_config(
    page_title="DiagnoSense AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS to clear out unnecessary paddings and structure the app
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
    </style>
""", unsafe_allow_html=True)

# Import components
from components.home import render_home_page
from components.disease import render_disease_page
from components.medicine import render_medicine_page
from components.comparison import render_comparison_page
from components.reports import render_reports_page

def main():
    st.sidebar.title("Navigation")
    
    # Sidebar options
    page = st.sidebar.radio(
        "Select a module",
        ["Home", "Disease Prediction", "Medicine Side Effects Prediction", "Model Comparison", "Reports"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("DiagnoSense Internal Dashboard")
    st.sidebar.caption("Models Load via PyTorch @st.cache_resource")
    
    # Routing
    if page == "Home":
        render_home_page()
    elif page == "Disease Prediction":
        render_disease_page()
    elif page == "Medicine Side Effects Prediction":
        render_medicine_page()
    elif page == "Model Comparison":
        render_comparison_page()
    elif page == "Reports":
        render_reports_page()

if __name__ == "__main__":
    main()
