import streamlit as st

def render_home_page():
    st.title("DiagnoSense Analytics")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to DiagnoSense
    
    This interface serves as a unified prediction portal and analytical dashboard for the **DiagnoSense Machine Learning Pipeline**. 
    
    The underlying architecture relies on PyTorch-accelerated Deep Neural Networks trained to execute two parallel healthcare classification tasks.
    """)
    
    st.subheader("Model Overview")
    st.markdown("""
    1. **Disease Prediction**: Predicts an underlying condition out of 16 highly imbalanced target classes based on a textual profile of comma-separated patient symptoms.
    2. **Drug Side Effects Prediction**: Given a drug and its therapeutic context, simultaneously infers multiple potential side effects out of an independent 154-label space.
    """)
    
    st.subheader("Instructions")
    st.markdown("""
    - Use the **Sidebar Navigation** to switch between inference interfaces and evaluation reports.
    - Enter descriptive symptoms in the Disease Prediction tab separated by commas.
    - Provide contextual information along with a precise drug name in the Side Effects tab to narrow down probability estimations.
    - Consult the **Model Comparison** and **Reports** pages for an in-depth look into the architectural trade-offs, Test Set evaluations, and exact execution details of the pipelines.
    """)
