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
    1. **Disease Prediction** - Predicts an underlying condition out of 20 disease classes based on symptom patterns. Trained using `symptom_disease_dataset.csv` with explicit splits (`symptom_disease_train/val/test.csv`).
    2. **Medicine Side Effects Prediction** - Given a medicine name and its therapeutic context, simultaneously infers multiple potential side effects across 550 possible labels. Trained on `medicine_dataset.csv` (248,000+ records).
    """)
    
    st.subheader("Instructions")
    st.markdown("""
    - Use the **Sidebar Navigation** to switch between modules.
    - In **Disease Prediction**, enter a comma-separated list of patient symptoms to receive a ranked list of probable conditions.
    - In **Medicine Side Effects Prediction**, enter a medicine name and optional context (uses, chemical class, therapeutic class) to infer probable side effects.
    - Consult **Model Comparison** and **Reports** for architectural details and test-set evaluation metrics.
    """)
