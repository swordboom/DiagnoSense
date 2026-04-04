import streamlit as st
import pandas as pd
from utils.inference import run_disease_prediction

def render_disease_page():
    st.title("Disease Prediction (Symptom → Disease)")
    st.markdown("---")
    st.markdown("Input a list of patient symptoms to predict the underlying disease. The model uses a Multi-Class PyTorch architecture trained on a highly imbalanced subset of 16 common diseases.")
    
    with st.form("disease_form"):
        symptoms_input = st.text_area(
            "Symptoms (Comma-separated)", 
            placeholder="e.g. runny nose, sneezing, sore throat, fever"
        )
        submitted = st.form_submit_button("Predict Disease")
        
    if submitted:
        if not symptoms_input.strip():
            st.error("Please enter at least one symptom.")
            return
            
        symptoms_list = [s.strip() for s in symptoms_input.split(',')]
        
        with st.spinner("Running ML Inference Pipeline..."):
            try:
                results_df = run_disease_prediction(symptoms_list)
                
                st.success("Prediction complete.")
                st.subheader("Top Predictions")
                
                # Format the confidence score to percentage for readability
                results_df["Confidence ( % )"] = (results_df["Confidence"] * 100).map("{:.2f}%".format)
                results_df = results_df.drop(columns=["Confidence"])
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
            except ValueError as e:
                st.error(f"Inference Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
