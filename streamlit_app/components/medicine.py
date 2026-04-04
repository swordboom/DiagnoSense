import streamlit as st
import pandas as pd
from utils.inference import run_medicine_side_effects_prediction

def render_medicine_page():
    st.title("Medicine Side Effects Prediction")
    st.markdown("---")
    st.markdown("Input a medicine name and associated context parameters to predict potential side effects. This model utilizes a Multi-Label PyTorch architecture with BCEWithLogits, trained on over 248,000 unique records.")
    
    with st.form("medicine_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Medicine Name (Required)", placeholder="e.g. Paracetamol")
            uses = st.text_input("Uses / Medical Condition (Optional)", placeholder="e.g. Fever, Headache")
            chemical_class = st.text_input("Chemical Class (Optional)", placeholder="")
        with col2:
            therapeutic_class = st.text_input("Therapeutic Class (Optional)", placeholder="")
            action_class = st.text_input("Action Class (Optional)", placeholder="")
            
        submitted = st.form_submit_button("Predict Side Effects")
        
    if submitted:
        if not name.strip() and not uses.strip():
            st.error("Please enter a Medicine Name or Uses.")
            return
            
        with st.spinner("Running ML Inference Pipeline..."):
            try:
                results_df = run_medicine_side_effects_prediction(
                    name=name,
                    uses=uses,
                    chemical_class=chemical_class,
                    therapeutic_class=therapeutic_class,
                    action_class=action_class
                )
                
                st.success("Prediction complete.")
                st.subheader("Predicted Side Effects")
                
                results_df["Probability"] = (results_df["Probability"] * 100).map("{:.2f}%".format)
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
            except ValueError as e:
                st.error(f"Inference Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
