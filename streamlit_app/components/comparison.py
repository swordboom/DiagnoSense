import streamlit as st

def render_comparison_page():
    st.title("Model Comparative Study")
    st.markdown("---")
    
    st.markdown("This dashboard presents key metrics and an overall comparative study of the two underlying neural network architectures trained during Phase 4.")
    
    st.subheader("Architectural Comparison")
    
    arch_data = [
        {"Feature": "Target Pipeline", "Model 1": "Health (Symptom to Disease)", "Model 2": "Medicine (Context to Side Effects)"},
        {"Feature": "Training Dataset", "Model 1": "health_dataset.csv (12,091 rows)", "Model 2": "medicine_dataset.csv (248,218 rows)"},
        {"Feature": "Input Feature", "Model 1": "TF-IDF Max 5,000 (symptom text)", "Model 2": "TF-IDF Max 10,000 (name, uses, class)"},
        {"Feature": "Network Architecture", "Model 1": "MLP 512-256 + BatchNorm + Dropout", "Model 2": "MLP 1024-512 + BatchNorm + Dropout"},
        {"Feature": "Output Space", "Model 1": "Mutually exclusive (1 of 16 classes)", "Model 2": "Independent multi-label (550 side effects)"},
        {"Feature": "Loss Function", "Model 1": "Weighted CrossEntropyLoss", "Model 2": "BCEWithLogitsLoss"},
        {"Feature": "Inference Threshold", "Model 1": "Argmax (softmax)", "Model 2": "P > 0.5 (sigmoid)"},
        {"Feature": "Key Challenge", "Model 1": "Extreme class imbalance", "Model 2": "High label cardinality and sparsity"}
    ]
    st.table(arch_data)
    
    st.subheader("Final Evaluation Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model 1: Disease Prediction**")
        st.metric(label="Test Accuracy", value="99.89%")
        st.metric(label="Macro F1-Score", value="99.97%")
        st.caption("Evaluated on 1,814 stratified test samples.")
        
    with col2:
        st.markdown("**Model 2: Medicine Side Effects Prediction**")
        st.metric(label="Micro F1-Score", value="94.79%")
        st.metric(label="Samples F1-Score", value="94.91%")
        st.metric(label="Macro F1-Score", value="89.44%")
        st.caption("Evaluated on 37,233 stratified test samples. 550 label classes.")
