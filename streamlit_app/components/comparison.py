import streamlit as st

def render_comparison_page():
    st.title("Model Comparative Study")
    st.markdown("---")
    
    st.markdown("This dashboard presents key metrics and an overall comparative study of the two underlying neural network architectures trained during Phase 4.")
    
    st.subheader("Architectural Comparison")
    
    arch_data = [
        {"Feature": "Target Pipeline", "Model 1": "Health (Symptom → Disease)", "Model 2": "Drug (Context → Side Effects)"},
        {"Feature": "Input Dynamics", "Model 1": "TF-IDF Max 5000 (Symptom Combos)", "Model 2": "TF-IDF Max 10000 (Drug/Condition Context)"},
        {"Feature": "Network Strategy", "Model 1": "MLP (512 → 256) + BatchNorm + Dropout", "Model 2": "MLP (1024 → 512) + BatchNorm + Dropout"},
        {"Feature": "Output Space", "Model 1": "Mutually Exclusive (1 of 16 Classes)", "Model 2": "Independent Multi-Label (Subset of 154)"},
        {"Feature": "Loss Function", "Model 1": "Weighted CrossEntropyLoss", "Model 2": "BCEWithLogitsLoss"},
        {"Feature": "Structural Challenge", "Model 1": "Extreme Class Imbalance", "Model 2": "High Label Cardinality and Sparsity"}
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
        st.markdown("**Model 2: Side Effects Prediction**")
        st.metric(label="Micro F1-Score", value="64.81%")
        st.metric(label="Samples F1-Score", value="62.99%")
        st.caption("Evaluated on 420 random test samples.")
