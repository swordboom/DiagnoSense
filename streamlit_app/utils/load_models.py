import os
import pickle
import torch
import streamlit as st

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from phase4_training import HealthModel, MedicineModel, DEVICE

MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_health_pipeline():
    """Load and cache the Health Model and its artifacts."""
    with open(os.path.join(MODELS_DIR, "health_tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "disease_label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
        
    num_classes = len(encoder.classes_)
    model = HealthModel(input_dim=len(tfidf.vocabulary_), num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "health_model.pth"), weights_only=True, map_location=DEVICE))
    model.eval()
    
    return model, tfidf, encoder

@st.cache_resource
def load_medicine_pipeline():
    """Load and cache the Medicine Model and its artifacts."""
    with open(os.path.join(MODELS_DIR, "medicine_tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "medicine_se_mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)
        
    num_labels = len(mlb.classes_)
    model = MedicineModel(input_dim=len(tfidf.vocabulary_), num_labels=num_labels).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "medicine_model.pth"), weights_only=True, map_location=DEVICE))
    model.eval()
    
    return model, tfidf, mlb
