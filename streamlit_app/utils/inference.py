import torch
import numpy as np
import pandas as pd
from utils.load_models import load_health_pipeline, load_medicine_pipeline
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
from phase4_training import DEVICE
from phase2_cleaning import normalize_text, normalize_symptom

def run_disease_prediction(symptoms_list):
    """Predict top diseases given a list of symptoms."""
    model, tfidf, encoder = load_health_pipeline()
    
    clean_symptoms = [normalize_symptom(s) for s in symptoms_list if pd.notna(s)]
    clean_symptoms = [s for s in clean_symptoms if s is not None]
    
    if not clean_symptoms:
        raise ValueError("No viable text after symptom cleaning.")
        
    symptoms_str = ', '.join(clean_symptoms)
    lemmatized = normalize_text(symptoms_str)
    
    tfidf_vec = tfidf.transform([lemmatized])
    x_tensor = torch.FloatTensor(tfidf_vec.toarray()).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    # Get top 5 predictions for the UI
    top_indices = probs.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        disease = encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        results.append({"Disease": disease, "Confidence": confidence})
        
    return pd.DataFrame(results)

def run_medicine_side_effects_prediction(name, uses="", chemical_class="", therapeutic_class="", action_class=""):
    """Predict side effects for a given medicine."""
    model, tfidf, mlb = load_medicine_pipeline()
    
    parts = []
    if name: parts.append(name.lower())
    if uses: parts.append(uses.lower())
    if chemical_class: parts.append(chemical_class.lower())
    if therapeutic_class: parts.append(therapeutic_class.lower())
    if action_class: parts.append(action_class.lower())
    
    input_str = ' '.join(parts)
    lemmatized = normalize_text(input_str)
    
    if not lemmatized.strip():
        raise ValueError("Not enough valid text in input.")
        
    tfidf_vec = tfidf.transform([lemmatized])
    x_tensor = torch.FloatTensor(tfidf_vec.toarray()).to(DEVICE)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    labels = mlb.classes_
    predicted_indices = np.where(probs > 0.5)[0]
    
    results = []
    if len(predicted_indices) > 0:
        for idx in predicted_indices:
            results.append({"Side Effect": labels[idx], "Probability": float(probs[idx])})
    else:
        # If none cross 0.5 threshold, return top 3
        top_indices = probs.argsort()[-3:][::-1]
        for idx in top_indices:
            results.append({"Side Effect": labels[idx], "Probability": float(probs[idx])})
            
    df = pd.DataFrame(results)
    # Sort by probability descending
    df = df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    return df
