"""
PHASE 6: API Deployment using FastAPI
=======================================
DiagnoSense - ML Pipeline
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Since we need the model definitions to load weights:
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
from phase4_training import HealthModel, MedicineModel, DEVICE
from phase2_cleaning import normalize_text, normalize_symptom

# ============================================================
# CONFIG
# ============================================================
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(
    title="DiagnoSense AI Inference API",
    description="GPU-accelerated ML pipeline predicting Diseases from Symptoms, and Side-Effects from Drugs.",
    version="1.0.0"
)

# ============================================================
# ARTIFACT GLOBALS & LIFESPAN
# ============================================================
health_model = None
health_tfidf = None
disease_encoder = None

medicine_model = None
medicine_tfidf = None
medicine_se_mlb = None

@app.on_event("startup")
async def startup_event():
    global health_model, health_tfidf, disease_encoder
    global medicine_model, medicine_tfidf, medicine_se_mlb
    
    print(f"Loading ML Artifacts to {DEVICE}...")
    
    # 1. Load Health Model & Artifacts
    with open(os.path.join(MODELS_DIR, "health_tfidf.pkl"), "rb") as f:
        health_tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "disease_label_encoder.pkl"), "rb") as f:
        disease_encoder = pickle.load(f)
        
    num_classes = len(disease_encoder.classes_)
    health_model = HealthModel(input_dim=len(health_tfidf.vocabulary_), num_classes=num_classes).to(DEVICE)
    health_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "health_model.pth"), weights_only=True, map_location=DEVICE))
    health_model.eval()
    
    # 2. Load Medicine Model & Artifacts
    with open(os.path.join(MODELS_DIR, "medicine_tfidf.pkl"), "rb") as f:
        medicine_tfidf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "medicine_se_mlb.pkl"), "rb") as f:
        medicine_se_mlb = pickle.load(f)
        
    num_labels = len(medicine_se_mlb.classes_)
    medicine_model = MedicineModel(input_dim=len(medicine_tfidf.vocabulary_), num_labels=num_labels).to(DEVICE)
    medicine_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "medicine_model.pth"), weights_only=True, map_location=DEVICE))
    medicine_model.eval()

# ============================================================
# SCHEMAS
# ============================================================
class SymptomInput(BaseModel):
    symptoms: List[str]
    
class DiseaseOutput(BaseModel):
    disease: str
    confidence: float
    all_probabilities: dict

class MedicineInput(BaseModel):
    name: str
    uses: Optional[str] = ""
    chemical_class: Optional[str] = ""
    therapeutic_class: Optional[str] = ""
    action_class: Optional[str] = ""

class MedicineOutput(BaseModel):
    side_effects: List[str]
    confidence_scores: dict

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def home():
    return {"status": "ok", "message": "DiagnoSense API is running.", "device": str(DEVICE)}

@app.post("/predict/disease", response_model=DiseaseOutput)
def predict_disease(payload: SymptomInput):
    if not payload.symptoms or len(payload.symptoms) == 0:
        raise HTTPException(status_code=400, detail="Empty symptoms list provided.")
        
    # Preprocess
    clean_symptoms = [normalize_symptom(s) for s in payload.symptoms if pd.notna(s)]
    clean_symptoms = [s for s in clean_symptoms if s is not None]
    
    if not clean_symptoms:
        raise HTTPException(status_code=400, detail="No viable text after symptom cleaning.")
        
    symptoms_str = ', '.join(clean_symptoms)
    lemmatized = normalize_text(symptoms_str)
    
    # Extract features
    tfidf_vec = health_tfidf.transform([lemmatized])
    x_tensor = torch.FloatTensor(tfidf_vec.toarray()).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = health_model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    # Formatting
    top_idx = np.argmax(probs)
    top_disease = disease_encoder.inverse_transform([top_idx])[0]
    
    probs_dict = {
        disease_encoder.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }
    
    return DiseaseOutput(
        disease=top_disease,
        confidence=float(probs[top_idx]),
        all_probabilities=probs_dict
    )

@app.post("/predict/side_effects", response_model=MedicineOutput)
def predict_side_effects(payload: MedicineInput):
    # Preprocess
    parts = []
    if payload.name: parts.append(payload.name.lower())
    if payload.uses: parts.append(payload.uses.lower())
    if payload.chemical_class: parts.append(payload.chemical_class.lower())
    if payload.therapeutic_class: parts.append(payload.therapeutic_class.lower())
    if payload.action_class: parts.append(payload.action_class.lower())
    
    input_str = ' '.join(parts)
    lemmatized = normalize_text(input_str)
    
    if not lemmatized.strip():
        raise HTTPException(status_code=400, detail="Not enough valid text in input payload.")
        
    # Extract features
    tfidf_vec = medicine_tfidf.transform([lemmatized])
    x_tensor = torch.FloatTensor(tfidf_vec.toarray()).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = medicine_model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    # Formatting
    labels = medicine_se_mlb.classes_
    predicted_indices = np.where(probs > 0.5)[0]
    
    side_effects = [labels[i] for i in predicted_indices]
    
    # Filter all non-zero probs if requested, or just include those that triggered
    confidence_dict = {
        labels[i]: float(probs[i])
        for i in predicted_indices
    }
    
    # If none crossed threshold, maybe include top 3 with lower confidence
    if not side_effects:
        top_k = 3
        top_indices = probs.argsort()[-top_k:][::-1]
        confidence_dict = {
            labels[i]: float(probs[i])
            for i in top_indices
        }
        
    return MedicineOutput(
        side_effects=side_effects,
        confidence_scores=confidence_dict
    )
