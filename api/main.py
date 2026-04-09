"""
PHASE 6: API Deployment using FastAPI
=====================================
Inference service for disease and side-effect prediction.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from phase2_cleaning import normalize_symptom, normalize_text
from phase4_training import DEVICE, HealthModel, MedicineModel


MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="DiagnoSense AI Inference API",
    description="ML service for disease prediction and medicine side-effect prediction.",
    version="2.0.0",
)


# ============================================================
# GLOBAL ARTIFACTS
# ============================================================
health_model: Optional[HealthModel] = None
health_tfidf = None
disease_encoder = None

medicine_model: Optional[MedicineModel] = None
medicine_tfidf = None
medicine_se_mlb = None
medicine_threshold = 0.5
medicine_fallback_top_k = 5


def _resolve_artifact(preferred: str, legacy: str) -> Path:
    preferred_path = MODELS_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return MODELS_DIR / legacy


def _load_threshold_settings() -> tuple[float, int]:
    path = MODELS_DIR / "medicine_threshold.json"
    if not path.exists():
        return 0.5, 5

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return float(payload.get("global_threshold", 0.5)), int(payload.get("fallback_top_k", 5))


@app.on_event("startup")
async def startup_event() -> None:
    global health_model, health_tfidf, disease_encoder
    global medicine_model, medicine_tfidf, medicine_se_mlb
    global medicine_threshold, medicine_fallback_top_k

    print(f"Loading artifacts on device: {DEVICE}")

    health_tfidf_path = _resolve_artifact("symptom_disease_tfidf.pkl", "health_tfidf.pkl")
    health_encoder_path = _resolve_artifact("symptom_disease_label_encoder.pkl", "disease_label_encoder.pkl")
    health_model_path = _resolve_artifact("symptom_disease_model.pth", "health_model.pth")

    with health_tfidf_path.open("rb") as handle:
        health_tfidf = pickle.load(handle)
    with health_encoder_path.open("rb") as handle:
        disease_encoder = pickle.load(handle)

    health_model = HealthModel(
        input_dim=len(health_tfidf.vocabulary_),
        num_classes=len(disease_encoder.classes_),
    ).to(DEVICE)
    health_model.load_state_dict(torch.load(health_model_path, map_location=DEVICE, weights_only=True))
    health_model.eval()

    with (MODELS_DIR / "medicine_tfidf.pkl").open("rb") as handle:
        medicine_tfidf = pickle.load(handle)
    with (MODELS_DIR / "medicine_se_mlb.pkl").open("rb") as handle:
        medicine_se_mlb = pickle.load(handle)

    medicine_model = MedicineModel(
        input_dim=len(medicine_tfidf.vocabulary_),
        num_labels=len(medicine_se_mlb.classes_),
    ).to(DEVICE)
    medicine_model.load_state_dict(
        torch.load(MODELS_DIR / "medicine_model.pth", map_location=DEVICE, weights_only=True)
    )
    medicine_model.eval()

    medicine_threshold, medicine_fallback_top_k = _load_threshold_settings()
    print(
        f"Loaded medicine inference settings: threshold={medicine_threshold:.2f}, "
        f"fallback_top_k={medicine_fallback_top_k}"
    )


# ============================================================
# SCHEMAS
# ============================================================
class SymptomInput(BaseModel):
    symptoms: List[str]


class DiseaseOutput(BaseModel):
    disease: str
    confidence: float
    all_probabilities: Dict[str, float]


class MedicineInput(BaseModel):
    name: str
    uses: Optional[str] = ""
    chemical_class: Optional[str] = ""
    therapeutic_class: Optional[str] = ""
    action_class: Optional[str] = ""


class MedicineOutput(BaseModel):
    side_effects: List[str]
    confidence_scores: Dict[str, float]


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def home() -> Dict[str, str]:
    return {"status": "ok", "message": "DiagnoSense API is running.", "device": str(DEVICE)}


@app.post("/predict/disease", response_model=DiseaseOutput)
def predict_disease(payload: SymptomInput) -> DiseaseOutput:
    if not payload.symptoms:
        raise HTTPException(status_code=400, detail="Empty symptoms list provided.")
    if health_model is None or health_tfidf is None or disease_encoder is None:
        raise HTTPException(status_code=503, detail="Health artifacts are not loaded.")

    clean_symptoms = [normalize_symptom(s) for s in payload.symptoms if pd.notna(s)]
    clean_symptoms = [s for s in clean_symptoms if s is not None]
    if not clean_symptoms:
        raise HTTPException(status_code=400, detail="No viable text after symptom cleaning.")

    text = normalize_text(", ".join(clean_symptoms))
    if not text:
        raise HTTPException(status_code=400, detail="No valid tokens after preprocessing.")

    tfidf_vec = health_tfidf.transform([text])
    x_tensor = torch.from_numpy(tfidf_vec.toarray().astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        logits = health_model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_disease = str(disease_encoder.inverse_transform([top_idx])[0])
    all_probs = {
        str(disease_encoder.inverse_transform([i])[0]): float(probs[i]) for i in range(len(probs))
    }

    return DiseaseOutput(disease=top_disease, confidence=float(probs[top_idx]), all_probabilities=all_probs)


@app.post("/predict/side_effects", response_model=MedicineOutput)
def predict_side_effects(payload: MedicineInput) -> MedicineOutput:
    if medicine_model is None or medicine_tfidf is None or medicine_se_mlb is None:
        raise HTTPException(status_code=503, detail="Medicine artifacts are not loaded.")

    parts = [
        payload.name,
        payload.uses or "",
        payload.chemical_class or "",
        payload.therapeutic_class or "",
        payload.action_class or "",
    ]
    text = normalize_text(" ".join([part.lower().strip() for part in parts if part and part.strip()]))
    if not text:
        raise HTTPException(status_code=400, detail="Not enough valid text in input payload.")

    tfidf_vec = medicine_tfidf.transform([text])
    x_tensor = torch.from_numpy(tfidf_vec.toarray().astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        logits = medicine_model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    labels = medicine_se_mlb.classes_
    selected = np.where(probs >= medicine_threshold)[0]
    if len(selected) == 0:
        selected = probs.argsort()[-medicine_fallback_top_k:][::-1]

    side_effects = [str(labels[idx]) for idx in selected]
    scores = {str(labels[idx]): float(probs[idx]) for idx in selected}

    return MedicineOutput(side_effects=side_effects, confidence_scores=scores)

