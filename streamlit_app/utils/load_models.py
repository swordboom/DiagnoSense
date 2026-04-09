import json
import pickle
import sys
from pathlib import Path

import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "src"))

from phase4_training import DEVICE, HealthModel, MedicineModel

MODELS_DIR = BASE_DIR / "models"


def _resolve_artifact(preferred: str, legacy: str) -> Path:
    preferred_path = MODELS_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return MODELS_DIR / legacy


@st.cache_resource
def load_health_pipeline():
    """Load and cache health model + artifacts."""
    tfidf_path = _resolve_artifact("symptom_disease_tfidf.pkl", "health_tfidf.pkl")
    encoder_path = _resolve_artifact("symptom_disease_label_encoder.pkl", "disease_label_encoder.pkl")
    model_path = _resolve_artifact("symptom_disease_model.pth", "health_model.pth")

    with tfidf_path.open("rb") as handle:
        tfidf = pickle.load(handle)
    with encoder_path.open("rb") as handle:
        encoder = pickle.load(handle)

    model = HealthModel(input_dim=len(tfidf.vocabulary_), num_classes=len(encoder.classes_)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    return model, tfidf, encoder


@st.cache_resource
def load_medicine_pipeline():
    """Load and cache medicine model + artifacts."""
    with (MODELS_DIR / "medicine_tfidf.pkl").open("rb") as handle:
        tfidf = pickle.load(handle)
    with (MODELS_DIR / "medicine_se_mlb.pkl").open("rb") as handle:
        mlb = pickle.load(handle)

    model = MedicineModel(input_dim=len(tfidf.vocabulary_), num_labels=len(mlb.classes_)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODELS_DIR / "medicine_model.pth", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    return model, tfidf, mlb


@st.cache_data
def load_medicine_threshold_settings():
    """Load tuned threshold and fallback behavior for medicine inference."""
    default = {"global_threshold": 0.5, "fallback_top_k": 5}
    path = MODELS_DIR / "medicine_threshold.json"
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "global_threshold": float(payload.get("global_threshold", default["global_threshold"])),
        "fallback_top_k": int(payload.get("fallback_top_k", default["fallback_top_k"])),
    }

