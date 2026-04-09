import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

from utils.load_models import (
    load_health_pipeline,
    load_medicine_pipeline,
    load_medicine_threshold_settings,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from phase2_cleaning import normalize_symptom, normalize_text
from phase4_training import DEVICE


def run_disease_prediction(symptoms_list: List[str]) -> pd.DataFrame:
    """Predict top diseases from a list of symptoms."""
    model, tfidf, encoder = load_health_pipeline()

    clean_symptoms = [normalize_symptom(s) for s in symptoms_list if pd.notna(s)]
    clean_symptoms = [s for s in clean_symptoms if s is not None]
    if not clean_symptoms:
        raise ValueError("No viable symptom text after cleaning.")

    symptoms_str = ", ".join(clean_symptoms)
    features = tfidf.transform([normalize_text(symptoms_str)])
    x_tensor = torch.from_numpy(features.toarray().astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = probs.argsort()[-5:][::-1]
    rows = [
        {
            "Disease": str(encoder.inverse_transform([idx])[0]),
            "Confidence": float(probs[idx]),
        }
        for idx in top_indices
    ]
    return pd.DataFrame(rows)


def run_medicine_side_effects_prediction(
    name: str,
    uses: str = "",
    chemical_class: str = "",
    therapeutic_class: str = "",
    action_class: str = "",
) -> pd.DataFrame:
    """Predict side effects using tuned threshold + fallback top-k."""
    model, tfidf, mlb = load_medicine_pipeline()
    settings = load_medicine_threshold_settings()
    threshold = settings["global_threshold"]
    fallback_top_k = settings["fallback_top_k"]

    parts = [name, uses, chemical_class, therapeutic_class, action_class]
    input_str = " ".join([p.lower().strip() for p in parts if p and p.strip()])
    lemmatized = normalize_text(input_str)
    if not lemmatized:
        raise ValueError("Not enough valid text in input.")

    features = tfidf.transform([lemmatized])
    x_tensor = torch.from_numpy(features.toarray().astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    labels = mlb.classes_
    selected = np.where(probs >= threshold)[0]

    if len(selected) == 0:
        selected = probs.argsort()[-fallback_top_k:][::-1]

    rows = [{"Side Effect": str(labels[idx]), "Probability": float(probs[idx])} for idx in selected]
    out = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    return out

