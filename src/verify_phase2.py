from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

print("=== Verifying Phase 2 Output ===\n")

# Symptom-disease splits
train_path = DATA_DIR / "symptom_disease_train.csv"
val_path = DATA_DIR / "symptom_disease_val.csv"
test_path = DATA_DIR / "symptom_disease_test.csv"
health_full_path = DATA_DIR / "symptom_disease_dataset.csv"

if train_path.exists() and val_path.exists() and test_path.exists():
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    full = pd.read_csv(health_full_path) if health_full_path.exists() else pd.concat([train, val, test], ignore_index=True)

    disease_col = next(col for col in train.columns if str(col).strip().lower() == "disease")
    symptom_cols = [col for col in train.columns if col != disease_col]

    print("--- Symptom-Disease Splits ---")
    print(f"Train shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Full shape: {full.shape}")
    print(f"Disease classes: {full[disease_col].nunique()}")
    print(f"Symptom features: {len(symptom_cols)}")

    print("\nTop disease distribution:")
    print(full[disease_col].value_counts().head(10).to_string())
else:
    print("--- Symptom-Disease Splits ---")
    print("Split files not found. Expected:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

# Medicine dataset
med = pd.read_csv(DATA_DIR / "medicine_cleaned.csv")
labels = np.load(DATA_DIR / "medicine_labels.npy")
print("\n--- Medicine Dataset (Cleaned) ---")
print(f"Shape: {med.shape}")
print(f"Columns: {list(med.columns)}")
print(f"Label matrix: {labels.shape}")
print(f"Avg labels/sample: {labels.sum(axis=1).mean():.2f}")
print(f"Label density: {labels.mean() * 100:.2f}%")

# Model files check
artifacts = [
    "symptom_disease_label_encoder.pkl",
    "medicine_se_mlb.pkl",
]
for fname in artifacts:
    path = MODELS_DIR / fname
    status = "OK" if path.exists() else "MISSING"
    size = path.stat().st_size if path.exists() else 0
    print(f"  [{status}] {fname} ({size} bytes)")

print("\n=== Phase 2 Verification Complete ===")

