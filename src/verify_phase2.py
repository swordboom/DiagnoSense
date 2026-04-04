import pandas as pd
import numpy as np
import os

data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

print("=== Verifying Phase 2 Output ===\n")

# Health dataset
h = pd.read_csv(os.path.join(data, "health_cleaned.csv"))
print("--- Health Dataset (Cleaned) ---")
print(f"Shape: {h.shape}")
print(f"Columns: {list(h.columns)}")
n_diseases = h["disease_encoded"].nunique()
print(f"Diseases: {n_diseases}")
print(f"\nDisease distribution:")
for disease, count in h["Disease"].value_counts().items():
    pct = count / len(h) * 100
    bar = "#" * int(pct / 2)
    print(f"  {disease:50s}: {count:5d} ({pct:5.1f}%) {bar}")

print(f"\nSample rows:")
for i in range(3):
    print(f"  [{i}] {h['Disease'].iloc[i]} | symptoms: {h['symptoms_text'].iloc[i][:70]}...")

print(f"\n--- Drug Side Effects (Cleaned) ---")
d = pd.read_csv(os.path.join(data, "drug_side_effects_cleaned.csv"))
print(f"Shape: {d.shape}")
print(f"Columns: {list(d.columns)}")

y = np.load(os.path.join(data, "drug_side_effects_labels.npy"))
print(f"Label matrix: {y.shape}")
print(f"Avg labels/sample: {y.sum(axis=1).mean():.1f}")
print(f"Label density: {y.sum() / y.size * 100:.2f}%")

print(f"\nSample rows:")
for i in range(3):
    print(f"  [{i}] Drug: {d['drug_name'].iloc[i]} | Input: {d['input_text'].iloc[i][:60]}...")
    print(f"       SE: {d['side_effects_clean'].iloc[i][:60]}...")

# Check model files
models = os.path.join(os.path.dirname(data), "models")
for fname in ["disease_label_encoder.pkl", "side_effects_mlb.pkl"]:
    fpath = os.path.join(models, fname)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"\n  [{status}] {fname} ({size} bytes)")

print("\n=== Phase 2 Verification Complete ===")
