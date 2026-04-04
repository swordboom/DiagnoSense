"""
PHASE 3: Feature Engineering & Train/Test Splitting
=====================================================
DiagnoSense - ML Pipeline
"""

import pandas as pd
import numpy as np
import os
import pickle
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# DATASET 1: Health Dataset (Stratified Split & TF-IDF)
# ============================================================
def process_health_dataset():
    print("=" * 70)
    print("  PHASE 3: Processing Dataset 1 (Health)")
    print("=" * 70)
    
    # 1. Load Data
    df = pd.read_csv(os.path.join(DATA_DIR, "health_cleaned.csv"))
    
    # Handle any potential NaNs in text before TF-IDF
    df['symptoms_clean'] = df['symptoms_clean'].fillna('')
    X = df['symptoms_clean'].values
    y = df['disease_encoded'].values
    
    # 2. Split (70% Train, 15% Val, 15% Test)
    # Using random split without stratification because some classes have < 2 samples in temp sets
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    
    # Val (15%) / Test (15%) -> equal split of temp
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.50, random_state=SEED
    )
    
    print(f"  Split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 3. TF-IDF Vectorization
    print("  Applying TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    print(f"  TF-IDF Features count: {X_train_tfidf.shape[1]}")
    
    # 4. Compute Class Weights due to class imbalance
    print("  Computing class weights for highly imbalanced data...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {c: w for c, w in zip(classes, weights)}
    
    # 5. Save Artifacts
    # Sparse matrices
    sp.save_npz(os.path.join(DATA_DIR, "health_X_train.npz"), X_train_tfidf)
    sp.save_npz(os.path.join(DATA_DIR, "health_X_val.npz"), X_val_tfidf)
    sp.save_npz(os.path.join(DATA_DIR, "health_X_test.npz"), X_test_tfidf)
    
    # Labels
    np.save(os.path.join(DATA_DIR, "health_y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "health_y_val.npy"), y_val)
    np.save(os.path.join(DATA_DIR, "health_y_test.npy"), y_test)
    
    # Models/Transformers
    with open(os.path.join(MODELS_DIR, "health_tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
        
    with open(os.path.join(MODELS_DIR, "health_class_weights.pkl"), "wb") as f:
        pickle.dump(class_weights_dict, f)
        
    print("  [DONE] Processing complete and files saved.")
    
    return {
        'train': (X_train_tfidf.shape, y_train.shape),
        'val': (X_val_tfidf.shape, y_val.shape),
        'test': (X_test_tfidf.shape, y_test.shape)
    }

# ============================================================
# DATASET 2: Medicine Dataset (Multi-label Split & TF-IDF)
# ============================================================
def process_medicine_dataset():
    print("\n" + "=" * 70)
    print("  PHASE 3: Processing Dataset 2 (Medicine Side Effects)")
    print("=" * 70)
    
    # 1. Load Data
    df = pd.read_csv(os.path.join(DATA_DIR, "medicine_cleaned.csv"))
    y_full = np.load(os.path.join(DATA_DIR, "medicine_labels.npy"))
    
    df['input_text_clean'] = df['input_text_clean'].fillna('')
    X = df['input_text_clean'].values
    
    # 2. Split (70% Train, 15% Val, 15% Test)
    # Using random split since sklearn train_test_split doesn't natively stratify multi-label 
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X, y_full, test_size=0.30, random_state=SEED
    )
    
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.50, random_state=SEED
    )
    
    print(f"  Split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # 3. TF-IDF Vectorization
    print("  Applying TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    print(f"  TF-IDF Features count: {X_train_tfidf.shape[1]}")
    
    # 4. Save Artifacts
    # Sparse matrices
    sp.save_npz(os.path.join(DATA_DIR, "medicine_X_train.npz"), X_train_tfidf)
    sp.save_npz(os.path.join(DATA_DIR, "medicine_X_val.npz"), X_val_tfidf)
    sp.save_npz(os.path.join(DATA_DIR, "medicine_X_test.npz"), X_test_tfidf)
    
    # Labels
    np.save(os.path.join(DATA_DIR, "medicine_y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "medicine_y_val.npy"), y_val)
    np.save(os.path.join(DATA_DIR, "medicine_y_test.npy"), y_test)
    
    # Models/Transformers
    with open(os.path.join(MODELS_DIR, "medicine_tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
        
    print("  [DONE] Processing complete and files saved.")
    
    return {
        'train': (X_train_tfidf.shape, y_train.shape),
        'val': (X_val_tfidf.shape, y_val.shape),
        'test': (X_test_tfidf.shape, y_test.shape)
    }

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    health_shapes = process_health_dataset()
    medicine_shapes = process_medicine_dataset()
    
    print("\n" + "=" * 70)
    print("  PHASE 3 — SUMMARY")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 1: Health (TF-IDF & Stratified Split)                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  Train             : X={str(health_shapes['train'][0]):<15} y={str(health_shapes['train'][1]):<15} │
  │  Val               : X={str(health_shapes['val'][0]):<15} y={str(health_shapes['val'][1]):<15} │
  │  Test              : X={str(health_shapes['test'][0]):<15} y={str(health_shapes['test'][1]):<15} │
  │  TF-IDF Vocab      : {str(health_shapes['train'][0][1]):<15}                              │
  │  Saved             : NPZ files, health_tfidf.pkl               │
  │                    : health_class_weights.pkl                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 2: Medicine (TF-IDF & Random Split)                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  Train             : X={str(medicine_shapes['train'][0]):<15} y={str(medicine_shapes['train'][1]):<15} │
  │  Val               : X={str(medicine_shapes['val'][0]):<15} y={str(medicine_shapes['val'][1]):<15} │
  │  Test              : X={str(medicine_shapes['test'][0]):<15} y={str(medicine_shapes['test'][1]):<15} │
  │  TF-IDF Vocab      : {str(medicine_shapes['train'][0][1]):<15}                              │
  │  Saved             : NPZ files, medicine_tfidf.pkl             │
  └─────────────────────────────────────────────────────────────────┘

  [DONE] PHASE 3 COMPLETE — Awaiting approval to proceed to Phase 4
""")
