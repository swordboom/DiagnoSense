"""
PHASE 2: Data Cleaning & Preprocessing
========================================
DiagnoSense - ML Pipeline
"""

import pandas as pd
import numpy as np
import re
import os
import csv
import pickle
import warnings
warnings.filterwarnings('ignore')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ============================================================
# CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HEALTH_PATH = os.path.join(DATA_DIR, "health_dataset.csv")
MEDICINE_PATH = os.path.join(DATA_DIR, "medicine_dataset.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ============================================================
# TEXT NORMALIZATION UTILITIES
# ============================================================
def normalize_text(text):
    """Lowercase, remove punctuation, lemmatize."""
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower().strip()
    # Remove parenthetical content like "(blue lips, tongue, nailbeds)"
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove punctuation except spaces and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)


def normalize_symptom(symptom):
    """Normalize a single symptom name."""
    if pd.isna(symptom) or str(symptom).strip() == '':
        return None
    s = str(symptom).lower().strip()
    # Remove content in parentheses
    s = re.sub(r'\([^)]*\)', '', s)
    # Remove special chars but keep spaces and hyphens
    s = re.sub(r'[^a-z0-9\s\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) < 2:
        return None
    return s


# ============================================================
# DATASET 1: Health Dataset Cleaning
# ============================================================
def clean_health_dataset():
    print("=" * 70)
    print("  CLEANING DATASET 1: health_dataset.csv")
    print("=" * 70)
    
    # Load with proper column handling
    max_cols = 0
    with open(HEALTH_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            max_cols = max(max_cols, len(row))
    
    col_names = ['Disease'] + [f'Symptom_{i}' for i in range(1, max_cols)]
    df = pd.read_csv(HEALTH_PATH, header=None, skiprows=1, names=col_names, on_bad_lines='warn')
    
    print(f"\n  BEFORE Cleaning:")
    print(f"    Shape: {df.shape}")
    print(f"    Diseases: {df['Disease'].nunique()}")
    print(f"    Duplicates: {df.duplicated().sum()}")
    print(f"    Sample Disease values: {df['Disease'].unique()[:5].tolist()}")
    
    # --- Step 1: Remove exact duplicates ---
    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\n  Step 1 - Remove duplicates: {before_dup} -> {len(df)} ({before_dup - len(df)} removed)")
    
    # --- Step 2: Normalize disease names ---
    df['Disease'] = df['Disease'].str.strip()
    print(f"  Step 2 - Disease names stripped")
    
    # --- Step 3: Normalize all symptom cells ---
    print(f"  Step 3 - Normalizing symptom text...")
    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]
    for col in symptom_cols:
        df[col] = df[col].apply(normalize_symptom)
    
    # --- Step 4: Create combined symptom text feature ---
    print(f"  Step 4 - Creating combined symptom text feature...")
    def combine_symptoms(row):
        symptoms = []
        for col in symptom_cols:
            val = row[col]
            if val is not None and pd.notna(val) and str(val).strip():
                symptoms.append(str(val).strip())
        # Remove duplicates while preserving order
        seen = set()
        unique_symptoms = []
        for s in symptoms:
            if s not in seen:
                seen.add(s)
                unique_symptoms.append(s)
        return ', '.join(unique_symptoms)
    
    df['symptoms_text'] = df.apply(combine_symptoms, axis=1)
    
    # Also create a lemmatized version for NLP
    df['symptoms_clean'] = df['symptoms_text'].apply(normalize_text)
    
    # --- Step 5: Remove rows with no symptoms ---
    before_empty = len(df)
    df = df[df['symptoms_text'].str.len() > 0].reset_index(drop=True)
    print(f"  Step 5 - Remove empty symptom rows: {before_empty} -> {len(df)} ({before_empty - len(df)} removed)")
    
    # --- Step 6: Create symptom count feature ---
    df['symptom_count'] = df['symptoms_text'].apply(lambda x: len(x.split(', ')) if x else 0)
    
    # --- Step 7: Encode disease labels ---
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['disease_encoded'] = le.fit_transform(df['Disease'])
    
    # Save label encoder
    models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'disease_label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # Print AFTER stats
    print(f"\n  AFTER Cleaning:")
    print(f"    Shape: {df.shape}")
    print(f"    Diseases: {df['Disease'].nunique()}")
    print(f"    Disease label mapping:")
    for label, disease in enumerate(le.classes_):
        count = (df['disease_encoded'] == label).sum()
        print(f"      {label:2d} -> {disease:45s} ({count:5d} samples)")
    
    print(f"\n    Symptom count stats:")
    print(f"      Min: {df['symptom_count'].min()}, Max: {df['symptom_count'].max()}, Mean: {df['symptom_count'].mean():.1f}")
    
    print(f"\n    Sample cleaned rows:")
    for i in range(3):
        print(f"      [{i}] Disease: {df['Disease'].iloc[i]}")
        print(f"          Symptoms: {df['symptoms_text'].iloc[i][:100]}...")
        print(f"          Clean:    {df['symptoms_clean'].iloc[i][:100]}...")
        print()
    
    # Keep essential columns for ML
    df_clean = df[['Disease', 'disease_encoded', 'symptoms_text', 'symptoms_clean', 'symptom_count']].copy()
    
    # Save cleaned dataset
    clean_path = os.path.join(DATA_DIR, "health_cleaned.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"  ✅ Saved cleaned dataset: {clean_path}")
    print(f"     Final shape: {df_clean.shape}")
    
    return df_clean


# ============================================================
# DATASET 2: Medicine Dataset Cleaning
# ============================================================
def clean_medicine_dataset():
    print("=" * 70)
    print("  CLEANING DATASET 2: medicine_dataset.csv")
    print("=" * 70)
    
    df = pd.read_csv(MEDICINE_PATH, on_bad_lines='warn')
    
    # We may have a huge number of rows.
    # The relevant columns are:
    # name, use0..use4, sideEffect0..sideEffect41, Chemical Class, Therapeutic Class, Action Class
    
    print(f"\n  BEFORE Cleaning:")
    print(f"    Shape: {df.shape}")
    
    # Extract side effects
    print("  Extracting side effects...")
    se_cols = [c for c in df.columns if c.startswith('sideEffect')]
    
    def get_side_effects(row):
        return [str(row[c]).strip().lower() for c in se_cols if pd.notna(row[c]) and str(row[c]).strip() != '']
        
    df['side_effects_list'] = df.apply(get_side_effects, axis=1)
    df['side_effects_count'] = df['side_effects_list'].apply(len)
    
    before_se = len(df)
    df = df[df['side_effects_count'] > 0].reset_index(drop=True)
    print(f"  Rows with side effects: {len(df)} / {before_se}")
    
    df['drug_name_clean'] = df['name'].fillna('').str.lower().str.strip()
    
    # Build multi-label targets
    print("  Building multi-label binary matrix...")
    all_side_effects = set()
    for se_list in df['side_effects_list']:
        all_side_effects.update(se_list)
        
    MIN_FREQUENCY = 50 # Filter noise for a 250k dataset
    se_counts = {}
    for se_list in df['side_effects_list']:
        for se in se_list:
            se_counts[se] = se_counts.get(se, 0) + 1
            
    frequent_se = sorted([se for se, c in se_counts.items() if c >= MIN_FREQUENCY])
    print(f"  Total unique side effects: {len(all_side_effects)}, Frequent (>={MIN_FREQUENCY}): {len(frequent_se)}")
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=frequent_se)
    
    df['side_effects_filtered'] = df['side_effects_list'].apply(
        lambda x: [se for se in x if se in set(frequent_se)]
    )
    y_matrix = mlb.fit_transform(df['side_effects_filtered'])
    
    models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'medicine_se_mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
        
    # Input Feature Synthesis
    print("  Synthesizing Input text feature...")
    use_cols = [c for c in df.columns if c.startswith('use')]
    
    def synthesize_input(row):
        parts = [str(row['drug_name_clean'])]
        uses = [str(row[c]) for c in use_cols if pd.notna(row[c]) and str(row[c]).strip() != '']
        parts.extend(uses)
        if pd.notna(row.get('Therapeutic Class')): parts.append(str(row['Therapeutic Class']))
        if pd.notna(row.get('Action Class')): parts.append(str(row['Action Class']))
        if pd.notna(row.get('Chemical Class')): parts.append(str(row['Chemical Class']))
        return " ".join(parts).lower()
        
    df['input_text'] = df.apply(synthesize_input, axis=1)
    df['input_text_clean'] = df['input_text'].apply(normalize_text)
    
    # Save output
    df_save = df[['name', 'drug_name_clean', 'input_text', 'input_text_clean', 'side_effects_list', 'side_effects_count']].copy()
    df_save['side_effects_clean'] = df_save['side_effects_list'].apply(lambda x: ", ".join(x))
    df_save = df_save.drop(columns=['side_effects_list'])
    
    clean_path = os.path.join(DATA_DIR, "medicine_cleaned.csv")
    df_save.to_csv(clean_path, index=False)
    
    y_path = os.path.join(DATA_DIR, "medicine_labels.npy")
    np.save(y_path, y_matrix)
    
    print(f"  ✅ Saved cleaned medicine dataset: {clean_path}")
    print(f"  ✅ Saved label matrix: {y_path} shape: {y_matrix.shape}")
    
    return df_save, y_matrix, frequent_se


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DiagnoSense — PHASE 2: Data Cleaning & Preprocessing")
    print("=" * 70 + "\n")
    
    # Clean Dataset 1
    df_health = clean_health_dataset()
    
    print()
    
    # Clean Dataset 2
    df_drug, y_matrix, se_labels = clean_medicine_dataset()
    
    # Final Summary
    print("\n" + "=" * 70)
    print("  PHASE 2 — SUMMARY")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 1: health_cleaned.csv                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  Rows              : {df_health.shape[0]:>10,}                              │
  │  Columns           : {df_health.shape[1]:>10}                              │
  │  Features          : symptoms_text, symptoms_clean              │
  │  Target            : disease_encoded ({df_health['disease_encoded'].nunique()} classes)             │
  │  Saved files       : health_cleaned.csv                        │
  │                     : disease_label_encoder.pkl                 │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 2: medicine_cleaned.csv                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  Rows              : {df_drug.shape[0]:>10,}                              │
  │  Columns           : {df_drug.shape[1]:>10}                              │
  │  Features          : input_text, input_text_clean               │
  │  Target labels     : {len(se_labels):>10} side effects                     │
  │  Label matrix      : {y_matrix.shape}                         │
  │  Avg labels/sample : {y_matrix.sum(axis=1).mean():>10.1f}                              │
  │  Saved files       : medicine_cleaned.csv                      │
  │                     : medicine_labels.npy                       │
  │                     : medicine_se_mlb.pkl                       │
  └─────────────────────────────────────────────────────────────────┘

  ✅ PHASE 2 COMPLETE — Awaiting approval to proceed to Phase 3
""")
