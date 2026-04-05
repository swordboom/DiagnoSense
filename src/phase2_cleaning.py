# -*- coding: utf-8 -*-
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
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)


def normalize_symptom(symptom):
    """Normalize a single symptom name."""
    if pd.isna(symptom) or str(symptom).strip() == '':
        return None
    s = str(symptom).lower().strip()
    s = re.sub(r'\([^)]*\)', '', s)
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

    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\n  Step 1 - Remove duplicates: {before_dup} -> {len(df)} ({before_dup - len(df)} removed)")

    df['Disease'] = df['Disease'].str.strip()
    print(f"  Step 2 - Disease names stripped")

    print(f"  Step 3 - Normalizing symptom text...")
    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]
    for col in symptom_cols:
        df[col] = df[col].apply(normalize_symptom)

    print(f"  Step 4 - Creating combined symptom text feature...")
    def combine_symptoms(row):
        symptoms = []
        for col in symptom_cols:
            val = row[col]
            if val is not None and pd.notna(val) and str(val).strip():
                symptoms.append(str(val).strip())
        seen = set()
        unique_symptoms = []
        for s in symptoms:
            if s not in seen:
                seen.add(s)
                unique_symptoms.append(s)
        return ', '.join(unique_symptoms)

    df['symptoms_text'] = df.apply(combine_symptoms, axis=1)
    df['symptoms_clean'] = df['symptoms_text'].apply(normalize_text)

    before_empty = len(df)
    df = df[df['symptoms_text'].str.len() > 0].reset_index(drop=True)
    print(f"  Step 5 - Remove empty symptom rows: {before_empty} -> {len(df)} ({before_empty - len(df)} removed)")

    df['symptom_count'] = df['symptoms_text'].apply(lambda x: len(x.split(', ')) if x else 0)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['disease_encoded'] = le.fit_transform(df['Disease'])

    models_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'disease_label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f"\n  AFTER Cleaning:")
    print(f"    Shape: {df.shape}")
    print(f"    Diseases: {df['Disease'].nunique()}")

    df_clean = df[['Disease', 'disease_encoded', 'symptoms_text', 'symptoms_clean', 'symptom_count']].copy()

    clean_path = os.path.join(DATA_DIR, "health_cleaned.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"  Saved cleaned dataset: {clean_path}")

    return df_clean


# ============================================================
# DATASET 2: Medicine Dataset Cleaning
# ============================================================
def clean_medicine_dataset():
    print("=" * 70)
    print("  CLEANING DATASET 2: medicine_dataset.csv")
    print("=" * 70)

    df = pd.read_csv(MEDICINE_PATH, on_bad_lines='warn')

    print(f"\n  BEFORE Cleaning:")
    print(f"    Shape: {df.shape}")

    se_cols = [c for c in df.columns if c.startswith('sideEffect')]

    def get_side_effects(row):
        return [str(row[c]).strip().lower() for c in se_cols if pd.notna(row[c]) and str(row[c]).strip() != '']

    df['side_effects_list'] = df.apply(get_side_effects, axis=1)
    df['side_effects_count'] = df['side_effects_list'].apply(len)

    before_se = len(df)
    df = df[df['side_effects_count'] > 0].reset_index(drop=True)
    print(f"  Rows with side effects: {len(df)} / {before_se}")

    df['drug_name_clean'] = df['name'].fillna('').str.lower().str.strip()

    print("  Building multi-label binary matrix...")
    all_side_effects = set()
    for se_list in df['side_effects_list']:
        all_side_effects.update(se_list)

    MIN_FREQUENCY = 50
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

    print("  Synthesizing input text feature...")
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

    df_save = df[['name', 'drug_name_clean', 'input_text', 'input_text_clean', 'side_effects_list', 'side_effects_count']].copy()
    df_save['side_effects_clean'] = df_save['side_effects_list'].apply(lambda x: ", ".join(x))
    df_save = df_save.drop(columns=['side_effects_list'])

    clean_path = os.path.join(DATA_DIR, "medicine_cleaned.csv")
    df_save.to_csv(clean_path, index=False)

    y_path = os.path.join(DATA_DIR, "medicine_labels.npy")
    np.save(y_path, y_matrix)

    print(f"  Saved cleaned medicine dataset: {clean_path}")
    print(f"  Saved label matrix: {y_path} shape: {y_matrix.shape}")

    return df_save, y_matrix, frequent_se


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DiagnoSense - PHASE 2: Data Cleaning & Preprocessing")
    print("=" * 70 + "\n")

    df_health = clean_health_dataset()
    print()

    df_drug, y_matrix, se_labels = clean_medicine_dataset()

    print("\n" + "=" * 70)
    print("  PHASE 2 - SUMMARY")
    print("=" * 70)
    print("-" * 40)
    print("  Dataset 1: health_cleaned.csv")
    print("-" * 40)
    print(f"  Rows     : {df_health.shape[0]}")
    print(f"  Columns  : {df_health.shape[1]}")
    print(f"  Target   : disease_encoded ({df_health['disease_encoded'].nunique()} classes)")
    print("-" * 40)
    print("  Dataset 2: medicine_cleaned.csv")
    print("-" * 40)
    print(f"  Rows           : {df_drug.shape[0]}")
    print(f"  Target labels  : {len(se_labels)} side effects")
    print(f"  Label matrix   : {y_matrix.shape}")
    print("-" * 40)
    print()
    print("  [DONE] PHASE 2 COMPLETE")
