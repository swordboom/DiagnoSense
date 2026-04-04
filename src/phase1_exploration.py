"""
PHASE 1: Data Loading & Initial Exploration
=============================================
DiagnoSense - ML Pipeline for Disease Prediction & Drug Side Effects
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU AVAILABILITY CHECK
# ============================================================
def check_gpu():
    print("=" * 70)
    print("  GPU / CUDA AVAILABILITY CHECK")
    print("=" * 70)
    
    try:
        import torch
        print(f"  PyTorch version    : {torch.__version__}")
        print(f"  CUDA available     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version       : {torch.version.cuda}")
            print(f"  GPU device         : {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  VRAM               : {vram:.1f} GB")
            print(f"  Compute capability : {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
            DEVICE = "cuda"
        else:
            print(f"  NOTE: CUDA not available with current PyTorch build")
            print(f"  PyTorch CUDA build : {torch.version.cuda}")
            DEVICE = "cpu"
    except ImportError:
        print("  PyTorch not installed")
        DEVICE = "cpu"
    
    # Check nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  nvidia-smi GPU     : {result.stdout.strip()}")
            print(f"  GPU is PRESENT (driver detected)")
        else:
            print(f"  nvidia-smi         : Not available")
    except:
        print(f"  nvidia-smi         : Not found")
    
    print(f"\n  >>> Device for training: {DEVICE.upper()}")
    print()
    return DEVICE

# ============================================================
# DATASET 1: Health Dataset (Symptom → Disease)
# ============================================================
def explore_health_dataset(filepath):
    print("=" * 70)
    print("  DATASET 1: Health Dataset (Symptom → Disease Prediction)")
    print("=" * 70)
    
    # This CSV has variable number of columns per row - need to handle that
    # First, find the max number of columns
    max_cols = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.reader(f)
        for row in reader:
            max_cols = max(max_cols, len(row))
    
    # Read with max columns - skip original header since it has fewer columns
    col_names = ['Disease'] + [f'Symptom_{i}' for i in range(1, max_cols)]
    df = pd.read_csv(filepath, header=None, skiprows=1, names=col_names, on_bad_lines='warn')
    
    print(f"\n  Shape              : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Max symptom cols   : {max_cols - 1}")
    
    print(f"\n  Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns[:5], 1):
        print(f"    {i}. {col}")
    print(f"    ... (up to Symptom_{max_cols-1})")
    
    # Data types
    print(f"\n  Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {dtype}: {count} columns")
    
    # Sample rows
    print(f"\n  Sample Rows (first 5):")
    display_cols = ['Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    print(df[display_cols].head().to_string(index=True))
    
    # Missing values analysis
    print(f"\n  Missing Values Summary:")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"    Total missing cells   : {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.1f}%)")
    
    # Show missing per column (only first few and the pattern)
    print(f"\n    Per-column missing counts (first 8 cols):")
    for col in df.columns[:8]:
        m = df[col].isnull().sum()
        print(f"      {col:15s}: {m:5d} ({m/len(df)*100:5.1f}%)")
    
    # Symptom fill pattern
    symptom_counts = df.iloc[:, 1:].notna().sum(axis=1)
    print(f"\n    Symptoms per row:")
    print(f"      Min   : {symptom_counts.min()}")
    print(f"      Max   : {symptom_counts.max()}")
    print(f"      Mean  : {symptom_counts.mean():.1f}")
    print(f"      Median: {symptom_counts.median():.0f}")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n  Duplicate Rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")
    
    # Disease distribution
    disease_counts = df['Disease'].value_counts()
    print(f"\n  Disease Distribution ({df['Disease'].nunique()} unique diseases):")
    print(f"    Top 10:")
    for disease, count in disease_counts.head(10).items():
        print(f"      {disease:45s}: {count:4d} ({count/len(df)*100:.1f}%)")
    print(f"    ...")
    print(f"    Bottom 5:")
    for disease, count in disease_counts.tail(5).items():
        print(f"      {disease:45s}: {count:4d} ({count/len(df)*100:.1f}%)")
    
    # Unique symptoms
    all_symptoms = set()
    for col in df.columns[1:]:
        vals = df[col].dropna().unique()
        all_symptoms.update(vals)
    all_symptoms.discard('')
    print(f"\n  Total Unique Symptoms: {len(all_symptoms)}")
    print(f"  Sample symptoms: {list(all_symptoms)[:10]}")
    
    print()
    return df

# ============================================================
# DATASET 2: Drug Side Effects
# ============================================================
def explore_drug_dataset(filepath):
    print("=" * 70)
    print("  DATASET 2: Drug Side Effects (Drug → Side Effects Prediction)")
    print("=" * 70)
    
    df = pd.read_csv(filepath, quotechar='"', on_bad_lines='warn')
    
    print(f"\n  Shape              : {df.shape[0]} rows x {df.shape[1]} columns")
    
    print(f"\n  Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        print(f"    {i:2d}. {col:35s} (non-null: {non_null:5d}, dtype: {df[col].dtype})")
    
    # Sample row
    print(f"\n  Sample Row (first row, key fields):")
    if 'drug_name' in df.columns:
        print(f"    drug_name       : {df['drug_name'].iloc[0]}")
    if 'medical_condition' in df.columns:
        print(f"    medical_condition: {df['medical_condition'].iloc[0]}")
    if 'side_effects' in df.columns:
        se_sample = str(df['side_effects'].iloc[0])[:200]
        print(f"    side_effects    : {se_sample}...")
    if 'generic_name' in df.columns:
        print(f"    generic_name    : {df['generic_name'].iloc[0]}")
    if 'drug_classes' in df.columns:
        print(f"    drug_classes    : {df['drug_classes'].iloc[0]}")
    if 'rating' in df.columns:
        print(f"    rating          : {df['rating'].iloc[0]}")
    
    # Missing values
    print(f"\n  Missing Values:")
    missing = df.isnull().sum()
    for col in df.columns:
        m = missing[col]
        if m > 0:
            print(f"    {col:35s}: {m:5d} ({m/len(df)*100:.1f}%)")
    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\n    Total missing   : {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.1f}%)")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n  Duplicate Rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")
    
    # Key column analysis
    if 'drug_name' in df.columns:
        print(f"\n  Drug Name Stats:")
        print(f"    Unique drugs            : {df['drug_name'].nunique()}")
        print(f"    Top 5 drugs by frequency:")
        for drug, count in df['drug_name'].value_counts().head(5).items():
            print(f"      {drug:30s}: {count}")
    
    if 'medical_condition' in df.columns:
        print(f"\n  Medical Condition Stats:")
        print(f"    Unique conditions       : {df['medical_condition'].nunique()}")
        print(f"    Top 10 conditions:")
        for cond, count in df['medical_condition'].value_counts().head(10).items():
            print(f"      {cond:40s}: {count:4d} ({count/len(df)*100:.1f}%)")
    
    if 'side_effects' in df.columns:
        print(f"\n  Side Effects Column Analysis:")
        se_lengths = df['side_effects'].dropna().str.len()
        print(f"    Non-null entries        : {df['side_effects'].notna().sum()}")
        print(f"    Text length - min       : {se_lengths.min()}")
        print(f"    Text length - max       : {se_lengths.max()}")
        print(f"    Text length - mean      : {se_lengths.mean():.0f}")
    
    if 'rating' in df.columns:
        rating_col = pd.to_numeric(df['rating'], errors='coerce')
        print(f"\n  Rating Stats:")
        print(f"    Min   : {rating_col.min()}")
        print(f"    Max   : {rating_col.max()}")
        print(f"    Mean  : {rating_col.mean():.2f}")
        print(f"    Median: {rating_col.median():.1f}")
    
    if 'rx_otc' in df.columns:
        print(f"\n  Rx/OTC Distribution:")
        for val, count in df['rx_otc'].value_counts().items():
            print(f"    {val:10s}: {count:4d}")
    
    print()
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    print("\n" + "=" * 70)
    print("  DiagnoSense — PHASE 1: Data Loading & Exploration")
    print("=" * 70 + "\n")
    
    device = check_gpu()
    
    # Dataset 1
    health_path = os.path.join(DATA_DIR, "health_dataset.csv")
    df_health = explore_health_dataset(health_path)
    
    # Dataset 2
    drug_path = os.path.join(DATA_DIR, "drug_side_effects.csv")
    df_drug = explore_drug_dataset(drug_path)
    
    # Final Summary
    print("=" * 70)
    print("  PHASE 1 — SUMMARY")
    print("=" * 70)
    
    # Count symptoms per row for health data
    symptom_counts = df_health.iloc[:, 1:].notna().sum(axis=1)
    all_symptoms = set()
    for col in df_health.columns[1:]:
        all_symptoms.update(df_health[col].dropna().unique())
    all_symptoms.discard('')
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 1: health_dataset.csv (Symptom → Disease)             │
  ├─────────────────────────────────────────────────────────────────┤
  │  Rows              : {df_health.shape[0]:>10,}                              │
  │  Columns           : {df_health.shape[1]:>10}                              │
  │  Unique Diseases   : {df_health['Disease'].nunique():>10}                              │
  │  Unique Symptoms   : {len(all_symptoms):>10}                              │
  │  Avg Symptoms/Row  : {symptom_counts.mean():>10.1f}                              │
  │  Missing Values    : {df_health.isnull().sum().sum():>10,}                              │
  │  Duplicates        : {df_health.duplicated().sum():>10}                              │
  │  Task              : Multi-class classification                │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Dataset 2: drug_side_effects.csv (Drug → Side Effects)        │
  ├─────────────────────────────────────────────────────────────────┤
  │  Rows              : {df_drug.shape[0]:>10,}                              │
  │  Columns           : {df_drug.shape[1]:>10}                              │
  │  Unique Drugs      : {df_drug['drug_name'].nunique() if 'drug_name' in df_drug.columns else 'N/A':>10}                              │
  │  Unique Conditions : {df_drug['medical_condition'].nunique() if 'medical_condition' in df_drug.columns else 'N/A':>10}                              │
  │  Missing Values    : {df_drug.isnull().sum().sum():>10,}                              │
  │  Duplicates        : {df_drug.duplicated().sum():>10}                              │
  │  Task              : Multi-label text classification           │
  └─────────────────────────────────────────────────────────────────┘

  GPU Status: {'CUDA AVAILABLE ✅' if device == 'cuda' else 'CPU ONLY ⚠️ (PyTorch CUDA build needed for GPU)'}
  
  ✅ PHASE 1 COMPLETE — Awaiting approval to proceed to Phase 2
""")
