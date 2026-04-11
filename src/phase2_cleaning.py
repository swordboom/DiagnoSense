# -*- coding: utf-8 -*-
"""
PHASE 2: Data Cleaning & Preprocessing
======================================
Build leakage-aware, quality-controlled datasets for both tasks.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


# ============================================================
# CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

HEALTH_SYNTHETIC_PATH = DATA_DIR / "symptom_disease_dataset.csv"
HEALTH_REALISTIC_SYNTHETIC_PATH = DATA_DIR / "symptom_disease_dataset_realistic.csv"
MEDICINE_PATH = DATA_DIR / "medicine_dataset.csv"

HEALTH_CLEAN_PATH = DATA_DIR / "health_cleaned.csv"
MEDICINE_CLEAN_PATH = DATA_DIR / "medicine_cleaned.csv"
MEDICINE_LABELS_PATH = DATA_DIR / "medicine_labels.npy"

HEALTH_LABEL_ENCODER_PATH = MODELS_DIR / "symptom_disease_label_encoder.pkl"
MEDICINE_MLB_PATH = MODELS_DIR / "medicine_se_mlb.pkl"

DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality.json"

MIN_SIDE_EFFECT_FREQUENCY = 75
HEALTH_REALISM_MIN_SYMPTOMS = 6
HEALTH_REALISM_MAX_SYMPTOMS = 11

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ============================================================
# TEXT UTILITIES
# ============================================================
def _normalize_basic(text: object) -> str:
    """Normalize text without lemmatization."""
    if pd.isna(text):
        return ""
    value = str(text).lower().strip()
    if not value:
        return ""
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^a-z0-9\s\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_text(text: object) -> str:
    """Lowercase, strip punctuation, remove stopwords, and lemmatize."""
    value = _normalize_basic(text)
    if not value:
        return ""

    tokens = value.split()
    normalized = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return " ".join(normalized).strip()


def normalize_symptom(symptom: object) -> Optional[str]:
    """Normalize a single symptom string."""
    value = _normalize_basic(symptom)
    if len(value) < 2:
        return None
    return value


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _canonical_key(items: Sequence[str]) -> str:
    return " | ".join(sorted(set(item for item in items if item)))


def _sample_weighted_unique(candidates: Sequence[str], weights: Sequence[float], k: int, rng: np.random.Generator) -> List[str]:
    if k <= 0:
        return []
    items = [str(x) for x in candidates if str(x).strip()]
    if not items:
        return []
    unique_items = list(dict.fromkeys(items))
    k = min(k, len(unique_items))

    weight_map = {str(c): float(w) for c, w in zip(candidates, weights)}
    probs = np.array([max(weight_map.get(item, 1.0), 1e-8) for item in unique_items], dtype=np.float64)
    probs = probs / probs.sum()

    selected_idx = rng.choice(len(unique_items), size=k, replace=False, p=probs)
    return [unique_items[int(i)] for i in selected_idx]


# ============================================================
# DATASET 1: HEALTH CLEANING
# ============================================================
def _resolve_health_source_path() -> Path:
    if HEALTH_SYNTHETIC_PATH.exists():
        return HEALTH_SYNTHETIC_PATH
    raise FileNotFoundError(f"Could not find required health source file: {HEALTH_SYNTHETIC_PATH}")

#cleaning symptom string
def _split_symptom_string(value: object) -> List[str]:
    if pd.isna(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []

    parts = [normalize_symptom(chunk) for chunk in raw.split(",")]
    parts = [p for p in parts if p]
    return _dedupe_preserve_order(parts)


def _prepare_health_dataframe(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Supported schemas:
    1) Disease, Symptoms (comma-separated symptoms)
    2) Binary symptom matrix columns + disease label column
    """
    preview = pd.read_csv(path, nrows=5)
    lower_cols = {str(c).strip().lower() for c in preview.columns}

    if {"disease", "symptoms"}.issubset(lower_cols):
        df = pd.read_csv(path)
        disease_col = next(col for col in df.columns if str(col).strip().lower() == "disease")
        symptoms_col = next(col for col in df.columns if str(col).strip().lower() == "symptoms")
        out = pd.DataFrame(
            {
                "Disease": df[disease_col],
                "symptoms_list": df[symptoms_col].map(_split_symptom_string),
            }
        )
        return out, "disease_symptoms_text"

    if "disease" not in lower_cols:
        raise ValueError(
            f"{path.name} must contain either (Disease, Symptoms) or a binary symptom matrix with a disease column. "
            f"Found: {list(preview.columns)}"
        )

    df = pd.read_csv(path)
    disease_col = next(col for col in df.columns if str(col).strip().lower() == "disease")
    symptom_cols = [col for col in df.columns if col != disease_col]
    if not symptom_cols:
        raise ValueError(f"{path.name} has no symptom feature columns.")

    values = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    normalized_symptoms = [normalize_symptom(str(col).replace("_", " ")) for col in symptom_cols]

    symptoms_list: List[List[str]] = []
    for row in values:
        active = [
            normalized_symptoms[idx]
            for idx, val in enumerate(row)
            if val > 0.5 and normalized_symptoms[idx]
        ]
        symptoms_list.append(_dedupe_preserve_order(active))

    out = pd.DataFrame(
        {
            "Disease": df[disease_col],
            "symptoms_list": symptoms_list,
        }
    )
    return out, "binary_symptom_matrix"

# Making dataset more work efficient
def _make_synthetic_health_more_realistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce synthetic shortcut patterns by increasing symptom overlap/noise and
    by lowering per-row symptom cardinality to a more realistic range.
    """
    rng = np.random.default_rng(SEED)
    rows = df.copy().reset_index(drop=True)
    rows["Disease"] = rows["Disease"].astype(str).str.strip()
    rows["symptoms_list"] = rows["symptoms_list"].map(lambda arr: _dedupe_preserve_order([x for x in arr if x]))

    disease_counter_map: Dict[str, Counter[str]] = {}
    global_counter: Counter[str] = Counter()
    for _, row in rows.iterrows():
        disease = row["Disease"]
        symptoms = row["symptoms_list"]
        disease_counter_map.setdefault(disease, Counter()).update(symptoms)
        global_counter.update(symptoms)

    global_counts = np.array(list(global_counter.values()), dtype=np.float32)
    common_threshold = int(np.percentile(global_counts, 60)) if len(global_counts) else 1
    common_pool = [sym for sym, cnt in global_counter.items() if cnt >= max(common_threshold, 1)]
    if not common_pool:
        common_pool = list(global_counter.keys())

    all_diseases = sorted(rows["Disease"].unique().tolist())

    augmented_records: List[Dict[str, object]] = []
    for _, row in rows.iterrows():
        disease = row["Disease"]
        original = row["symptoms_list"]
        disease_counter = disease_counter_map[disease]
        disease_pool = list(disease_counter.keys())

        target_count = int(rng.integers(HEALTH_REALISM_MIN_SYMPTOMS, HEALTH_REALISM_MAX_SYMPTOMS + 1))

        keep_from_original = max(3, min(len(original), int(round(len(original) * rng.uniform(0.40, 0.65)))))
        preserved = _sample_weighted_unique(
            candidates=original,
            weights=[disease_counter.get(sym, 1.0) for sym in original],
            k=keep_from_original,
            rng=rng,
        )

        disease_target = max(3, min(target_count - 2, target_count))
        disease_samples = _sample_weighted_unique(
            candidates=disease_pool,
            weights=[disease_counter[sym] for sym in disease_pool],
            k=disease_target,
            rng=rng,
        )

        merged = _dedupe_preserve_order([*preserved, *disease_samples])

        shared_budget = max(0, target_count - len(merged))
        shared_candidates = [sym for sym in common_pool if sym not in merged]
        if shared_budget > 0 and shared_candidates:
            shared = _sample_weighted_unique(
                candidates=shared_candidates,
                weights=[global_counter[sym] for sym in shared_candidates],
                k=shared_budget,
                rng=rng,
            )
            merged = _dedupe_preserve_order([*merged, *shared])

        if rng.random() < 0.35 and len(all_diseases) > 1:
            other_disease = disease
            for _ in range(8):
                candidate = all_diseases[int(rng.integers(0, len(all_diseases)))]
                if candidate != disease:
                    other_disease = candidate
                    break
            other_pool = list(disease_counter_map[other_disease].keys())
            if other_pool:
                injected = other_pool[int(rng.integers(0, len(other_pool)))]
                if injected not in merged:
                    merged.append(injected)

        if len(merged) > HEALTH_REALISM_MAX_SYMPTOMS:
            merged = _sample_weighted_unique(
                candidates=merged,
                weights=[global_counter.get(sym, 1.0) for sym in merged],
                k=HEALTH_REALISM_MAX_SYMPTOMS,
                rng=rng,
            )
        if len(merged) < HEALTH_REALISM_MIN_SYMPTOMS:
            top_up_candidates = [sym for sym in common_pool if sym not in merged]
            need = min(HEALTH_REALISM_MIN_SYMPTOMS - len(merged), len(top_up_candidates))
            if need > 0:
                merged.extend(
                    _sample_weighted_unique(
                        candidates=top_up_candidates,
                        weights=[global_counter[sym] for sym in top_up_candidates],
                        k=need,
                        rng=rng,
                    )
                )

        merged = _dedupe_preserve_order([normalize_symptom(sym) for sym in merged if normalize_symptom(sym)])
        if len(merged) < 3:
            fallback = _sample_weighted_unique(
                candidates=disease_pool,
                weights=[disease_counter[sym] for sym in disease_pool],
                k=min(max(3, HEALTH_REALISM_MIN_SYMPTOMS), len(disease_pool)),
                rng=rng,
            )
            merged = _dedupe_preserve_order([normalize_symptom(sym) for sym in fallback if normalize_symptom(sym)])

        augmented_records.append({"Disease": disease, "symptoms_list": merged})

    out = pd.DataFrame(augmented_records)
    out["Disease"] = out["Disease"].astype(str).str.strip()
    out["symptoms_list"] = out["symptoms_list"].map(_dedupe_preserve_order)
    return out

#cleaning health dataset 
def clean_health_dataset() -> tuple[pd.DataFrame, Dict[str, object]]:
    health_source_path = _resolve_health_source_path()

    print("=" * 70)
    print(f"  CLEANING DATASET 1: {health_source_path.name}")
    print("=" * 70)

    df, source_schema = _prepare_health_dataframe(health_source_path)
    realism_augmented = False
    realism_output_file = None

    if health_source_path.name == HEALTH_SYNTHETIC_PATH.name and source_schema == "disease_symptoms_text":
        df = _make_synthetic_health_more_realistic(df)
        realism_augmented = True
        realism_output_file = HEALTH_REALISTIC_SYNTHETIC_PATH.name
        pd.DataFrame(
            {
                "Disease": df["Disease"].astype(str),
                "Symptoms": df["symptoms_list"].map(lambda x: ", ".join(x)),
            }
        ).to_csv(HEALTH_REALISTIC_SYNTHETIC_PATH, index=False)

    stats: Dict[str, object] = {
        "source_file": health_source_path.name,
        "source_schema": source_schema,
        "realism_augmented": float(1 if realism_augmented else 0),
        "rows_raw": float(len(df)),
        "disease_classes_raw": float(df["Disease"].nunique()),
    }
    if realism_output_file:
        stats["realistic_file"] = realism_output_file

    df = df.dropna(subset=["Disease"]).copy()
    df["Disease"] = df["Disease"].astype(str).str.strip()
    df = df[df["Disease"] != ""].copy()

    df["symptoms_list"] = df["symptoms_list"].map(_dedupe_preserve_order)
    df = df[df["symptoms_list"].map(len) > 0].copy()

    df["symptom_key"] = df["symptoms_list"].map(_canonical_key)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Disease", "symptom_key"]).copy()
    stats["rows_removed_exact_duplicates"] = float(before_dedup - len(df))

    key_class_counts = df.groupby("symptom_key")["Disease"].nunique()
    ambiguous_keys = set(key_class_counts[key_class_counts > 1].index)
    ambiguous_rows = int(df["symptom_key"].isin(ambiguous_keys).sum())
    if ambiguous_rows:
        df = df[~df["symptom_key"].isin(ambiguous_keys)].copy()
    stats["rows_removed_ambiguous_symptom_keys"] = float(ambiguous_rows)

    df["symptoms_text"] = df["symptoms_list"].map(lambda x: ", ".join(x))
    df["symptoms_clean"] = df["symptoms_text"].map(normalize_text)
    df["symptom_count"] = df["symptoms_list"].map(len)

    le = LabelEncoder()
    df["disease_encoded"] = le.fit_transform(df["Disease"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with HEALTH_LABEL_ENCODER_PATH.open("wb") as handle:
        pickle.dump(le, handle)

    out_df = df[
        [
            "Disease",
            "disease_encoded",
            "symptoms_text",
            "symptoms_clean",
            "symptom_count",
            "symptom_key",
        ]
    ].reset_index(drop=True)

    out_df.to_csv(HEALTH_CLEAN_PATH, index=False)

    stats.update(
        {
            "rows_clean": float(len(out_df)),
            "disease_classes_clean": float(out_df["Disease"].nunique()),
            "unique_symptom_keys_clean": float(out_df["symptom_key"].nunique()),
            "avg_symptoms_per_row": float(out_df["symptom_count"].mean()),
            "min_class_count_clean": float(out_df["Disease"].value_counts().min()),
            "max_class_count_clean": float(out_df["Disease"].value_counts().max()),
            "class_balance_ratio_clean": float(out_df["Disease"].value_counts().min() / out_df["Disease"].value_counts().max()),
        }
    )

    print(f"  Source file: {health_source_path}")
    if realism_augmented:
        print(f"  Realism augmentation applied: yes ({HEALTH_REALISTIC_SYNTHETIC_PATH})")
    print(f"  Raw rows: {int(stats['rows_raw'])}")
    print(f"  Clean rows: {int(stats['rows_clean'])}")
    print(f"  Disease classes: {int(stats['disease_classes_clean'])}")
    print(f"  Removed duplicates: {int(stats['rows_removed_exact_duplicates'])}")
    print(f"  Removed ambiguous keys: {int(stats['rows_removed_ambiguous_symptom_keys'])}")
    print(f"  Saved: {HEALTH_CLEAN_PATH}")

    return out_df, stats


# ============================================================
# DATASET 2: MEDICINE CLEANING
# ============================================================
def _extract_side_effect_list(row: pd.Series, se_cols: Sequence[str]) -> List[str]:
    values = [_normalize_basic(row[col]) for col in se_cols]
    values = [val for val in values if val]
    return _dedupe_preserve_order(values)


def _compose_medicine_text(row: pd.Series, use_cols: Sequence[str]) -> str:
    chunks: List[str] = []

    base_name = _normalize_basic(row.get("name", ""))
    if base_name:
        chunks.append(base_name)

    for col in use_cols:
        value = _normalize_basic(row.get(col, ""))
        if value:
            chunks.append(value)

    for col in ("Chemical Class", "Therapeutic Class", "Action Class"):
        value = _normalize_basic(row.get(col, ""))
        if value and value != "na":
            chunks.append(value)

    return " ".join(chunks).strip()


def _union_lists(values: Iterable[Sequence[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for arr in values:
        for value in arr:
            if value and value not in seen:
                seen.add(value)
                merged.append(value)
    return merged

# cleaning medicine dataset
def clean_medicine_dataset() -> tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    print("\n" + "=" * 70)
    print("  CLEANING DATASET 2: medicine_dataset.csv")
    print("=" * 70)

    df = pd.read_csv(MEDICINE_PATH, on_bad_lines="warn", low_memory=False)
    se_cols = [col for col in df.columns if col.startswith("sideEffect")]
    use_cols = [col for col in df.columns if col.startswith("use")]

    stats: Dict[str, object] = {"rows_raw": float(len(df))}

    df["side_effects_list"] = df.apply(lambda row: _extract_side_effect_list(row, se_cols), axis=1)
    df["side_effects_count"] = df["side_effects_list"].map(len)
    df = df[df["side_effects_count"] > 0].copy()
    stats["rows_after_non_empty_side_effects"] = float(len(df))

    df["input_text"] = df.apply(lambda row: _compose_medicine_text(row, use_cols), axis=1)
    df["input_text_clean"] = df["input_text"].map(normalize_text)
    df = df[df["input_text_clean"] != ""].copy()

    df["side_effects_key"] = df["side_effects_list"].map(_canonical_key)
    before_exact = len(df)
    df = df.drop_duplicates(subset=["input_text_clean", "side_effects_key"]).copy()
    stats["rows_removed_exact_duplicates"] = float(before_exact - len(df))

    grouped = (
        df.groupby("input_text_clean", as_index=False)
        .agg(
            {
                "name": "first",
                "input_text": "first",
                "side_effects_list": _union_lists,
            }
        )
        .reset_index(drop=True)
    )

    grouped["side_effects_count"] = grouped["side_effects_list"].map(len)
    grouped["side_effects_clean"] = grouped["side_effects_list"].map(lambda x: ", ".join(x))
    grouped["drug_name_clean"] = grouped["name"].map(_normalize_basic)

    counter: Counter[str] = Counter()
    for labels in grouped["side_effects_list"]:
        counter.update(labels)

    frequent_labels = sorted([label for label, count in counter.items() if count >= MIN_SIDE_EFFECT_FREQUENCY])
    frequent_set = set(frequent_labels)

    grouped["side_effects_filtered"] = grouped["side_effects_list"].map(
        lambda labels: [label for label in labels if label in frequent_set]
    )
    grouped = grouped[grouped["side_effects_filtered"].map(len) > 0].copy().reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=frequent_labels)
    y_matrix = mlb.fit_transform(grouped["side_effects_filtered"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MEDICINE_MLB_PATH.open("wb") as handle:
        pickle.dump(mlb, handle)

    save_df = grouped[
        [
            "name",
            "drug_name_clean",
            "input_text",
            "input_text_clean",
            "side_effects_count",
            "side_effects_clean",
        ]
    ].copy()

    save_df.to_csv(MEDICINE_CLEAN_PATH, index=False)
    np.save(MEDICINE_LABELS_PATH, y_matrix)

    stats.update(
        {
            "rows_clean": float(len(save_df)),
            "unique_input_text_clean": float(save_df["input_text_clean"].nunique()),
            "label_classes_kept": float(len(frequent_labels)),
            "avg_labels_per_row": float(y_matrix.sum(axis=1).mean()),
            "label_density": float(y_matrix.mean()),
            "min_label_frequency_threshold": float(MIN_SIDE_EFFECT_FREQUENCY),
        }
    )

    print(f"  Raw rows: {int(stats['rows_raw'])}")
    print(f"  Clean rows: {int(stats['rows_clean'])}")
    print(f"  Unique text contexts: {int(stats['unique_input_text_clean'])}")
    print(f"  Labels kept: {int(stats['label_classes_kept'])}")
    print(f"  Avg labels/sample: {stats['avg_labels_per_row']:.2f}")
    print(f"  Saved dataset: {MEDICINE_CLEAN_PATH}")
    print(f"  Saved labels: {MEDICINE_LABELS_PATH} shape={y_matrix.shape}")

    return save_df, y_matrix, stats

# Saving report
def save_data_quality_report(health_stats: Dict[str, object], medicine_stats: Dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "seed": SEED,
        "health": health_stats,
        "medicine": medicine_stats,
    }
    with DATA_QUALITY_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"  Saved quality report: {DATA_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DiagnoSense - PHASE 2: Data Cleaning & Preprocessing")
    print("=" * 70 + "\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    health_df, health_stats = clean_health_dataset()
    med_df, med_labels, medicine_stats = clean_medicine_dataset()
    save_data_quality_report(health_stats, medicine_stats)

    print("\n" + "=" * 70)
    print("  PHASE 2 - SUMMARY")
    print("=" * 70)
    print(f"  Health rows      : {len(health_df)}")
    print(f"  Health classes   : {health_df['Disease'].nunique()}")
    print(f"  Medicine rows    : {len(med_df)}")
    print(f"  Medicine labels  : {med_labels.shape[1]}")
    print(f"  Label density    : {med_labels.mean():.4f}")
    print("\n  [DONE] PHASE 2 COMPLETE")
