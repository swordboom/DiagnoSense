"""
PHASE 3: Feature Engineering & Leakage-Safe Splitting
======================================================
Creates reproducible train/val/test splits and TF-IDF features.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from phase2_cleaning import normalize_text


# ============================================================
# CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

SPLIT_METADATA_PATH = REPORTS_DIR / "split_metadata.json"
SYMPTOM_DISEASE_PREFIX = "symptom_disease"

SYMPTOM_DISEASE_FULL_PATH = DATA_DIR / "symptom_disease_dataset.csv"
SYMPTOM_DISEASE_TRAIN_PATH = DATA_DIR / "symptom_disease_train.csv"
SYMPTOM_DISEASE_VAL_PATH = DATA_DIR / "symptom_disease_val.csv"
SYMPTOM_DISEASE_TEST_PATH = DATA_DIR / "symptom_disease_test.csv"

LEGACY_HEALTH_TRAIN_PATH = DATA_DIR / "train.csv"
LEGACY_HEALTH_VAL_PATH = DATA_DIR / "val.csv"
LEGACY_HEALTH_TEST_PATH = DATA_DIR / "test.csv"

SYMPTOM_DISEASE_TFIDF_PATH = MODELS_DIR / "symptom_disease_tfidf.pkl"
SYMPTOM_DISEASE_CLASS_WEIGHTS_PATH = MODELS_DIR / "symptom_disease_class_weights.pkl"
SYMPTOM_DISEASE_ENCODER_PATH = MODELS_DIR / "symptom_disease_label_encoder.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_sparse_split(
    prefix: str,
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    X_test: sp.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    sp.save_npz(DATA_DIR / f"{prefix}_X_train.npz", X_train)
    sp.save_npz(DATA_DIR / f"{prefix}_X_val.npz", X_val)
    sp.save_npz(DATA_DIR / f"{prefix}_X_test.npz", X_test)
    np.save(DATA_DIR / f"{prefix}_y_train.npy", y_train)
    np.save(DATA_DIR / f"{prefix}_y_val.npy", y_val)
    np.save(DATA_DIR / f"{prefix}_y_test.npy", y_test)


def _resolve_health_split_paths() -> Dict[str, Path]:
    preferred = {
        "train": SYMPTOM_DISEASE_TRAIN_PATH,
        "val": SYMPTOM_DISEASE_VAL_PATH,
        "test": SYMPTOM_DISEASE_TEST_PATH,
    }
    legacy = {
        "train": LEGACY_HEALTH_TRAIN_PATH,
        "val": LEGACY_HEALTH_VAL_PATH,
        "test": LEGACY_HEALTH_TEST_PATH,
    }

    resolved: Dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        if preferred[split_name].exists():
            resolved[split_name] = preferred[split_name]
            continue
        if legacy[split_name].exists():
            resolved[split_name] = legacy[split_name]
            continue
        raise FileNotFoundError(
            f"Missing required symptom-disease split for '{split_name}'. "
            f"Checked: {preferred[split_name]} and {legacy[split_name]}"
        )

    return resolved


def _find_column_case_insensitive(df: pd.DataFrame, target: str) -> str:
    for col in df.columns:
        if str(col).strip().lower() == target.lower():
            return str(col)
    raise ValueError(f"Missing required column '{target}' in split file. Found: {list(df.columns)}")


def _symptom_signature(values: np.ndarray) -> np.ndarray:
    binary = (values > 0.5).astype(np.uint8)
    return np.array(["".join(row.astype(str).tolist()) for row in binary], dtype=object)


def _build_health_aligned_dataframe(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    disease_col = _find_column_case_insensitive(df, "disease")
    lower_map = {str(col).strip().lower(): str(col) for col in df.columns if str(col) != disease_col}

    aligned = pd.DataFrame(index=df.index)
    for feat in feature_cols:
        src_col = lower_map.get(str(feat).strip().lower())
        if src_col is None:
            aligned[feat] = 0.0
        else:
            aligned[feat] = pd.to_numeric(df[src_col], errors="coerce").fillna(0.0)

    aligned["disease"] = df[disease_col].astype(str).str.strip()
    return aligned


def _enforce_zero_overlap_health_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    disease_col = _find_column_case_insensitive(train_df, "disease")
    base_feature_cols = [str(col) for col in train_df.columns if str(col) != disease_col]
    if not base_feature_cols:
        raise ValueError("Health split files have no symptom feature columns.")

    train_aligned = _build_health_aligned_dataframe(train_df, base_feature_cols)
    val_aligned = _build_health_aligned_dataframe(val_df, base_feature_cols)
    test_aligned = _build_health_aligned_dataframe(test_df, base_feature_cols)

    train_aligned["split_origin"] = "train"
    val_aligned["split_origin"] = "val"
    test_aligned["split_origin"] = "test"
    full = pd.concat([train_aligned, val_aligned, test_aligned], axis=0, ignore_index=True)
    full = full[full["disease"] != ""].copy().reset_index(drop=True)

    values = full[base_feature_cols].to_numpy(dtype=np.float32)
    full["symptom_key"] = _symptom_signature(values)

    by_origin = {name: set(full.loc[full["split_origin"] == name, "symptom_key"].tolist()) for name in ("train", "val", "test")}
    original_overlap = {
        "train_val": int(len(by_origin["train"] & by_origin["val"])),
        "train_test": int(len(by_origin["train"] & by_origin["test"])),
        "val_test": int(len(by_origin["val"] & by_origin["test"])),
    }

    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(splitter_1.split(full, full["disease"], groups=full["symptom_key"]))
    temp = full.iloc[temp_idx].copy().reset_index(drop=True)

    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(splitter_2.split(temp, temp["disease"], groups=temp["symptom_key"]))

    train_new = full.iloc[train_idx].copy().reset_index(drop=True)
    val_new = temp.iloc[val_rel].copy().reset_index(drop=True)
    test_new = temp.iloc[test_rel].copy().reset_index(drop=True)

    train_keys = set(train_new["symptom_key"].tolist())
    val_keys = set(val_new["symptom_key"].tolist())
    test_keys = set(test_new["symptom_key"].tolist())
    final_overlap = {
        "train_val": int(len(train_keys & val_keys)),
        "train_test": int(len(train_keys & test_keys)),
        "val_test": int(len(val_keys & test_keys)),
    }

    train_out = train_new[base_feature_cols + ["disease"]].copy()
    val_out = val_new[base_feature_cols + ["disease"]].copy()
    test_out = test_new[base_feature_cols + ["disease"]].copy()

    details = {
        "enforced_zero_overlap": True,
        "rows_total": int(len(full)),
        "unique_symptom_keys_total": int(full["symptom_key"].nunique()),
        "split_sizes": {
            "train": int(len(train_out)),
            "val": int(len(val_out)),
            "test": int(len(test_out)),
        },
        "original_overlap_from_input_splits": original_overlap,
        "post_enforcement_overlap": final_overlap,
    }
    return train_out, val_out, test_out, details


def _split_to_text_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    disease_col = _find_column_case_insensitive(df, "disease")
    symptom_cols = [col for col in df.columns if col != disease_col]
    if not symptom_cols:
        raise ValueError("No symptom columns found after excluding the disease column.")

    values = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    symptom_names = np.array([str(col).replace("_", " ") for col in symptom_cols], dtype=object)

    texts = np.array([", ".join(symptom_names[row > 0.5]) for row in values], dtype=object)
    texts = np.array([normalize_text(text) for text in texts], dtype=object)
    labels = df[disease_col].astype(str).str.strip().to_numpy()

    if np.any(texts == ""):
        bad_count = int(np.sum(texts == ""))
        raise ValueError(f"Found {bad_count} rows with no active symptoms in split file.")

    return texts, labels, [str(x) for x in symptom_cols]


def _overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    return len(set(a.tolist()) & set(b.tolist()))


# ============================================================
# DATASET 1: HEALTH
# ============================================================
def process_health_dataset() -> Dict[str, Tuple[int, int]]:
    print("=" * 70)
    print("  PHASE 3: Processing Dataset 1 (Symptom -> Disease)")
    print("=" * 70)

    split_paths = _resolve_health_split_paths()
    train_raw = pd.read_csv(split_paths["train"])
    val_raw = pd.read_csv(split_paths["val"])
    test_raw = pd.read_csv(split_paths["test"])

    train_df, val_df, test_df, split_details = _enforce_zero_overlap_health_splits(train_raw, val_raw, test_raw)

    X_train_text, y_train_raw, symptom_cols = _split_to_text_and_labels(train_df)
    X_val_text, y_val_raw, _ = _split_to_text_and_labels(val_df)
    X_test_text, y_test_raw, _ = _split_to_text_and_labels(test_df)

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_raw, y_val_raw, y_test_raw], axis=0))
    y_train = encoder.transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    with SYMPTOM_DISEASE_ENCODER_PATH.open("wb") as handle:
        pickle.dump(encoder, handle)

    print("  Split source: pre-generated CSV files (re-grouped by symptom signature)")
    print(f"  Train file: {split_paths['train'].name}")
    print(f"  Val file  : {split_paths['val'].name}")
    print(f"  Test file : {split_paths['test'].name}")
    print(f"  Input overlap train-val/test/val-test: {split_details['original_overlap_from_input_splits']}")
    print(f"  Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  Text overlap train-val: {_overlap_count(X_train_text, X_val_text)}")
    print(f"  Text overlap train-test: {_overlap_count(X_train_text, X_test_text)}")
    print(f"  Text overlap val-test: {_overlap_count(X_val_text, X_test_text)}")
    print(f"  Symptom features: {len(symptom_cols)}")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.98)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    _save_sparse_split(
        SYMPTOM_DISEASE_PREFIX,
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf,
        y_train,
        y_val,
        y_test,
    )

    with SYMPTOM_DISEASE_TFIDF_PATH.open("wb") as handle:
        pickle.dump(tfidf, handle)
    with SYMPTOM_DISEASE_CLASS_WEIGHTS_PATH.open("wb") as handle:
        pickle.dump(class_weights, handle)

    return {
        "train": X_train_tfidf.shape,
        "val": X_val_tfidf.shape,
        "test": X_test_tfidf.shape,
        "train_text": X_train_text,
        "val_text": X_val_text,
        "test_text": X_test_text,
        "split_method": "pre_split_csv_group_enforced_no_overlap",
        "source_files": {
            "dataset": SYMPTOM_DISEASE_FULL_PATH.name if SYMPTOM_DISEASE_FULL_PATH.exists() else "",
            "train": split_paths["train"].name,
            "val": split_paths["val"].name,
            "test": split_paths["test"].name,
        },
        "split_details": split_details,
    }


# ============================================================
# DATASET 2: MEDICINE
# ============================================================
def process_medicine_dataset() -> Dict[str, Tuple[int, int]]:
    print("\n" + "=" * 70)
    print("  PHASE 3: Processing Dataset 2 (Medicine Side Effects)")
    print("=" * 70)

    df = pd.read_csv(DATA_DIR / "medicine_cleaned.csv")
    y_full = np.load(DATA_DIR / "medicine_labels.npy")

    df["input_text_clean"] = df["input_text_clean"].fillna("")
    X_text = df["input_text_clean"].to_numpy()
    groups = df["input_text_clean"].to_numpy()

    splitter_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(splitter_1.split(X_text, y_full, groups=groups))

    X_train = X_text[train_idx]
    y_train = y_full[train_idx]

    X_temp = X_text[temp_idx]
    y_temp = y_full[temp_idx]
    temp_groups = groups[temp_idx]

    splitter_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel_idx, test_rel_idx = next(splitter_2.split(X_temp, y_temp, groups=temp_groups))

    X_val = X_temp[val_rel_idx]
    y_val = y_temp[val_rel_idx]
    X_test = X_temp[test_rel_idx]
    y_test = y_temp[test_rel_idx]

    print(f"  Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"  Group overlap train-val: {_overlap_count(X_train, X_val)}")
    print(f"  Group overlap train-test: {_overlap_count(X_train, X_test)}")
    print(f"  Group overlap val-test: {_overlap_count(X_val, X_test)}")

    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=3, max_df=0.99)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    _save_sparse_split("medicine", X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test)

    with (MODELS_DIR / "medicine_tfidf.pkl").open("wb") as handle:
        pickle.dump(tfidf, handle)

    return {
        "train": X_train_tfidf.shape,
        "val": X_val_tfidf.shape,
        "test": X_test_tfidf.shape,
        "train_text": X_train,
        "val_text": X_val,
        "test_text": X_test,
    }


def _load_existing_metadata() -> Dict[str, object]:
    if SPLIT_METADATA_PATH.exists():
        with SPLIT_METADATA_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_split_metadata(health_shapes: Dict | None = None, medicine_shapes: Dict | None = None) -> None:
    metadata: Dict[str, object] = _load_existing_metadata()
    metadata["seed"] = SEED

    if health_shapes is not None:
        metadata["health"] = {
            "train_shape": list(health_shapes["train"]),
            "val_shape": list(health_shapes["val"]),
            "test_shape": list(health_shapes["test"]),
            "split_method": health_shapes.get("split_method", "unknown"),
            "source_files": health_shapes.get("source_files", {}),
            "text_overlap_train_val": _overlap_count(health_shapes["train_text"], health_shapes["val_text"]),
            "text_overlap_train_test": _overlap_count(health_shapes["train_text"], health_shapes["test_text"]),
            "text_overlap_val_test": _overlap_count(health_shapes["val_text"], health_shapes["test_text"]),
            "split_details": health_shapes.get("split_details", {}),
        }

    if medicine_shapes is not None:
        metadata["medicine"] = {
            "train_shape": list(medicine_shapes["train"]),
            "val_shape": list(medicine_shapes["val"]),
            "test_shape": list(medicine_shapes["test"]),
            "text_overlap_train_val": _overlap_count(medicine_shapes["train_text"], medicine_shapes["val_text"]),
            "text_overlap_train_test": _overlap_count(medicine_shapes["train_text"], medicine_shapes["test_text"]),
            "text_overlap_val_test": _overlap_count(medicine_shapes["val_text"], medicine_shapes["test_text"]),
        }

    with SPLIT_METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"  Saved split metadata: {SPLIT_METADATA_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiagnoSense Phase 3 feature engineering")
    parser.add_argument(
        "--task",
        choices=["all", "symptom_disease", "medicine"],
        default="all",
        help="Choose which pipeline to process.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_health = args.task in ("all", "symptom_disease")
    run_medicine = args.task in ("all", "medicine")

    health = process_health_dataset() if run_health else None
    medicine = process_medicine_dataset() if run_medicine else None
    save_split_metadata(health_shapes=health, medicine_shapes=medicine)

    print("\n" + "=" * 70)
    print("  PHASE 3 - SUMMARY")
    print("=" * 70)
    if health is not None:
        print(f"  Symptom-Disease train/val/test: {health['train']} | {health['val']} | {health['test']}")
    if medicine is not None:
        print(f"  Medicine train/val/test: {medicine['train']} | {medicine['val']} | {medicine['test']}")
    print("\n  [DONE] PHASE 3 COMPLETE")