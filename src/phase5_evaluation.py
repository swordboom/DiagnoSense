"""
PHASE 5: Model Evaluation & Report Generation
=================================================
DiagnoSense - ML Pipeline
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# We reuse models and dataset class from phase 4
from phase4_training import SparseDataset, HealthModel, MedicineModel, DEVICE, DATA_DIR, MODELS_DIR

# ============================================================
# EVALUATION: Health Model
# ============================================================
def evaluate_health():
    print("\n" + "=" * 70)
    print("  EVALUATING: Health Model")
    print("=" * 70)
    
    # Load test data
    X_test = sp.load_npz(os.path.join(DATA_DIR, "health_X_test.npz"))
    y_test = np.load(os.path.join(DATA_DIR, "health_y_test.npy"))
    
    # Load label encoder to get actual disease names
    with open(os.path.join(MODELS_DIR, "disease_label_encoder.pkl"), "rb") as f:
        disease_encoder = pickle.load(f)
    disease_classes = disease_encoder.classes_
    
    # Load model
    model = HealthModel(input_dim=X_test.shape[1], num_classes=len(disease_classes)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "health_model.pth"), weights_only=True))
    model.eval()
    
    test_dataset = SparseDataset(X_test, y_test, task='multiclass')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1_macro:.4f}")
    
    target_names = [str(x)[:25] for x in disease_classes]
    
    # Due to severe class imbalance, some classes might have 0 support in the split,
    # or warnings might be thrown, handle them properly.
    # Expected labels
    labels = list(range(len(disease_classes)))
    report_dict = classification_report(all_targets, all_preds, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, labels=labels, target_names=target_names, zero_division=0)
    
    print("\n" + report_str)
    
    return acc, f1_macro, report_dict, report_str

# ============================================================
# EVALUATION: Medicine Model
# ============================================================
def evaluate_medicine():
    print("\n" + "=" * 70)
    print("  EVALUATING: Medicine Side Effects Model")
    print("=" * 70)
    
    # Load test data
    X_test = sp.load_npz(os.path.join(DATA_DIR, "medicine_X_test.npz"))
    y_test = np.load(os.path.join(DATA_DIR, "medicine_y_test.npy"))
    
    # Load MLB
    with open(os.path.join(MODELS_DIR, "medicine_se_mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)
    se_classes = mlb.classes_
    
    num_labels = len(se_classes)
    
    # Load model
    model = MedicineModel(input_dim=X_test.shape[1], num_labels=num_labels).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "medicine_model.pth"), weights_only=True))
    model.eval()
    
    test_dataset = SparseDataset(X_test, y_test, task='multilabel')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    all_preds_probs = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds_probs.append(probs.cpu().numpy())
            
    all_preds_probs = np.vstack(all_preds_probs)
    # Thresholding for multi-label
    all_preds = (all_preds_probs > 0.5).astype(int)
    
    # Metrics
    micro_f1 = f1_score(y_test, all_preds, average='micro')
    macro_f1 = f1_score(y_test, all_preds, average='macro', zero_division=0)
    samples_f1 = f1_score(y_test, all_preds, average='samples', zero_division=0)
    
    print(f"  Micro F1   : {micro_f1:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Samples F1 : {samples_f1:.4f}")
    
    report_dict = classification_report(y_test, all_preds, target_names=se_classes, output_dict=True, zero_division=0)
    
    return micro_f1, macro_f1, samples_f1, report_dict

# ============================================================
# GENERATE REPORT
# ============================================================
def generate_report(health_metrics, medicine_metrics):
    h_acc, h_f1, h_report, h_str = health_metrics
    d_mi_f1, d_ma_f1, d_sa_f1, _ = medicine_metrics
    
    reports_dir = os.path.join(os.path.dirname(DATA_DIR), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "evaluation.md")
    
    content = f"""# DiagnoSense Evaluation Report & Comparative Study

## 1. Executive Summary
This document summarizes the final evaluation metrics for the **DiagnoSense** Machine Learning pipelines. Both models were accelerated using PyTorch (CUDA 12.8) and trained on an RTX 3050 GPU.

- **Pipeline 1 (Health)** achieves `{h_acc:.2%}` accuracy using Multi-Class PyTorch MLP with Class Weighting.
- **Pipeline 2 (Medicine)** achieves `{d_mi_f1:.2%}` micro-F1 using Multi-Label PyTorch MLP with BCEWithLogits.

---

## 2. Model 1: Symptom → Disease (Multi-Class)
**Task:** Given a set of patient symptoms, predict the exact disease classification out of 16 heavily imbalanced classes.

**Architecture:**
- Input TF-IDF features (Max 5,000)
- 2 Hidden Layers (512 -> 256) with BatchNorm and Dropout 
- CrossEntropyLoss with Computed Class Weights (Critical for detecting minority classes like Measles)

**Results (Test Set = 1,814 samples):**
- **Accuracy:** {h_acc:.4f}
- **F1 Score (Macro):** {h_f1:.4f}

**Classification Report snippet:**
```text
{h_str}
```

---

## 3. Model 2: Medicine → Side Effects (Multi-Label)
**Task:** Given a medicine name and medical condition, predict one or multiple potential side effects.

**Architecture:**
- Input TF-IDF features (Max 10,000)
- 2 Hidden Layers (1024 -> 512) with BatchNorm and Dropout
- BCEWithLogitsLoss for independent thresholding at `P > 0.5`

**Results (Test Set = 420 samples):**
- **Micro F1:** {d_mi_f1:.4f} (Global contribution of all labels)
- **Macro F1:** {d_ma_f1:.4f} (Average per label, unweighted)
- **Samples F1:** {d_sa_f1:.4f} (Average F1 purely calculated per-row)

---

## 4. Architectural Comparative Study

| Feature | Health Pipeline (Model 1) | Medicine Pipeline (Model 2) |
|---------|----------------------------|--------------------------|
| **Input Feature** | Symptom texts combinations | Medicine name, uses, class text |
| **Output Type** | Mutually Exclusive class (1 of 16) | Independent subsets |
| **Loss Function** | Weighted `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| **Activation** | Softmax (implicit in CrossEntropy) | Sigmoid thresholding (`P > 0.5`) |
| **Primary Metric**| Accuracy / Macro-F1 | Sample-F1 / Micro-F1 |
| **Key Challenge** | Extreme data skewness | Label cardinality and sparsity |

*Report auto-generated by Phase 5.*
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"  ✅ Report saved to: {report_path}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    h_metrics = evaluate_health()
    d_metrics = evaluate_medicine()
    generate_report(h_metrics, d_metrics)
    print("\n  [DONE] PHASE 5 COMPLETE — Evaluation report generated.")
