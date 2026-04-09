# DiagnoSense Evaluation Report

## Executive Summary
This report summarizes the final evaluation for the two DiagnoSense pipelines after leakage-safe cleaning, deduplication, and group-aware splitting.

- Health pipeline accuracy: **68.17%**
- Health pipeline macro-F1: **67.86%**
- Health ROC-AUC (micro/macro): **0.9755 / 0.9731**
- Medicine pipeline micro-F1: **92.63%**
- Medicine pipeline samples-F1: **93.14%**
- Medicine ROC-AUC (micro/macro): **0.9996 / 0.9992**
- Tuned global medicine threshold: **0.90**

## Primary Metric Selection
- Health primary metric: **Accuracy** (single-label multiclass classification with one ground-truth disease per sample).
- Medicine primary metric: **Micro-F1** (multi-label prediction where each sample can have multiple side effects and labels are imbalanced).
- Cross-task references: health micro-F1 = **68.17%**, medicine subset accuracy (exact-match) = **79.09%**.

## Health Pipeline (Symptom -> Disease)
- Test samples: 3600
- Accuracy: 0.6817
- Micro F1 (reference): 0.6817
- Macro precision: 0.6784
- Macro recall: 0.6823
- Macro F1: 0.6786
- ROC-AUC (micro / macro): 0.9755 / 0.9731
- PR-AUC (micro / macro): 0.7678 / 0.7438

### Classification Report
```text
                   precision    recall  f1-score   support

Allergic_Rhinitis       0.72      0.71      0.71       179
           Anemia       0.78      0.72      0.75       188
        Arthritis       0.78      0.85      0.82       167
           Asthma       0.64      0.68      0.66       182
       Bronchitis       0.56      0.52      0.54       181
          COVID19       0.70      0.78      0.74       180
       Chickenpox       0.65      0.64      0.64       168
      Common_Cold       0.66      0.64      0.65       196
           Dengue       0.71      0.72      0.72       177
         Diabetes       0.87      0.84      0.85       195
              Flu       0.64      0.60      0.62       178
   Food_Poisoning       0.52      0.58      0.55       188
             GERD       0.83      0.83      0.83       174
  Gastroenteritis       0.52      0.34      0.41       189
     Hypertension       0.73      0.84      0.78       191
          Malaria       0.54      0.54      0.54       167
         Migraine       0.80      0.90      0.84       174
        Pneumonia       0.64      0.63      0.63       187
          Typhoid       0.61      0.63      0.62       172
              UTI       0.69      0.67      0.68       167

         accuracy                           0.68      3600
        macro avg       0.68      0.68      0.68      3600
     weighted avg       0.68      0.68      0.68      3600

```

## Medicine Pipeline (Context -> Side Effects)
- Test samples: 32960
- Labels evaluated: 474
- Threshold tuning metric: validation micro-F1
- Selected threshold: 0.90
- Subset accuracy (exact match, reference): 0.7909
- Micro precision: 0.9143
- Micro recall: 0.9386
- Micro F1: 0.9263
- Macro F1: 0.8854
- Samples F1: 0.9314
- ROC-AUC (micro / macro): 0.9996 / 0.9992
- PR-AUC (micro / macro): 0.9808 / 0.9341
- Labels used for curve metrics: 474

## Overfitting & Bias Diagnostics

### Health Diagnostics
- Train accuracy: 0.7605
- Train macro-F1: 0.7586
- Train-test accuracy gap: 0.0788
- Train-test macro-F1 gap: 0.0800
- Overfitting risk: **high**
- Class recall spread: 0.5579
- Class balance bias risk proxy: **high**
- Lowest recall class: Gastroenteritis (0.3386)
- Dataset realism warning (over-separable): **false**

### Medicine Diagnostics
- Train micro-F1: 0.9442
- Train samples-F1: 0.9487
- Train-test micro-F1 gap: 0.0179
- Train-test samples-F1 gap: 0.0173
- Overfitting risk: **low**
- Label recall spread: 0.8947
- Label balance bias risk proxy: **high**

## Standard Model Benchmarks (Same Splits)

### Health (Multi-Class)
| model | accuracy | macro_f1 | macro_precision | macro_recall |
| --- | --- | --- | --- | --- |
| LogisticRegression | 68.53% | 68.28% | 68.17% | 68.67% |
| DiagnoSense-MLP | 68.17% | 67.86% | 67.84% | 68.23% |
| LinearSVC | 66.03% | 65.57% | 65.40% | 66.15% |
| MultinomialNB | 64.81% | 64.32% | 64.23% | 64.92% |
| DummyMostFrequent | 5.22% | 0.50% | 0.26% | 5.00% |

### Medicine (Multi-Label)
| model | micro_f1 | macro_f1 | samples_f1 | micro_precision | micro_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| DiagnoSense-MLP | 92.63% | 88.54% | 93.14% | 91.43% | 93.86% | 0.9000 |
| OVR-SGDLogLoss | 91.65% | 81.92% | 91.09% | 92.23% | 91.07% | 0.4000 |
| OVR-MultinomialNB | 83.73% | 71.08% | 83.38% | 84.04% | 83.42% | 0.9000 |

## Visualization Artifacts
- Health confusion matrix: `./plots/health_confusion_matrix.png`
- ROC curve + AUC: `./plots/roc_auc_curves.png`
- Precision-recall curve: `./plots/precision_recall_curves.png`
- Training vs validation loss: `./plots/training_validation_loss.png`

## Notes
- Split metadata and leakage checks: `reports/split_metadata.json`
- Data quality summary: `reports/data_quality.json`
- Machine-readable metrics: `reports/metrics.json`
