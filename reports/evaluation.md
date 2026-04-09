# DiagnoSense Evaluation Report

## Executive Summary
This report summarizes the final evaluation for the two DiagnoSense pipelines after leakage-safe cleaning, deduplication, and group-aware splitting.

- Health pipeline accuracy: **68.67%**
- Health pipeline macro-F1: **68.37%**
- Medicine pipeline micro-F1: **92.37%**
- Medicine pipeline samples-F1: **92.92%**
- Tuned global medicine threshold: **0.90**

## Health Pipeline (Symptom -> Disease)
- Test samples: 3600
- Accuracy: 0.6867
- Macro precision: 0.6835
- Macro recall: 0.6867
- Macro F1: 0.6837

### Classification Report
```text
                   precision    recall  f1-score   support

Allergic_Rhinitis       0.64      0.70      0.67       180
           Anemia       0.78      0.72      0.75       180
        Arthritis       0.80      0.86      0.83       180
           Asthma       0.68      0.64      0.66       180
       Bronchitis       0.60      0.56      0.58       180
          COVID19       0.72      0.79      0.75       180
       Chickenpox       0.61      0.69      0.65       180
      Common_Cold       0.62      0.56      0.59       180
           Dengue       0.66      0.67      0.67       180
         Diabetes       0.85      0.88      0.86       180
              Flu       0.69      0.59      0.64       180
   Food_Poisoning       0.48      0.49      0.49       180
             GERD       0.81      0.87      0.84       180
  Gastroenteritis       0.51      0.37      0.43       180
     Hypertension       0.78      0.81      0.79       180
          Malaria       0.60      0.59      0.60       180
         Migraine       0.86      0.87      0.87       180
        Pneumonia       0.69      0.73      0.71       180
          Typhoid       0.61      0.64      0.63       180
              UTI       0.69      0.69      0.69       180

         accuracy                           0.69      3600
        macro avg       0.68      0.69      0.68      3600
     weighted avg       0.68      0.69      0.68      3600

```

## Medicine Pipeline (Context -> Side Effects)
- Test samples: 32960
- Labels evaluated: 474
- Threshold tuning metric: validation micro-F1
- Selected threshold: 0.90
- Micro precision: 0.9103
- Micro recall: 0.9375
- Micro F1: 0.9237
- Macro F1: 0.8826
- Samples F1: 0.9292

## Overfitting & Bias Diagnostics

### Health Diagnostics
- Train accuracy: 0.7404
- Train macro-F1: 0.7380
- Train-test accuracy gap: 0.0537
- Train-test macro-F1 gap: 0.0543
- Overfitting risk: **high**
- Class recall spread: 0.5056
- Class balance bias risk proxy: **high**
- Lowest recall class: Gastroenteritis (0.3722)
- Dataset realism warning (over-separable): **false**

### Medicine Diagnostics
- Train micro-F1: 0.9421
- Train samples-F1: 0.9470
- Train-test micro-F1 gap: 0.0184
- Train-test samples-F1 gap: 0.0177
- Overfitting risk: **low**
- Label recall spread: 0.8947
- Label balance bias risk proxy: **high**

## Standard Model Benchmarks (Same Splits)

### Health (Multi-Class)
| model | accuracy | macro_f1 | macro_precision | macro_recall |
| --- | --- | --- | --- | --- |
| LogisticRegression | 69.03% | 68.85% | 68.82% | 69.03% |
| DiagnoSense-MLP | 68.67% | 68.37% | 68.35% | 68.67% |
| LinearSVC | 66.31% | 65.93% | 65.81% | 66.31% |
| MultinomialNB | 65.08% | 64.79% | 64.82% | 65.08% |
| DummyMostFrequent | 5.00% | 0.48% | 0.25% | 5.00% |

### Medicine (Multi-Label)
| model | micro_f1 | macro_f1 | samples_f1 | micro_precision | micro_recall | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| DiagnoSense-MLP | 92.37% | 88.26% | 92.92% | 91.03% | 93.75% | 0.9000 |
| OVR-SGDLogLoss | 91.65% | 81.92% | 91.09% | 92.23% | 91.07% | 0.4000 |
| OVR-MultinomialNB | 83.73% | 71.08% | 83.38% | 84.04% | 83.42% | 0.9000 |

## Notes
- Split metadata and leakage checks: `reports/split_metadata.json`
- Data quality summary: `reports/data_quality.json`
- Machine-readable metrics: `reports/metrics.json`
