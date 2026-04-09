# DiagnoSense

DiagnoSense is a dual-pipeline medical ML project with:
- Symptom -> Disease prediction (multi-class, source: `symptom_disease_dataset.csv`)
- Medicine context -> Side effects prediction (multi-label)

This version uses leakage-safe splits, tuned thresholding for medicine inference, and benchmark reporting in Phase 5.

## Metric Interpretation

- Health pipeline (`Symptom -> Disease`) is a **single-label multiclass** task, so the primary metric is **Accuracy**.
- Medicine pipeline (`Context -> Side Effects`) is a **multilabel** task, so the primary metric is **Micro F1**.
- For cross-task reference in generated reports, we also log:
  - Health `micro_f1`
  - Medicine `subset_accuracy` (exact-match accuracy)

## Project Structure

```
DiagnoSense/
|-- data/
|   |-- symptom_disease_dataset.csv
|   |-- symptom_disease_train.csv
|   |-- symptom_disease_val.csv
|   |-- symptom_disease_test.csv
|   |-- symptom_disease_metadata.json
|   |-- medicine_dataset.csv
|   |-- medicine_cleaned.csv
|   |-- symptom_disease_X_train/val/test.npz
|   |-- symptom_disease_y_train/val/test.npy
|   |-- medicine_X_train/val/test.npz
|   `-- medicine_y_train/val/test.npy
|-- models/
|   |-- symptom_disease_model.pth
|   |-- medicine_model.pth
|   |-- symptom_disease_tfidf.pkl
|   |-- medicine_tfidf.pkl
|   |-- symptom_disease_label_encoder.pkl
|   |-- medicine_se_mlb.pkl
|   `-- medicine_threshold.json
|-- reports/
|   |-- split_metadata.json
|   |-- training_history.json
|   |-- metrics.json
|   `-- evaluation.md
|-- src/
|   |-- phase2_cleaning.py
|   |-- phase3_feature_engineering.py
|   |-- phase4_training.py
|   `-- phase5_evaluation.py
|-- api/main.py
`-- streamlit_app/app.py
```

## Full Pipeline Run

```bash
python src/phase2_cleaning.py
python src/phase3_feature_engineering.py
python src/phase4_training.py
python src/phase5_evaluation.py
```

## Generated Evaluation Plots

Phase 5 now generates:

- `reports/plots/health_confusion_matrix.png`
- `reports/plots/roc_auc_curves.png`
- `reports/plots/precision_recall_curves.png`
- `reports/plots/training_validation_loss.png`

## Run Streamlit

```bash
streamlit run streamlit_app/app.py
```

## Run API

```bash
python run_api.py
```
