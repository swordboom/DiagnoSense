# DiagnoSense

DiagnoSense is a dual-pipeline medical ML project with:
- Symptom -> Disease prediction (multi-class, source: `symptom_disease_dataset.csv`)
- Medicine context -> Side effects prediction (multi-label)

This version uses leakage-safe splits, tuned thresholding for medicine inference, and benchmark reporting in Phase 5.

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
python src/phase3_feature_engineering.py --task symptom_disease
python src/phase4_training.py --task symptom_disease
python src/phase5_evaluation.py
```

## Run Streamlit

```bash
streamlit run streamlit_app/app.py
```

## Run API

```bash
python run_api.py
```

