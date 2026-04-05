# DiagnoSense

GPU-accelerated medical machine learning system with two independent PyTorch inference pipelines and a Streamlit analytics dashboard.

---

## Overview

DiagnoSense trains and serves two clinical prediction models:

| | Health Pipeline | Medicine Pipeline |
|---|---|---|
| **Task** | Symptom → Disease (multi-class) | Medicine → Side Effects (multi-label) |
| **Dataset** | `health_dataset.csv` (12,091 rows) | `medicine_dataset.csv` (248,218 rows) |
| **Labels** | 16 disease classes | 550 side effect labels |
| **Architecture** | PyTorch MLP (512→256) | PyTorch MLP (1024→512) |
| **Loss** | Weighted CrossEntropyLoss | BCEWithLogitsLoss |
| **Test Accuracy** | 99.89% accuracy, 99.97% Macro-F1 | 94.79% Micro-F1, 94.91% Samples-F1 |
| **Hardware** | NVIDIA RTX 3050 (CUDA 12.8) | NVIDIA RTX 3050 (CUDA 12.8) |

---

## Project Structure

```
DiagnoSense/
├── data/
│   ├── health_dataset.csv              # Raw health/symptom data
│   ├── medicine_dataset.csv            # Raw medicine/side-effects data
│   ├── health_cleaned.csv              # Cleaned (generated)
│   └── medicine_cleaned.csv            # Cleaned (generated)
├── models/                             # Trained artifacts (generated)
│   ├── health_model.pth
│   ├── medicine_model.pth
│   ├── health_tfidf.pkl
│   ├── medicine_tfidf.pkl
│   ├── disease_label_encoder.pkl
│   └── medicine_se_mlb.pkl
├── reports/
│   └── evaluation.md                   # Auto-generated evaluation report
├── src/
│   ├── phase2_cleaning.py              # Data cleaning & preprocessing
│   ├── phase3_feature_engineering.py   # TF-IDF vectorization + train/val/test splits
│   ├── phase4_training.py              # PyTorch model training (GPU)
│   └── phase5_evaluation.py            # Evaluation + report generation
├── api/
│   └── main.py                         # FastAPI REST inference server
├── streamlit_app/
│   ├── app.py                          # Streamlit entry point
│   ├── components/
│   │   ├── home.py
│   │   ├── disease.py
│   │   ├── medicine.py
│   │   ├── comparison.py
│   │   └── reports.py
│   └── utils/
│       ├── inference.py                # Prediction logic
│       └── load_models.py              # Cached model loading
└── run_api.py                          # FastAPI launcher
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (CPU fallback available)

### Environment (exact versions used)

| Package | Version |
|---|---|
| Python | 3.13.7 |
| torch | 2.11.0+cu128 |
| scikit-learn | 1.6.1 |
| streamlit | 1.52.2 |
| fastapi | 0.111.0 |
| uvicorn | 0.30.1 |
| pandas | 2.2.3 |
| numpy | 2.2.6 |
| nltk | 3.9.2 |

---

## Setup & Running

### Step 1 — Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install streamlit fastapi uvicorn scikit-learn pandas numpy nltk
```

> For CPU-only (no GPU), install the standard torch:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Step 2 — Download NLTK data (first time only)

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

### Step 3 — Launch the Streamlit dashboard

The trained model weights are already present in `models/`. Simply run:

```bash
streamlit run streamlit_app/app.py
```

Then open **http://localhost:8501** in your browser.

### Step 4 (Optional) — Launch the FastAPI server

```bash
python run_api.py
```

API docs available at **http://localhost:8000/docs**

**Endpoints:**
- `POST /predict/disease` — Predict disease from symptoms
- `POST /predict/side_effects` — Predict medicine side effects

---

## Retraining from Scratch

Only needed if you modify the datasets or model architecture. Run in order:

```bash
# 1. Clean and preprocess raw data
python src/phase2_cleaning.py

# 2. Build TF-IDF features and train/val/test splits
python src/phase3_feature_engineering.py

# 3. Train both PyTorch models (GPU recommended)
python src/phase4_training.py

# 4. Evaluate and regenerate the report
python src/phase5_evaluation.py
```

Training times on RTX 3050 6GB:
- Health model: ~18 epochs, < 1 minute
- Medicine model: ~39 epochs, ~20 minutes

---

## Evaluation Results

Full report: [`reports/evaluation.md`](reports/evaluation.md)

### Health Model (Symptom → Disease)
- Test samples: 1,814
- Accuracy: **99.89%**
- Macro F1: **99.97%**

### Medicine Model (Medicine → Side Effects)
- Test samples: 37,233
- Micro F1: **94.79%**
- Macro F1: **89.44%**
- Samples F1: **94.91%**