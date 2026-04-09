"""
PHASE 4: Model Training (PyTorch)
=================================
Trains the health and medicine models with regularization and early stopping.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIG
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS_HEALTH = 40
EPOCHS_MEDICINE = 35
PATIENCE = 7
MAX_GRAD_NORM = 1.0
SYMPTOM_DISEASE_PREFIX = "symptom_disease"


def print_device_banner() -> None:
    print("=" * 70)
    print(f"  PHASE 4: TRAINING ON DEVICE => {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU => {torch.cuda.get_device_name(0)}")
    print("=" * 70)


# ============================================================
# DATASET WRAPPER
# ============================================================
class SparseDataset(Dataset):
    def __init__(self, X_sparse: sp.csr_matrix, y_array: np.ndarray, task: str = "multiclass"):
        self.X = X_sparse
        self.y = y_array
        self.task = task

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_row = self.X[idx].toarray().astype(np.float32).ravel()
        x_tensor = torch.from_numpy(x_row)

        if self.task == "multiclass":
            y_tensor = torch.tensor(int(self.y[idx]), dtype=torch.long)
        else:
            y_tensor = torch.from_numpy(self.y[idx].astype(np.float32))

        return x_tensor, y_tensor


# ============================================================
# MODELS
# ============================================================
class HealthModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MedicineModel(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(384, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# TRAINER
# ============================================================
@dataclass
class EarlyStopping:
    patience: int
    path: Path
    best_loss: float = float("inf")
    counter: int = 0
    early_stop: bool = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def _train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    save_path: Path,
) -> List[Dict[str, float]]:
    stopper = EarlyStopping(patience=PATIENCE, path=save_path)
    history: List[Dict[str, float]] = []

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        stopper.step(val_loss, model)

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"    Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if stopper.early_stop:
            print(f"    Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    return history


def _build_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# PIPELINES
# ============================================================
def train_health_pipeline() -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("  TRAINING DATASET 1: Symptom -> Disease")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_train.npz")
    X_val = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_val.npz")
    y_train = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_train.npy")
    y_val = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_val.npy")

    with (MODELS_DIR / "symptom_disease_class_weights.pkl").open("rb") as handle:
        class_weights: Dict[int, float] = pickle.load(handle)

    num_classes = int(max(class_weights.keys())) + 1
    weights = np.ones(num_classes, dtype=np.float32)
    for cls_id, weight in class_weights.items():
        weights[int(cls_id)] = float(weight)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    train_dataset = SparseDataset(X_train, y_train, task="multiclass")
    val_dataset = SparseDataset(X_val, y_val, task="multiclass")

    train_loader = _build_loader(train_dataset, shuffle=True)
    val_loader = _build_loader(val_dataset, shuffle=False)

    model = HealthModel(input_dim=X_train.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    model_path = MODELS_DIR / "symptom_disease_model.pth"
    history = _train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS_HEALTH,
        save_path=model_path,
    )

    print("  [DONE] Symptom-disease model training complete")
    return {
        "epochs_ran": len(history),
        "best_val_loss": min(h["val_loss"] for h in history),
        "history": history,
    }


def train_medicine_pipeline() -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("  TRAINING DATASET 2: Medicine (Context -> Side Effects)")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / "medicine_X_train.npz")
    X_val = sp.load_npz(DATA_DIR / "medicine_X_val.npz")
    y_train = np.load(DATA_DIR / "medicine_y_train.npy")
    y_val = np.load(DATA_DIR / "medicine_y_val.npy")

    num_labels = y_train.shape[1]

    positive_counts = y_train.sum(axis=0).astype(np.float32)
    negative_counts = (len(y_train) - positive_counts).astype(np.float32)
    pos_weight = (negative_counts / np.maximum(positive_counts, 1.0)).astype(np.float32)
    pos_weight = np.clip(pos_weight, 1.0, 25.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

    train_dataset = SparseDataset(X_train, y_train, task="multilabel")
    val_dataset = SparseDataset(X_val, y_val, task="multilabel")

    train_loader = _build_loader(train_dataset, shuffle=True)
    val_loader = _build_loader(val_dataset, shuffle=False)

    model = MedicineModel(input_dim=X_train.shape[1], num_labels=num_labels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    model_path = MODELS_DIR / "medicine_model.pth"
    history = _train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS_MEDICINE,
        save_path=model_path,
    )

    print("  [DONE] Medicine model training complete")
    return {
        "epochs_ran": len(history),
        "best_val_loss": min(h["val_loss"] for h in history),
        "history": history,
    }


def _load_existing_history(path: Path) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_training_history(
    symptom_disease_info: Dict[str, object] | None = None,
    medicine_info: Dict[str, object] | None = None,
) -> None:
    path = REPORTS_DIR / "training_history.json"
    out = _load_existing_history(path)
    if symptom_disease_info is not None:
        out["symptom_disease"] = symptom_disease_info
    if medicine_info is not None:
        out["medicine"] = medicine_info
    with path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"  Saved training history: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiagnoSense Phase 4 model training")
    parser.add_argument(
        "--task",
        choices=["all", "symptom_disease", "medicine"],
        default="all",
        help="Choose which model pipeline to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_health = args.task in ("all", "symptom_disease")
    run_medicine = args.task in ("all", "medicine")

    print_device_banner()
    health_result = train_health_pipeline() if run_health else None
    medicine_result = train_medicine_pipeline() if run_medicine else None
    save_training_history(symptom_disease_info=health_result, medicine_info=medicine_result)
    print("\n  [DONE] PHASE 4 COMPLETE")

