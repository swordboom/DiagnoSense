"""
PHASE 4: Model Training (PyTorch GPU Accelerated)
===================================================
DiagnoSense - ML Pipeline
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG & SETUP
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"============================================================")
print(f"  PHASE 4: TRAINING ON DEVICE => {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU => {torch.cuda.get_device_name(0)}")
print(f"============================================================")

BATCH_SIZE = 128
EPOCHS_HEALTH = 50
EPOCHS_DRUG = 50
PATIENCE = 7

# ============================================================
# PYTORCH DATASETS
# ============================================================
class SparseDataset(Dataset):
    def __init__(self, X_sparse, y_array, task='multiclass'):
        self.X = X_sparse
        self.y = y_array
        self.task = task
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        # Convert sparse row to dense tensor
        x_row = self.X[idx].toarray().flatten()
        x_tensor = torch.FloatTensor(x_row)
        
        if self.task == 'multiclass':
            y_tensor = torch.LongTensor([self.y[idx]])[0]
        else:
            y_tensor = torch.FloatTensor(self.y[idx])
            
        return x_tensor, y_tensor

# ============================================================
# PYTORCH MODELS
# ============================================================
class HealthModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HealthModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class MedicineModel(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MedicineModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, x):
        return self.net(x)

# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pth'):
        self.patience = patience
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path):
    early_stopping = EarlyStopping(patience=PATIENCE, path=save_path)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"    Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model

# ============================================================
# PROCESSING PIPELINES
# ============================================================
def train_health_pipeline():
    print("\n" + "=" * 70)
    print("  TRAINING DATASET 1: Health (Symptom → Disease)")
    print("=" * 70)
    
    X_train = sp.load_npz(os.path.join(DATA_DIR, "health_X_train.npz"))
    X_val = sp.load_npz(os.path.join(DATA_DIR, "health_X_val.npz"))
    
    y_train = np.load(os.path.join(DATA_DIR, "health_y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "health_y_val.npy"))
    
    with open(os.path.join(MODELS_DIR, "health_class_weights.pkl"), "rb") as f:
        weight_dict = pickle.load(f)
        
    num_classes = len(weight_dict)
    weights = torch.FloatTensor([weight_dict[i] for i in range(num_classes)]).to(DEVICE)
    
    train_dataset = SparseDataset(X_train, y_train, task='multiclass')
    val_dataset = SparseDataset(X_val, y_val, task='multiclass')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))
    
    model = HealthModel(input_dim=X_train.shape[1], num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_path = os.path.join(MODELS_DIR, "health_model.pth")
    print(f"  Training started (Input Dims: {X_train.shape[1]})")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS_HEALTH, model_path)
    print("  ✅ Health Model Training Complete")

def train_medicine_pipeline():
    print("\n" + "=" * 70)
    print("  TRAINING DATASET 2: Medicine Side Effects (Multi-label)")
    print("=" * 70)
    
    X_train = sp.load_npz(os.path.join(DATA_DIR, "medicine_X_train.npz"))
    X_val = sp.load_npz(os.path.join(DATA_DIR, "medicine_X_val.npz"))
    
    y_train = np.load(os.path.join(DATA_DIR, "medicine_y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "medicine_y_val.npy"))
    
    num_labels = y_train.shape[1]
    
    train_dataset = SparseDataset(X_train, y_train, task='multilabel')
    val_dataset = SparseDataset(X_val, y_val, task='multilabel')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE.type=='cuda'))
    
    model = MedicineModel(input_dim=X_train.shape[1], num_labels=num_labels).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model_path = os.path.join(MODELS_DIR, "medicine_model.pth")
    print(f"  Training started (Input Dims: {X_train.shape[1]}, Labels: {num_labels})")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS_DRUG, model_path)
    print("  ✅ Medicine Side Effects Model Training Complete")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_health_pipeline()
    train_medicine_pipeline()
    print("\n  [DONE] PHASE 4 COMPLETE — Models saved to models/ directory.")
