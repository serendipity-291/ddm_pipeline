import os
import random
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import sys
import argparse
import mlflow
import json

# --- 1. Environment & Hardware Setup ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Professional Trainer | Device: {DEVICE}")

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment("retrain_pipeline")
    MLFLOW_ENABLED = True
except Exception as e:
    print(f"[WARNING] MLflow init failed: {e}. Training will continue without MLflow.")
    MLFLOW_ENABLED = False

PROCESSED_PATH = "data/processed/cwru_features.parquet"
INTERIM_DIR = "data/interim"
MODEL_SAVE_DIR = "data/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 2. Model Architecture ---
# FaultMLP is defined in ml_utils.py (single source of truth).
import sys, os as _os
sys.path.insert(0, _os.path.dirname(__file__))
from ml_utils import FaultMLP


class WDCNN(nn.Module):
    """
    Wide Deep Convolutional Neural Network.
    Industry Standard for CWRU Signal Classification.
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.layer2 = self._make_block(16, 32)
        self.layer3 = self._make_block(32, 64)
        self.layer4 = self._make_block(64, 64)
        self.layer5 = self._make_block(64, 64, pool=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def _make_block(self, in_c, out_c, pool=True):
        layers = [nn.Conv1d(in_c, out_c, 3, padding=1), nn.BatchNorm1d(out_c), nn.ReLU()]
        if pool: layers.append(nn.MaxPool1d(2, 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class FaultLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 3. Custom Dataset for Raw Segments ---
class CWRURawDataset(Dataset):
    def __init__(self, feature_df, transform=None):
        # Implementation note: For 1D CNN, we usually need raw segments.
        # But here we focus on the existing processed features for simplicity 
        # unless specifically instructed to reload the GBs of raw data.
        self.X = torch.FloatTensor(feature_df.drop(columns=['label', 'fault_type']).values)
        self.y = torch.LongTensor(feature_df['label'].values)
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_professional(model, train_loader, val_loader, epochs=30, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0

    # Start MLflow run (best-effort)
    _mlflow_run = None
    try:
        if MLFLOW_ENABLED:
            _mlflow_run = mlflow.start_run(run_name=f"train_{model.__class__.__name__}_{datetime.now().strftime('%m%d_%H%M')}")
            mlflow.log_param("architecture", model.__class__.__name__)
            mlflow.log_param("epochs", epochs)
            if class_weights is not None:
                mlflow.log_param("class_weights", class_weights.tolist())
            # Log dataset version for reproducibility (DVC-style)
            data_version_path = "data/processed/.data_version"
            if os.path.exists(data_version_path):
                data_hash = open(data_version_path).read().strip()
                mlflow.log_param("dataset_version", data_hash)
                mlflow.log_param("dataset_snapshot_uri", f"minio://mlops-data/snapshots/{data_hash}/")
    except Exception as e:
        print(f"[WARNING] MLflow start_run failed: {e}")
        _mlflow_run = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # CNN expects (batch, channels, length). MLP expects (batch, features).
            if isinstance(model, WDCNN): xb = xb.unsqueeze(1)
            elif isinstance(model, FaultLSTM): xb = xb.unsqueeze(-1)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if isinstance(model, WDCNN): xb = xb.unsqueeze(1)
                elif isinstance(model, FaultLSTM): xb = xb.unsqueeze(-1)
                
                out = model(xb)
                _, pred = torch.max(out, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        
        acc = correct / total
        print(f" >> Val Acc: {acc:.2%}")
        try:
            if _mlflow_run:
                mlflow.log_metric("val_accuracy", acc, step=epoch)
        except Exception:
            pass

        if acc > best_acc:
            best_acc = acc
            final_path = os.path.join(MODEL_SAVE_DIR, f"best_{model.__class__.__name__}.pth")
            torch.save(model.state_dict(), final_path)

    try:
        if _mlflow_run:
            mlflow.log_metric("best_val_accuracy", best_acc)
    except Exception:
        pass

    # ── Collect per-sample predictions on validation set (for McNemar's test) ──
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if isinstance(model, WDCNN): xb = xb.unsqueeze(1)
            elif isinstance(model, FaultLSTM): xb = xb.unsqueeze(-1)
            out = model(xb)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    preds_path = os.path.join(MODEL_SAVE_DIR, "test_predictions.csv")
    import csv
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        writer.writerows(zip(all_labels, all_preds))
    print(f"[McNemar] Predictions saved → {preds_path}")

    # ── MLflow Model Registry ──────────────────────────────────────────────────
    try:
        if _mlflow_run:
            # Load best checkpoint back into model
            best_path = os.path.join(MODEL_SAVE_DIR, f"best_{model.__class__.__name__}.pth")
            if os.path.isfile(best_path):
                model.load_state_dict(torch.load(best_path, map_location=DEVICE))
            model.eval()

            # Log the PyTorch model artifact
            mlflow.pytorch.log_model(model.cpu(), "pytorch-model")

            # Log predictions CSV artifact
            mlflow.log_artifact(preds_path, artifact_path="evaluation")
            
            # Log dynamic classes map
            classes_path = "data/processed/classes.json"
            if os.path.exists(classes_path):
                mlflow.log_artifact(classes_path, artifact_path="")

            # Register model in MLflow Model Registry (stage starts as None/Staging)
            run_id = mlflow.active_run().info.run_id
            try:
                mv = mlflow.register_model(
                    model_uri=f"runs:/{run_id}/pytorch-model",
                    name="bearing_fault_classifier",
                )
                print(f"[Registry] Registered: bearing_fault_classifier v{mv.version}")
            except Exception as ex:
                print(f"[Registry] Note (non-fatal): {ex}")

            mlflow.end_run()
    except Exception as e:
        print(f"[WARNING] MLflow model logging failed: {e}")
        try:
            mlflow.end_run()
        except Exception:
            pass

    return best_acc


# --- 5. Main ---
if __name__ == "__main__":
    seed = int(os.environ.get("TRAINING_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df = pd.read_parquet(PROCESSED_PATH)
    if "fault_class" in df.columns:
        class_map_df = df[["label", "fault_class"]].drop_duplicates().sort_values("label")
        classes = {int(r.label): str(r.fault_class) for r in class_map_df.itertuples()}
        with open("data/processed/classes.json", "w", encoding="utf-8") as f:
            json.dump(classes, f)
        print(f"Classes mapping saved: {classes}")

    meta = ['fault_type', 'fault_class', 'fault_diameter', 'load', 'rpm', 'sampling_rate_hz']
    df = df.drop(columns=[c for c in meta if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['label']), df['label'], test_size=0.2, stratify=df['label']
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.joblib"))
    print(f"Scaler saved to {MODEL_SAVE_DIR}/scaler.joblib")

    # Compute class weights (inverse frequency) for weighted CrossEntropyLoss
    from collections import Counter
    counts = Counter(y_train.tolist())
    num_cls = len(counts)
    total   = sum(counts.values())
    class_weights = torch.tensor(
        [total / counts[i] for i in range(num_cls)], dtype=torch.float
    )
    print(f"Class distribution (train): {dict(sorted(counts.items()))}")
    print(f"Class weights: {dict(zip(range(num_cls), class_weights.tolist()))}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
    test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.LongTensor(y_test.values))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    input_dim = X_train.shape[1]
    num_classes = int(df["label"].max()) + 1

    model = FaultMLP(input_dim, num_classes).to(DEVICE)
    best_acc = train_professional(
        model, train_loader, test_loader,
        epochs=args.epochs,
        class_weights=class_weights
    )

    print(f"Training Complete. Best Val Acc: {best_acc:.4f}. Model saved to data/models/best_FaultMLP.pth")
