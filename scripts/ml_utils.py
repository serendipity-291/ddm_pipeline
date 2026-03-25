"""
ml_utils.py — Shared ML utilities for the DDM pipeline.

Provides:
  - FaultMLP: model architecture (single source of truth)
  - CLASS_NAMES: label map shared by training, simulator, and inference
  - extract_features(): real-time per-window feature extraction
  - load_model_from_registry(): loads model from MLflow @production alias
    with automatic fallback to local .pth file
"""
import os
import numpy as np
import scipy.stats
from scipy.fft import fft
import pywt
import torch
import torch.nn as nn
import joblib

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES    = {0: "B", 1: "Normal"}
MODEL_REGISTRY = "bearing_fault_classifier"
PROD_ALIAS     = "production"
LOCAL_MODEL_PATH  = "data/models/best_FaultMLP.pth"
LOCAL_SCALER_PATH = "data/models/scaler.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Architecture (single source of truth) ────────────────────────────────
class FaultMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Feature Extraction (synchronized with feature_engineering.py) ──────────────
def extract_features(window: np.ndarray, fs: int = 12000) -> np.ndarray:
    """Extract 22 features from a raw vibration window (shape: [N])."""
    window = np.array(window, dtype=float).copy()
    abs_w  = np.abs(window)
    rms    = np.sqrt(np.mean(window ** 2))
    peak   = np.max(abs_w)
    mean_abs = np.mean(abs_w)

    features = [
        np.mean(window),
        np.std(window),
        peak,
        np.min(window),
        rms,
        scipy.stats.kurtosis(window),
        scipy.stats.skew(window),
        peak - np.min(window),                           # peak-to-peak
        peak / rms if rms != 0 else 0,                   # crest factor
        rms / mean_abs if mean_abs != 0 else 0,          # shape factor
        peak / mean_abs if mean_abs != 0 else 0,         # impulse factor
    ]

    # FFT spectral features (3)
    n = len(window)
    f_coeffs = fft(window)
    psd   = np.abs(f_coeffs[: n // 2]) ** 2 / n
    freqs = np.linspace(0, fs / 2, n // 2)
    total_energy = np.sum(psd)
    features.extend([
        total_energy,
        np.sum(freqs * psd) / total_energy if total_energy > 0 else 0,  # spectral centroid
        freqs[np.argmax(psd)] if len(psd) > 0 else 0,                  # dominant freq
    ])

    # Wavelet Packet Decomposition energy (8 bands at level 3)
    wp      = pywt.WaveletPacket(data=window, wavelet="db4", mode="symmetric", maxlevel=3)
    nodes   = [node.path for node in wp.get_level(3, "freq")]
    energies = [np.sum(np.square(wp[node].data)) for node in nodes]
    te = np.sum(energies)
    features.extend([e / te if te > 0 else 0 for e in energies])

    return np.array(features).reshape(1, -1)  # shape: (1, 22)


# ── Model + Scaler Loading ─────────────────────────────────────────────────────
def load_scaler(path: str = LOCAL_SCALER_PATH):
    """Load the StandardScaler used during training."""
    return joblib.load(path)


def load_model_from_registry(
    mlflow_url: str | None = None,
    fallback_pth: str = LOCAL_MODEL_PATH,
) -> tuple:
    """
    Load FaultMLP from MLflow Model Registry (@production alias).
    Also loads `classes.json` from artifacts.
    Falls back to local .pth if Registry is unavailable or model not registered.

    Returns:
        (model, class_mapping, source) 
    """
    mlflow_url = mlflow_url or os.environ.get("MLFLOW_URL", "http://mlflow:5000")
    import json

    def _load_local_classes():
        classes_path = "data/processed/classes.json"
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = json.load(f)
            return {int(k): str(v) for k, v in classes.items()}
        return CLASS_NAMES.copy()

    # ── Try MLflow Registry ────────────────────────────────────────────────────
    try:
        import mlflow
        import mlflow.pytorch
        mlflow.set_tracking_uri(mlflow_url)
        
        client = mlflow.MlflowClient()
        version = client.get_model_version_by_alias(MODEL_REGISTRY, PROD_ALIAS)
        run_id = version.run_id
        
        try:
            local_class_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="classes.json")
            with open(local_class_path, "r") as f:
                class_mapping = json.load(f)
                class_mapping = {int(k): str(v) for k, v in class_mapping.items()}
        except Exception as e:
            print(f"[ml_utils] Could not download classes.json: {e}")
            class_mapping = _load_local_classes()

        model_uri = f"models:/{MODEL_REGISTRY}@{PROD_ALIAS}"
        model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        model.eval()
        print(f"[ml_utils] Loaded model from Registry: {model_uri}")
        return model, class_mapping, "mlflow_registry"
    except Exception as e:
        print(f"[ml_utils] Registry load failed ({e}), falling back to local .pth")

    # ── Fallback: local .pth ───────────────────────────────────────────────────
    try:
        class_mapping = _load_local_classes()
    except Exception:
        class_mapping = CLASS_NAMES.copy()
        
    try:
        scaler = joblib.load(LOCAL_SCALER_PATH)
        input_dim = scaler.mean_.shape[0]
    except Exception:
        input_dim = 22
        
    num_classes = len(class_mapping)
    model     = FaultMLP(input_dim, num_classes=num_classes).to(DEVICE)
    if os.path.exists(fallback_pth):
        model.load_state_dict(torch.load(fallback_pth, map_location=DEVICE))
        print(f"[ml_utils] Loaded model from local .pth: {fallback_pth}")
    else:
        print(f"[ml_utils] WARNING: No model found at {fallback_pth}, using untrained weights")
    model.eval()
    return model, class_mapping, "local_pth"


def predict_window(
    window: np.ndarray,
    model: nn.Module,
    scaler,
    class_mapping: dict,
) -> dict:
    """
    Run one inference pass on a raw vibration window.
    """
    feats  = extract_features(window)
    # Extracts scalar features directly from NumPy array before scaling
    rms_val = float(feats.flatten()[4])
    kurtosis_val = float(feats.flatten()[5])
    
    # Use named DataFrame if scaler was fitted on one — avoids sklearn UserWarning
    if hasattr(scaler, "feature_names_in_"):
        import pandas as pd
        feats = pd.DataFrame(feats, columns=scaler.feature_names_in_)
    scaled = scaler.transform(feats)

    with torch.no_grad():
        tensor = torch.FloatTensor(scaled).to(DEVICE)
        output = model(tensor)
        prob   = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)

    pred_idx  = int(pred.item())
    probs_list = prob.cpu().numpy().tolist()[0]
    
    num_output_classes = len(class_mapping)

    return {
        "predicted_class": class_mapping.get(pred_idx, str(pred_idx)),
        "predicted_label": pred_idx,
        "confidence":      round(float(conf.item()), 6),
        "probabilities":   {class_mapping.get(i, str(i)): round(probs_list[i], 6) for i in range(num_output_classes)},
        "rms":             rms_val,
        "kurtosis":        kurtosis_val
    }
