from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
import subprocess
import requests
import numpy as np

# Shared ML utilities
sys.path.insert(0, os.path.dirname(__file__) if os.path.dirname(__file__) else ".")
from ml_utils import load_model_from_registry, load_scaler, predict_window

# ── Global inference state (loaded at startup) ────────────────────────────────
_model        = None
_scaler       = None
_class_mapping= None
_model_source = "not_loaded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load @production model from MLflow Registry on startup."""
    global _model, _scaler, _class_mapping, _model_source
    try:
        _model, _class_mapping, _model_source = load_model_from_registry()
        _scaler = load_scaler()
        print(f"[startup] Inference model ready — source: {_model_source}")
    except Exception as e:
        print(f"[startup] WARNING: Model not loaded ({e}). /predict will return 503.")
    yield  # app runs here


app = FastAPI(title="CWRU Fault Simulation API", lifespan=lifespan)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]
)

# Use sys.executable → resolves correctly in Docker and on Windows host
PYTHON_PATH = sys.executable

# InfluxDB config — env var from docker-compose
INFLUXDB_URL   = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = "UziSGCgplwTUlHTdRiHWPIwFasDqSPbKxqfx5C_I7rsZuICEvvgAbRD3L1_a8U4R48f7mmJs9QMxX0dmjjNdEg=="
INFLUXDB_ORG   = "my-org"
INFLUXDB_BUCKET = "bearing_data"
MLFLOW_URL     = os.environ.get("MLFLOW_URL", "http://mlflow:5000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIGGER_FILE = "data/fault_trigger.txt"
SIM_SCRIPT   = "scripts/sensor_simulator.py"
sim_process  = None
VALID_FAULT_MODES = ["B"]


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "sim_running":  sim_process is not None and sim_process.poll() is None,
        "model_source": _model_source,
        "model_ready":  _model is not None,
    }


@app.post("/sim/start")
def start_sim():
    global sim_process
    if sim_process and sim_process.poll() is None:
        return {"message": "Simulation already running"}
    os.makedirs("data", exist_ok=True)
    with open(TRIGGER_FILE, 'w') as f:
        f.write("normal")
    env = {**os.environ, "INFLUXDB_URL": INFLUXDB_URL}
    sim_process = subprocess.Popen([PYTHON_PATH, SIM_SCRIPT], env=env)
    return {"message": "Simulation started", "pid": sim_process.pid}


@app.post("/sim/stop")
def stop_sim():
    global sim_process
    if sim_process and sim_process.poll() is None:
        sim_process.terminate()
        sim_process = None
        return {"message": "Simulation stopped"}
    return {"message": "Simulation is not running"}


@app.post("/sim/inject")
def inject_fault(fault_type: str = "B"):
    if fault_type not in VALID_FAULT_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid fault_type. Valid: {VALID_FAULT_MODES}")
    os.makedirs("data", exist_ok=True)
    with open(TRIGGER_FILE, 'w') as f:
        f.write(fault_type)
    return {"message": f"Fault injected: {fault_type}", "mode": fault_type}


@app.post("/sim/reset")
def reset_fault():
    os.makedirs("data", exist_ok=True)
    with open(TRIGGER_FILE, 'w') as f:
        f.write("reset_all")   # simulator will set mode=normal AND drift_active=False
    return {"message": "System reset to normal (fault + drift cleared)", "mode": "normal"}


@app.post("/sim/drift")
def toggle_drift(active: bool = True):
    os.makedirs("data", exist_ok=True)
    with open(TRIGGER_FILE, 'w') as f:
        f.write("drift_on" if active else "drift_off")
    return {"message": f"Drift {'enabled' if active else 'disabled'}"}


@app.post("/sim/clear_data")
def clear_data():
    """Purge all data from the InfluxDB bucket. InfluxDB 2.x requires a predicate."""
    # Stop simulation first if it's running
    global sim_process
    if sim_process:
        sim_process.terminate()
        sim_process = None

    url = f"{INFLUXDB_URL}/api/v2/delete?org={INFLUXDB_ORG}&bucket={INFLUXDB_BUCKET}"
    headers = {"Authorization": f"Token {INFLUXDB_TOKEN}", "Content-Type": "application/json"}
    
    for measurement in ["inference_results", "system_alerts"]:
        body = {
            "start": "1970-01-01T00:00:00Z",
            "stop":  "2099-01-01T00:00:00Z",
            "predicate": f'_measurement="{measurement}"'
        }
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 204:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"InfluxDB error {response.status_code}: {response.text}"
            )
            
    return {"message": "Simulation stopped and all data cleared"}


@app.get("/drift/status")
def drift_status():
    """Query the latest drift alert state from InfluxDB system_alerts."""
    flux = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -10m)
      |> filter(fn: (r) => r._measurement == "system_alerts")
      |> filter(fn: (r) => r._field == "is_drifted" or r._field == "drift_score")
      |> last()
    """
    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        tables = client.query_api().query(flux, org=INFLUXDB_ORG)
        result = {"is_drifted": False, "drift_score": None, "last_check": None}
        for table in tables:
            for record in table.records:
                if record.get_field() == "is_drifted":
                    result["is_drifted"] = bool(record.get_value())
                    result["last_check"] = record.get_time().isoformat()
                elif record.get_field() == "drift_score":
                    result["drift_score"] = round(float(record.get_value()), 4)
        client.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"InfluxDB query error: {str(e)}")


@app.post("/train/feature_engineering")
def trigger_feature_engineering():
    """Run the feature engineering script inside the api_server container. Defaults to retrain mode."""
    try:
        # Runs data/interim + MinIO drift data -> data/processed/cwru_features.parquet
        result = subprocess.run(
            [PYTHON_PATH, "scripts/feature_engineering.py", "--mode", "retrain"],
            capture_output=True, text=True, check=True
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e.stderr}")


@app.post("/train/model_training")
def trigger_model_training(epochs: int = 50):
    """Run the model training script inside the api_server container."""
    try:
        # Runs data/processed -> data/models/best_FaultMLP.pth
        # Pass epochs as argument
        result = subprocess.run(
            [PYTHON_PATH, "scripts/model_training.py", "--epochs", str(epochs)],
            capture_output=True, text=True, check=True
        )
        # Parse output for accuracy if possible
        val_acc = None
        for line in result.stdout.splitlines():
            if "Val Acc:" in line:
                try:
                    val_acc = float(line.split("Val Acc:")[-1].strip().replace("%", "")) / 100
                except:
                    pass

        return {
            "status": "success",
            "val_accuracy": val_acc,
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e.stderr}")


@app.post("/train/promote")
def promote_model(run_id: str = None):
    """
    Champion vs Challenger promotion with McNemar's statistical significance test.

    A new model (challenger) is promoted to Production ONLY if:
      1. Absolute accuracy improvement > MIN_IMPROVEMENT_THRESHOLD (0.5%)
      2. McNemar's test p-value < P_VALUE_THRESHOLD (0.05)
         — meaning the improvement is statistically significant, not noise.

    Uses MLflow Model Registry stage transitions (None → Production).
    The previous Production version is automatically archived.
    """
    global _model, _scaler, _class_mapping, _model_source
    import mlflow
    from mlflow.tracking import MlflowClient
    import numpy as np
    import csv
    from scipy.stats import chi2

    MIN_IMPROVEMENT = 0.005   # 0.5% minimum absolute gain to consider promotion
    P_VALUE_THRESH  = 0.05    # significance level

    def mcnemar_test(preds_challenger, preds_champion, y_true):
        """
        McNemar's test for paired classifier comparison.
        Returns p-value: low p → challenger is significantly better.
        """
        champ_ok  = np.array(preds_champion)   == np.array(y_true)
        chall_ok  = np.array(preds_challenger)  == np.array(y_true)
        b = int(np.sum( champ_ok & ~chall_ok))   # champion right, challenger wrong
        c = int(np.sum(~champ_ok &  chall_ok))   # champion wrong, challenger right
        if b + c == 0:
            return 1.0   # identical error patterns
        # McNemar statistic with continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - chi2.cdf(stat, df=1))

    def load_predictions(artifact_path):
        """Load y_true/y_pred from a local CSV path."""
        y_true, y_pred = [], []
        try:
            with open(artifact_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    y_true.append(int(row["y_true"]))
                    y_pred.append(int(row["y_pred"]))
        except Exception:
            pass
        return y_true, y_pred

    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = MlflowClient()

        # ── 1. Find challenger: latest run with best_val_accuracy ──────────────
        # Resolve experiment ID (experiment_names not supported in mlflow 3.x)
        exp = mlflow.get_experiment_by_name("retrain_pipeline")
        exp_ids = [exp.experiment_id] if exp else []

        challenger_runs = mlflow.search_runs(
            experiment_ids=exp_ids,
            filter_string="metrics.best_val_accuracy > 0",
            max_results=1,
            order_by=["start_time DESC"],
            output_format="list",
        ) if exp_ids else []
        if not challenger_runs:
            return {"decision": "no_challenger", "reason": "No completed training runs found"}

        challenger_run  = challenger_runs[0]
        challenger_acc  = challenger_run.data.metrics.get("best_val_accuracy", 0.0)
        challenger_id   = challenger_run.info.run_id

        # ── 2. Find current Production model via MLflow 3.x Alias API ─────────
        MODEL_NAME       = "bearing_fault_classifier"
        PROD_ALIAS       = "production"

        champion_acc     = 0.0
        champion_preds   = []
        champion_true    = []
        champion_version = None
        try:
            mv_champion = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
            champion_version = mv_champion.version
            champion_run     = client.get_run(mv_champion.run_id)
            champion_acc     = champion_run.data.metrics.get("best_val_accuracy", 0.0)
            # Download champion predictions for McNemar test
            try:
                tmp_dir    = client.download_artifacts(mv_champion.run_id, "evaluation")
                preds_file = os.path.join(tmp_dir, "test_predictions.csv")
                champion_true, champion_preds = load_predictions(preds_file)
            except Exception:
                pass
        except Exception:
            pass  # no production alias yet

        # ── 3. Compute statistical significance ────────────────────────────────
        acc_delta = challenger_acc - champion_acc

        # Load challenger predictions
        challenger_preds_path = "data/models/test_predictions.csv"
        y_true, challenger_preds = load_predictions(challenger_preds_path)

        # Run McNemar's test (only if we have both sets of predictions)
        p_value = 1.0
        mcnemar_available = (len(champion_preds) > 0 and len(challenger_preds) > 0
                             and len(champion_preds) == len(challenger_preds))
        if mcnemar_available:
            p_value = mcnemar_test(challenger_preds, champion_preds, y_true or champion_true)

        # ── 4. Promotion decision ──────────────────────────────────────────────
        decision = "kept_current"
        reason   = ""

        if acc_delta < MIN_IMPROVEMENT:
            reason = (f"Δacc={acc_delta:+.4f} is below minimum threshold "
                      f"({MIN_IMPROVEMENT:.1%}). Not worth the risk.")
        elif mcnemar_available and p_value >= P_VALUE_THRESH:
            reason = (f"Improvement not statistically significant "
                      f"(p={p_value:.4f} ≥ {P_VALUE_THRESH}). Could be noise.")
        else:
            decision = "promoted"
            if not mcnemar_available:
                reason = "No champion predictions available — promoted on accuracy delta alone."

        # ── 5. Execute promotion via MLflow 3.x Alias ─────────────────────────
        new_version = None
        if decision == "promoted":
            # Register challenger (idempotent if already registered from training)
            try:
                mv = mlflow.register_model(
                    model_uri=f"runs:/{challenger_id}/pytorch-model",
                    name=MODEL_NAME,
                )
                new_version = mv.version
            except Exception:
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                matching  = [v for v in versions if v.run_id == challenger_id]
                new_version = matching[0].version if matching else None

            if new_version:
                # Tag old champion as @prev_production before promoting challenger
                if champion_version:
                    try:
                        client.set_registered_model_alias(MODEL_NAME, "prev_production", champion_version)
                    except Exception:
                        pass
                # Point @production alias at new version
                client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, new_version)

        reload_result = None
        if decision == "promoted" and new_version:
            try:
                _model, _class_mapping, _model_source = load_model_from_registry()
                _scaler = load_scaler()
                reload_result = {"status": "reloaded", "model_source": _model_source}
            except Exception as reload_err:
                reload_result = {"status": "reload_failed", "error": str(reload_err)}

        response = {
            "status":            decision,
            "decision":          decision,
            "challenger_acc":    round(challenger_acc, 6),
            "champion_acc":      round(champion_acc, 6),
            "acc_delta":         round(acc_delta, 6),
            "p_value":           round(p_value, 4) if mcnemar_available else "N/A (no champion predictions)",
            "mcnemar_available": mcnemar_available,
        }
        if reason:
            response["reason"] = reason
        if new_version:
            response["registry_version"] = new_version
            response["registry_alias"]   = f"models:/{MODEL_NAME}@{PROD_ALIAS}"
        if reload_result is not None:
            response["reload"] = reload_result

        return response

    except Exception as e:
        # If MLflow is unavailable, skip promotion gracefully
        return {"decision": "skipped", "reason": str(e)}


# ── Inference Endpoints ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    window:          List[float]            # raw vibration samples (typically 2048)
    # Optional simulation metadata — provided by sensor_simulator to enrich InfluxDB
    injected_mode:   str  = "unknown"       # "normal", "B", "IR", "OR"
    drift_active:    bool = False
    drift_metric:    float = 0.0            # 0 (clean) → 35 (heavy drift)
    throughput_wps:  float = 0.0            # windows/sec from simulator
    push_to_influxdb: bool = True           # False → pure inference, no InfluxDB write


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run FaultMLP inference on a raw vibration window and (optionally) push to InfluxDB.

    When called by sensor_simulator: push_to_influxdb=True (default).
      → Inference result + metadata are written to InfluxDB as 'inference_results'.
    When called externally for ad-hoc inference: push_to_influxdb=False.
      → Pure inference response only.

    Input:  { "window": [<float×2048>], "injected_mode": "IR", "drift_active": false, ... }
    Output: { "predicted_class": "IR", "confidence": 0.987, "probabilities": {...} }
    """
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /model/reload first.")
    if len(req.window) < 64:
        raise HTTPException(status_code=400, detail="Window too short. Minimum 64 samples.")

    import time as _time
    t_inf  = _time.perf_counter()
    result = predict_window(np.array(req.window), _model, _scaler, _class_mapping)
    latency_ms = (_time.perf_counter() - t_inf) * 1000

    result["model_source"]  = _model_source
    result["latency_ms"]    = round(latency_ms, 3)

    # ── Write to InfluxDB (when called from simulator or any push-enabled client) ──
    if req.push_to_influxdb:
        try:
            from influxdb_client import InfluxDBClient, Point, WritePrecision
            from influxdb_client.client.write_api import SYNCHRONOUS
            client    = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
            write_api = client.write_api(write_options=SYNCHRONOUS)
            point = (
                Point("inference_results")
                .tag("sensor",        "main_bearing")
                .tag("injected_mode", req.injected_mode)
                .tag("drift_active",  str(req.drift_active))
                .tag("model_source",  _model_source)
                .field("predicted_class",      result["predicted_label"])
                .field("confidence",           result["confidence"])
                .field("drift_metric",         req.drift_metric)
                .field("inference_latency_ms", latency_ms)
                .field("throughput_wps",       req.throughput_wps)
                .field("rms",                  result["rms"])
                .field("kurtosis",             result["kurtosis"])
                .time(_time.time_ns(), WritePrecision.NS)
            )
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            client.close()
            result["influxdb_written"] = True
        except Exception as e:
            result["influxdb_written"] = False
            result["influxdb_error"]   = str(e)

    return result


@app.post("/model/reload")
def reload_model():
    """
    Hot-reload the @production model from MLflow Registry.
    Call this after /train/promote to update the inference endpoint without restart.
    """
    global _model, _scaler, _class_mapping, _model_source
    try:
        _model, _class_mapping, _model_source = load_model_from_registry()
        _scaler = load_scaler()
        return {"status": "reloaded", "model_source": _model_source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {e}")


@app.get("/model/status")
def model_status():
    """Return current loaded model info."""
    return {
        "model_ready":  _model is not None,
        "model_source": _model_source,
        "registry_uri": f"models:/bearing_fault_classifier@production",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
