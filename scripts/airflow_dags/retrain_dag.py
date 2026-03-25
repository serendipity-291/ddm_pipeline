"""
Airflow Retrain DAG
===================
Triggered by drift_monitor.py when K–S drift detection on RMS (see drift_monitor)
fires the configured consecutive checks. Runs the full retraining pipeline and
promotes the model if improved (McNemar / accuracy rules in api_server).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
API_BASE   = os.environ.get("API_URL", "http://api_server:8000")
MLFLOW_URI = os.environ.get("MLFLOW_URL", "http://mlflow:5000")

# ── DAG defaults ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ── Task Functions ────────────────────────────────────────────────────────────

def run_feature_engineering(**context):
    """Trigger feature engineering on the API server."""
    import requests
    log.info(f"Triggering feature engineering at {API_BASE}/train/feature_engineering...")
    response = requests.post(f"{API_BASE}/train/feature_engineering")
    
    if response.status_code != 200:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    
    data = response.json()
    output = data.get("output", "")
    log.info("Feature engineering complete.")
    log.info("Feature engineering mode: retrain")

    for line in output.splitlines():
        if "drift_windows_ingested=" in line:
            log.info(f"Drift ingest stats: {line.strip()}")

    summary_lines = []
    capture = False
    for line in output.splitlines():
        if "=== Final balanced distribution ===" in line:
            capture = True
            continue
        if capture and len(summary_lines) < 6:
            if not line.strip() and summary_lines:
                break
            summary_lines.append(line)
    if summary_lines:
        log.info("Retrain class distribution:\n%s", "\n".join(summary_lines))
    log.info(f"API Output: {output[:500]}...")


def run_model_training(**context):
    """Trigger model training on the API server."""
    import requests
    log.info(f"Triggering model training at {API_BASE}/train/model_training...")
    # Defaulting to 50 epochs for retraining
    response = requests.post(f"{API_BASE}/train/model_training", params={"epochs": 50})
    
    if response.status_code != 200:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    
    data = response.json()
    val_acc = data.get("val_accuracy")
    log.info(f"Model training complete. Val Acc: {val_acc}")
    
    if val_acc is not None:
        context["ti"].xcom_push(key="val_accuracy", value=val_acc)
    
    log.info(f"API Output: {data.get('output')[:500]}...")


def evaluate_and_promote(**context):
    """
    Trigger model evaluation and promotion on the API server.
    """
    import requests
    log.info(f"Triggering model promotion at {API_BASE}/train/promote...")
    response = requests.post(f"{API_BASE}/train/promote")
    
    if response.status_code != 200:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    
    data = response.json()
    decision = data.get("decision", data.get("status", "unknown"))
    log.info(f"Promotion decision: {decision}")

    if decision == "promoted":
        log.info(
            "New model promoted. challenger_acc=%s champion_acc=%s acc_delta=%s",
            data.get("challenger_acc"),
            data.get("champion_acc"),
            data.get("acc_delta"),
        )
        reload_info = data.get("reload")
        if reload_info:
            log.info(f"Reload result after promote: {reload_info}")
    elif decision == "kept_current":
        log.info(
            "Kept current production model. challenger_acc=%s champion_acc=%s reason=%s",
            data.get("challenger_acc"),
            data.get("champion_acc"),
            data.get("reason"),
        )
    elif decision in ("no_challenger", "skipped"):
        log.info(f"Promotion skipped. reason={data.get('reason')}")
    else:
        log.info(f"Promotion response: {data}")


# ── DAG Definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="retrain_dag",
    description="Auto-retraining pipeline triggered by drift detection",
    default_args=default_args,
    schedule=None,          # External trigger only (via drift_monitor)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "retrain", "drift"],
) as dag:

    t1 = PythonOperator(task_id="feature_engineering", python_callable=run_feature_engineering)
    t2 = PythonOperator(task_id="model_training",      python_callable=run_model_training)
    t3 = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)

    t1 >> t2 >> t3
