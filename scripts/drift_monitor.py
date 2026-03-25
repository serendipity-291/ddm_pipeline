"""
Drift Monitor Service
=====================
Polls InfluxDB ``inference_results`` on a fixed interval (``POLL_INTERVAL``).

Drift is **data-distribution drift**, not model confidence: it compares RMS feature
values from the **baseline** stream (``drift_active == False`` in the last ~2h) to
the **current** stream (last ~30s) using a two-sample Kolmogorov–Smirnov test.
If ``p_value < DRIFT_P_VALUE_THRESHOLD`` for ``CONSECUTIVE_CHECKS`` polls in a row,
the monitor treats that as drift and:

  1. Writes alerts to InfluxDB ``system_alerts`` (``drift_score`` ≈ ``1 - p_value``).
  2. On sustained drift, logs an event to MLflow and triggers Airflow ``retrain_dag``.

Classifier softmax confidence is written by the API but is **not** used here.
"""

import os
import time
import logging
from datetime import datetime, timezone

import mlflow
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DRIFT] %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
INFLUXDB_URL   = os.environ.get("INFLUXDB_URL",   "http://localhost:8086")
INFLUXDB_TOKEN = "UziSGCgplwTUlHTdRiHWPIwFasDqSPbKxqfx5C_I7rsZuICEvvgAbRD3L1_a8U4R48f7mmJs9QMxX0dmjjNdEg=="
INFLUXDB_ORG   = "my-org"
INFLUXDB_BUCKET = "bearing_data"

AIRFLOW_URL    = os.environ.get("AIRFLOW_URL",   "http://localhost:8080")
AIRFLOW_USER   = os.environ.get("AIRFLOW_USER",  "admin")
AIRFLOW_PASS   = os.environ.get("AIRFLOW_PASS",  "admin")
MLFLOW_URL     = os.environ.get("MLFLOW_URL",    "http://localhost:5000")

import scipy.stats as stats

DRIFT_P_VALUE_THRESHOLD = 0.05 # p-value < 0.05 means distributions are significantly different
CONSECUTIVE_CHECKS    = 3      # Must fail this many times in a row
POLL_INTERVAL         = 10     # Seconds between checks
WINDOW_COUNT          = 60     # Unused; reserved for future window-based metrics


def get_rms_data(query_api, time_range="-2h", filter_clause="") -> list[float]:
    """Fetch RMS values from InfluxDB with optional filtering."""
    flux = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {time_range})
      |> filter(fn: (r) => r._measurement == "inference_results")
      |> filter(fn: (r) => r._field == "rms")
      {filter_clause}
      |> keep(columns: ["_value"])
    """
    try:
        tables = query_api.query(flux, org=INFLUXDB_ORG)
        values = []
        for table in tables:
            for record in table.records:
                values.append(float(record.get_value()))
        return values
    except Exception as e:
        log.error(f"InfluxDB Query Error: {e}")
        return []

def check_ks_drift(query_api) -> float | None:
    """Compare recent RMS vs historical baseline RMS using K-S test. Returns p-value."""
    # 1. Fetch Baseline (Healthy data: drift_active == false)
    baseline_filter = '|> filter(fn: (r) => r["drift_active"] == "False")'
    baseline_rms = get_rms_data(query_api, time_range="-2h", filter_clause=baseline_filter)
    
    # limit baseline size for performance
    if len(baseline_rms) > 2000:
        import random
        baseline_rms = random.sample(baseline_rms, 2000)
        
    # 2. Fetch Current Window (Last 30s)
    current_rms = get_rms_data(query_api, time_range="-30s", filter_clause="")
    
    if len(baseline_rms) < 50 or len(current_rms) < 10:
        return None # Not enough data
        
    # 3. K-S Test
    stat, p_value = stats.ks_2samp(baseline_rms, current_rms)
    return p_value


def write_drift_event(write_api, drift_score: float, is_drifted: bool, trigger_count: int):
    """Write drift alert to system_alerts measurement in InfluxDB."""
    point = Point("system_alerts") \
        .tag("type", "drift_detection") \
        .field("drift_score",   float(drift_score)) \
        .field("is_drifted",    1 if is_drifted else 0) \
        .field("trigger_count", int(trigger_count)) \
        .time(datetime.utcnow())
    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)


def trigger_airflow_dag(dag_id: str = "retrain_dag"):
    """POST to Airflow REST API to trigger a DAG run."""
    import requests
    # Airflow 2.x API with prefix
    url = f"{AIRFLOW_URL}/airflow/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": f"drift_triggered_{int(time.time())}",
        "conf": {"triggered_by": "drift_monitor"}
    }
    try:
        r = requests.post(url, json=payload, auth=(AIRFLOW_USER, AIRFLOW_PASS), timeout=10)
        if r.status_code in (200, 201):
            log.info(f"Airflow DAG '{dag_id}' triggered successfully.")
        else:
            log.error(f"Airflow trigger failed: {r.status_code} {r.text}")
    except Exception as e:
        log.error(f"Airflow connection error: {e}")


def log_to_mlflow(drift_score: float, trigger_count: int):
    """Log drift detection event to MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_experiment("drift_monitoring")
        with mlflow.start_run(run_name=f"drift_event_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metric("drift_score",    drift_score)
            mlflow.log_metric("trigger_count",  trigger_count)
            mlflow.log_param("threshold",       DRIFT_P_VALUE_THRESHOLD)
            mlflow.log_param("action",          "airflow_retrain_triggered")
        log.info("MLflow drift event logged.")
    except Exception as e:
        log.error(f"MLflow logging error: {e}")


def main():
    log.info(f"Drift Monitor started | p_value_threshold={DRIFT_P_VALUE_THRESHOLD} | poll={POLL_INTERVAL}s")
    client    = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    write_api = client.write_api(write_options=SYNCHRONOUS)

    consecutive = 0
    already_triggered = False

    while True:
        try:
            p_val = check_ks_drift(query_api)
            if p_val is None:
                log.info("Not enough data in InfluxDB to compute K-S Test yet. Waiting...")
                time.sleep(POLL_INTERVAL)
                continue
            
            # For logging/dashboard, we invert the p-value into a "Drift Score" (0.0=No Drift, 1.0=Severe Drift)
            # p_value == 1.0 -> score 0.0
            # p_value == 0.0 -> score 1.0
            drift_score = 1.0 - p_val

            if p_val < DRIFT_P_VALUE_THRESHOLD:
                consecutive += 1
                log.warning(f"K-S Test flagged Drift! p-value: {p_val:.2e} (score: {drift_score:.1%}) | consecutive={consecutive}/{CONSECUTIVE_CHECKS}")
                write_drift_event(write_api, drift_score, is_drifted=(consecutive >= CONSECUTIVE_CHECKS), trigger_count=consecutive)

                if consecutive >= CONSECUTIVE_CHECKS and not already_triggered:
                    log.info("DRIFT DETECTED! Triggering Airflow retraining pipeline...")
                    log_to_mlflow(drift_score, consecutive)
                    trigger_airflow_dag("retrain_dag")
                    already_triggered = True
            else:
                if consecutive > 0:
                    log.info(f"Distribution recovered (p-value: {p_val:.2f}). Resetting counter.")
                consecutive = 0
                already_triggered = False
                write_drift_event(write_api, drift_score, is_drifted=False, trigger_count=0)
                log.info(f"Distribution OK (p-value: {p_val:.2f})")

        except Exception as e:
            log.error(f"Monitor error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
