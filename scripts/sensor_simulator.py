"""sensor_simulator.py — Real-time CWRU fault simulator.

Generates sliding windows from pre-loaded vibration data pools (with optional
drift injection), then calls POST /predict on the api_server to:
  1. Run FaultMLP inference (single model instance owned by api_server)
  2. Write result + metadata to InfluxDB

This keeps inference logic in one place (api_server) — if the @production model
is updated (/model/reload), the simulator automatically benefits with no restart.
"""
import os
import sys
import time
import requests

import pandas as pd
import numpy as np

import io
import uuid

# ── Configuration ───────────────────────────────────────────────────────────────
API_BASE    = os.environ.get("API_BASE", "http://localhost:8000")
PREDICT_URL = f"{API_BASE}/predict"
MINIO_URL   = os.environ.get("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS= os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET= os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET= "drift-data"

NORMAL_DATA  = "data/interim/normal/Normal_0.parquet"
FAULT_POOLS  = {
    "B":  "data/interim/12k_drive_end/B007_0.parquet",
    "IR": "data/interim/12k_drive_end/IR007_0.parquet",
    "OR": "data/interim/12k_drive_end/OR007@6_0.parquet",
}
TRIGGER_FILE = "data/fault_trigger.txt"


class SensorSimulator:
    def __init__(self):
        print(f"Python:      {sys.executable}")
        print(f"Predict URL: {PREDICT_URL}")
        print("Loading data pools...")

        self.normal_pool = pd.read_parquet(NORMAL_DATA)["vibration_de"].values
        self.fault_pools = {}
        for name, path in FAULT_POOLS.items():
            if os.path.exists(path):
                print(f"  Pool [{name}]: {path}")
                self.fault_pools[name] = pd.read_parquet(path)["vibration_de"].values
            else:
                print(f"  WARNING: Pool [{name}] not found at {path}")

        self.current_mode  = "normal"
        self.drift_active  = False
        self.drift_idx     = 0.0     # 0.0 (Normal) -> 1.0 (100% IR)
        self.drift_buffer  = []      # To collect and upload chunks
        self.idx = 0
        
        # Init MinIO client for drift data uploading
        try:
            import boto3
            from botocore.client import Config
            self.s3 = boto3.client("s3", endpoint_url=MINIO_URL, aws_access_key_id=MINIO_ACCESS,
                                   aws_secret_access_key=MINIO_SECRET, config=Config(signature_version="s3v4"))
            try:
                self.s3.head_bucket(Bucket=MINIO_BUCKET)
            except:
                self.s3.create_bucket(Bucket=MINIO_BUCKET)
            self.minio_ready = True
            print(f"[Simulator] Connected to MinIO bucket '{MINIO_BUCKET}'")
        except Exception as e:
            self.minio_ready = False
            print(f"[Simulator] WARNING: MinIO not reachable: {e}")

    def upload_drift_chunk(self):
        """Upload a batch of drift windows to MinIO"""
        if not self.minio_ready or not self.drift_buffer: return
        try:
            df = pd.DataFrame({"window_data": self.drift_buffer})
            parquet_buf = io.BytesIO()
            df.to_parquet(parquet_buf, index=False)
            file_name = f"drift_chunk_{uuid.uuid4().hex[:8]}.parquet"
            self.s3.put_object(Bucket=MINIO_BUCKET, Key=file_name, Body=parquet_buf.getvalue())
            print(f"  [MinIO] Uploaded drift chunk: {file_name} ({len(self.drift_buffer)} windows)")
            self.drift_buffer = []
        except Exception as e:
            print(f"  [MinIO Error] {e}")

    def check_trigger(self):
        if not os.path.exists(TRIGGER_FILE):
            return
        try:
            with open(TRIGGER_FILE) as f:
                cmd = f.read().strip()
            if cmd in ("normal", "standby"):
                self.current_mode = "normal"
            elif cmd == "reset_all":             # reset fault AND drift together
                self.current_mode = "normal"
                self.drift_active = False
                self.drift_snr_db = 40.0
            elif cmd in self.fault_pools:        # "B", "IR", "OR"
                self.current_mode = cmd
            elif cmd == "drift_on":
                if not self.drift_active:
                    self.drift_active = True
                    self.drift_idx    = 0.0
                    self.drift_buffer = []
            elif cmd == "drift_off":
                if self.drift_active:
                    self.upload_drift_chunk() # Flush remaining
                self.drift_active = False
                self.drift_idx    = 0.0
        except Exception:
            pass

    def apply_drift(self, window_normal: np.ndarray) -> np.ndarray:
        """Progressively blend Normal with IR. Increases drift_idx by 0.05 per cycle."""
        self.drift_idx = min(1.0, self.drift_idx + 0.05)
        
        # Fetch an IR window for blending
        ir_pool = self.fault_pools.get("IR", self.normal_pool)
        ir_idx = self.idx % (len(ir_pool) - 2048)
        window_ir = ir_pool[ir_idx : ir_idx + 2048]
        
        # Blend mathematically
        window_drift = (1.0 - self.drift_idx) * window_normal + self.drift_idx * window_ir
        
        # Add slight static GN
        signal_power = np.mean(window_drift ** 2)
        if signal_power > 0:
            snr_linear  = 10 ** (20.0 / 10.0) # Fixed 20 dB noise for realism
            noise_power = signal_power / snr_linear
            window_drift = window_drift + np.random.normal(0, np.sqrt(noise_power), len(window_drift))
            
        return window_drift

    def run(self):
        print("Simulation started — calling /predict → InfluxDB…")
        t_epoch   = time.perf_counter()
        win_count = 0
        try:
            while True:
                self.check_trigger()
                pool = self.fault_pools.get(self.current_mode, self.normal_pool)

                # Sliding window (2048 samples, stride 512)
                if self.idx + 2048 > len(pool):
                    self.idx = 0
                window    = pool[self.idx : self.idx + 2048].copy()
                self.idx += 512

                if self.drift_active:
                    window = self.apply_drift(window)
                    self.drift_buffer.append(window.tolist())
                    if len(self.drift_buffer) >= 20:
                        self.upload_drift_chunk()

                win_count += 1
                elapsed    = time.perf_counter() - t_epoch
                throughput = win_count / elapsed if elapsed > 0 else 0.0
                drift_metric = self.drift_idx * 100.0  # 0% → 100%

                # ── Call /predict → API server does inference + writes to InfluxDB ──
                payload = {
                    "window":         window.tolist(),
                    "injected_mode":  self.current_mode,
                    "drift_active":   self.drift_active,
                    "drift_metric":   float(drift_metric),
                    "throughput_wps": float(throughput),
                    "push_to_influxdb": True,
                }
                try:
                    r      = requests.post(PREDICT_URL, json=payload, timeout=5)
                    result = r.json()
                    print(
                        f"Mode: {self.current_mode:6} | "
                        f"Pred: {result.get('predicted_label','?')} "
                        f"({result.get('predicted_class','?'):6}) | "
                        f"Conf: {result.get('confidence', 0):.2%} | "
                        f"Lat: {result.get('latency_ms', 0):.1f}ms | "
                        f"Blend: {self.drift_idx:.1%} | "
                        f"InfluxDB: {result.get('influxdb_written', '?')}"
                    )
                except Exception as e:
                    print(f"[WARN] /predict call failed: {e}")

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("Simulator stopped.")


if __name__ == "__main__":
    SensorSimulator().run()
