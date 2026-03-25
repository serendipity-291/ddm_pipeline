import os
import argparse
import json
import io
import pandas as pd
import numpy as np
import scipy.stats
from scipy.fft import fft
import pywt

INTERIM_DATA_PATH = "data/interim"
PROCESSED_DATA_PATH = "data/processed"
WINDOW_SIZE = 2048 # Common window size for 12kHz sampling

MINIO_URL = os.environ.get("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
DRIFT_BUCKET = os.environ.get("DRIFT_BUCKET", "drift-data")

def extract_wavelet_features(window, wavelet='db4', level=3):
    """
    Extracts energy features from Wavelet Packet Decomposition (WPD).
    Level 3 WPD gives 8 frequency bands.
    """
    wp = pywt.WaveletPacket(data=window, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    
    features = {}
    total_energy = 0
    energies = []
    
    for i, node in enumerate(nodes):
        energy = np.sum(np.square(wp[node].data))
        energies.append(energy)
        total_energy += energy
        
    # Normalize energy features
    for i, energy in enumerate(energies):
        features[f'wpd_energy_node_{i}'] = energy / total_energy if total_energy > 0 else 0
        
    return features

def extract_time_features(window):
    """
    Extracts 10 common time-domain features.
    """
    abs_window = np.abs(window)
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(abs_window)
    
    features = {
        'mean': np.mean(window),
        'std': np.std(window),
        'max': peak,
        'min': np.min(window),
        'rms': rms,
        'kurtosis': scipy.stats.kurtosis(window),
        'skewness': scipy.stats.skew(window),
        'peak_to_peak': peak - np.min(window),
        'crest_factor': peak / rms if rms != 0 else 0,
        'shape_factor': rms / np.mean(abs_window) if np.mean(abs_window) != 0 else 0,
        'impulse_factor': peak / np.mean(abs_window) if np.mean(abs_window) != 0 else 0
    }
    return features

def extract_freq_features(window, fs):
    """
    Extracts basic frequency-domain features.
    """
    n = len(window)
    f_coeffs = fft(window)
    psd = np.abs(f_coeffs[:n//2])**2 / n
    freqs = np.linspace(0, fs/2, n//2)
    
    # Simple spectral features
    total_energy = np.sum(psd)
    spectral_centroid = np.sum(freqs * psd) / total_energy if total_energy > 0 else 0
    
    features = {
        'spectral_energy_total': total_energy,
        'spectral_centroid': spectral_centroid,
        'spectral_max_freq': freqs[np.argmax(psd)] if len(psd) > 0 else 0
    }
    return features

def process_interim_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        if 'vibration_de' not in df.columns:
            return None
        
        fs = df['sampling_rate_hz'].iloc[0]
        vibration = df['vibration_de'].values
        
        # Metadata
        meta = {
            'fault_type': df['fault_type'].iloc[0],
            'fault_diameter': df['fault_diameter_inch'].iloc[0],
            'load': df['load_hp'].iloc[0],
            'rpm': df['rpm'].iloc[0]
        }
        
        # Segmentation and Feature Extraction
        feature_list = []
        n_windows = len(vibration) // WINDOW_SIZE
        
        for i in range(n_windows):
            window = vibration[i*WINDOW_SIZE : (i+1)*WINDOW_SIZE].copy()
            
            # Extract features
            time_feat = extract_time_features(window)
            freq_feat = extract_freq_features(window, fs)
            wave_feat = extract_wavelet_features(window)
            
            # Combine
            combined = {**meta, **time_feat, **freq_feat, **wave_feat}
            feature_list.append(combined)
            
        return pd.DataFrame(feature_list)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_window_feature_row(window, meta, fs=12000):
    if len(window) < WINDOW_SIZE:
        return None
    clipped = np.array(window[:WINDOW_SIZE], dtype=float)
    time_feat = extract_time_features(clipped)
    freq_feat = extract_freq_features(clipped, fs)
    wave_feat = extract_wavelet_features(clipped)
    return {**meta, **time_feat, **freq_feat, **wave_feat}


def build_baseline_features():
    # Only use 12k_drive_end (consistent 12kHz Drive End sensor)
    # and normal (same 12kHz sensor). Exclude 12k_fan_end and 48k_drive_end
    # to avoid mixing incompatible sampling rates and sensor positions.
    allowed_categories = {"12k_drive_end", "normal"}
    all_features = []

    for category in os.listdir(INTERIM_DATA_PATH):
        cat_path = os.path.join(INTERIM_DATA_PATH, category)
        if not os.path.isdir(cat_path):
            continue
        if category not in allowed_categories:
            print(f"Skipping category: {category} (not in allowed categories)")
            continue

        for filename in os.listdir(cat_path):
            if not filename.endswith(".parquet"):
                continue
            file_path = os.path.join(cat_path, filename)
            print(f"Processing {filename}...")
            df_features = process_interim_file(file_path)
            if df_features is not None:
                all_features.append(df_features)

    if not all_features:
        return pd.DataFrame()

    baseline_df = pd.concat(all_features, ignore_index=True)
    baseline_df["fault_class"] = baseline_df["fault_type"].apply(
        lambda x: "OR" if str(x).startswith("OR") else str(x)
    )
    baseline_df = baseline_df[baseline_df["fault_class"].isin({"Normal", "B"})].copy()
    return baseline_df


def load_drift_features_from_minio():
    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        print("[Retrain] boto3 not available, skipping drift MinIO ingest.")
        return pd.DataFrame(), 0

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        paginator = s3.get_paginator("list_objects_v2")
        feature_rows = []
        ingested = 0

        for page in paginator.paginate(Bucket=DRIFT_BUCKET):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if not key.endswith(".parquet"):
                    continue
                body = s3.get_object(Bucket=DRIFT_BUCKET, Key=key)["Body"].read()
                df_chunk = pd.read_parquet(io.BytesIO(body))
                if "window_data" not in df_chunk.columns:
                    continue
                for window_data in df_chunk["window_data"].tolist():
                    row = extract_window_feature_row(
                        window_data,
                        meta={
                            "fault_type": "IR",
                            "fault_class": "IR",
                            "fault_diameter": 0.007,
                            "load": 1,
                            "rpm": 0.0,
                        },
                    )
                    if row is not None:
                        feature_rows.append(row)
                        ingested += 1

        if not feature_rows:
            return pd.DataFrame(), 0
        return pd.DataFrame(feature_rows), ingested
    except Exception as e:
        print(f"[Retrain] MinIO drift ingest failed: {e}")
        return pd.DataFrame(), 0

def balance_classes(df, target=4000):
    """Undersample majority and upsample minority classes to `target` each."""
    parts = []
    for cls in sorted(df['label'].unique()):
        chunk = df[df['label'] == cls]
        label_name = chunk['fault_class'].iloc[0]
        if len(chunk) > target:
            chunk = chunk.sample(n=target, random_state=42)
            print(f"  [{label_name}] undersampled: {len(chunk)} → {target}")
        elif len(chunk) < target:
            chunk = chunk.sample(n=target, replace=True, random_state=42)
            print(f"  [{label_name}] upsampled:    {len(chunk)} → {target}")
        else:
            print(f"  [{label_name}] unchanged:    {len(chunk)}")
        parts.append(chunk)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="baseline", choices=["baseline", "retrain"])
    args = parser.parse_args()

    print("Starting feature engineering...")
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    baseline_df = build_baseline_features()
    if baseline_df.empty:
        print("No baseline features extracted.")
        return

    drift_df = pd.DataFrame()
    drift_windows = 0
    if args.mode == "retrain":
        drift_df, drift_windows = load_drift_features_from_minio()
        print(f"[Retrain] drift_windows_ingested={drift_windows}")

    if args.mode == "retrain" and not drift_df.empty:
        final_df = pd.concat([baseline_df, drift_df], ignore_index=True)
    else:
        final_df = baseline_df.copy()

    if not final_df.empty:
        final_df['label'] = pd.Categorical(final_df['fault_class']).codes

        # Print raw distribution before balancing
        print("\n=== Raw class distribution ===")
        raw_dist = final_df.groupby(['fault_class', 'label']).size().reset_index(name='count')
        print(raw_dist.to_string(index=False))

        # Balance all present classes to 4000 samples each
        print("\n=== Balancing classes to 4000 each ===")
        final_df = balance_classes(final_df, target=4000)

        print("\n=== Final balanced distribution ===")
        bal_dist = final_df.groupby(['fault_class', 'label']).size().reset_index(name='count')
        print(bal_dist.to_string(index=False))
        print(f"Total windows: {len(final_df)}")

        output_path = os.path.join(PROCESSED_DATA_PATH, "cwru_features.parquet")
        final_df.to_parquet(output_path, index=False)
        final_df.head(100).to_csv(
            os.path.join(PROCESSED_DATA_PATH, "cwru_features_sample.csv"), index=False
        )
        class_mapping = (
            final_df[["label", "fault_class"]]
            .drop_duplicates()
            .sort_values("label")
        )
        classes_path = os.path.join(PROCESSED_DATA_PATH, "classes.json")
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump({int(r.label): str(r.fault_class) for r in class_mapping.itertuples()}, f)
        print(f"\nSaved to {output_path}")
        print(f"Mode: {args.mode} | classes saved to {classes_path}")

        # ── DVC-style Dataset Versioning ────────────────────────────────────────
        # Compute SHA256 hash of the parquet file for reproducibility tracking.
        # The hash is written to .data_version so model_training.py can log it
        # as an MLflow param, and a MinIO snapshot is uploaded for long-term storage.
        try:
            import hashlib
            with open(output_path, "rb") as f:
                data_hash = hashlib.sha256(f.read()).hexdigest()[:12]

            version_file = os.path.join(PROCESSED_DATA_PATH, ".data_version")
            with open(version_file, "w") as f:
                f.write(data_hash)
            print(f"[DVC] Dataset version: {data_hash}")

            # Upload snapshot to MinIO (best-effort — non-fatal if unavailable)
            try:
                import boto3
                from botocore.client import Config

                MINIO_URL      = os.environ.get("MINIO_URL", "http://minio:9000")
                MINIO_ACCESS   = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
                MINIO_SECRET   = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
                MINIO_BUCKET   = "mlops-data"
                snapshot_key   = f"snapshots/{data_hash}/cwru_features.parquet"

                s3 = boto3.client(
                    "s3",
                    endpoint_url=MINIO_URL,
                    aws_access_key_id=MINIO_ACCESS,
                    aws_secret_access_key=MINIO_SECRET,
                    config=Config(signature_version="s3v4"),
                )
                # Ensure bucket exists
                try:
                    s3.head_bucket(Bucket=MINIO_BUCKET)
                except Exception:
                    s3.create_bucket(Bucket=MINIO_BUCKET)

                # Only upload if this version doesn't already exist
                try:
                    s3.head_object(Bucket=MINIO_BUCKET, Key=snapshot_key)
                    print(f"[DVC] Snapshot {data_hash} already exists in MinIO — skipping upload.")
                except Exception:
                    s3.upload_file(output_path, MINIO_BUCKET, snapshot_key)
                    print(f"[DVC] Snapshot uploaded → minio://{MINIO_BUCKET}/{snapshot_key}")

            except ImportError:
                print("[DVC] boto3 not installed — skipping MinIO snapshot (hash saved locally).")
            except Exception as minio_err:
                print(f"[DVC] MinIO upload skipped (non-fatal): {minio_err}")

        except Exception as hash_err:
            print(f"[DVC] Versioning skipped (non-fatal): {hash_err}")

    else:
        print("No features extracted.")


if __name__ == "__main__":
    main()
