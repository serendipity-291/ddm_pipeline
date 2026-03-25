"""
setup_grafana.py
================
Provisions the DDM MLOps Monitor dashboard on Grafana via REST API.
Idempotent — safe to run multiple times.

Usage: python scripts/setup_grafana.py
"""

import os
import requests
from config import get_env

GRAFANA_URL = get_env("GRAFANA_URL", "http://localhost/grafana")
GRAFANA_USER = get_env("GRAFANA_USER", "admin")
GRAFANA_PASS = get_env("GRAFANA_PASS", "password123")
INFLUXDB_TOKEN = get_env("INFLUXDB_TOKEN", required=True)

INFLUX_DS   = "InfluxDB"
BUCKET      = "bearing_data"
ORG         = "my-org"
MEASUREMENT = "inference_results"
ALERTS_M    = "system_alerts"
DASHBOARD_UID   = "ddm_pipeline"
DASHBOARD_TITLE = "DDM – MLOps Monitor"

SESSION = requests.Session()
SESSION.auth = (GRAFANA_USER, GRAFANA_PASS)
SESSION.headers.update({"Content-Type": "application/json"})


# ── Flux query helpers ────────────────────────────────────────────────────────
def flux_mean(field, start="-5m"):
    """Single aggregated value (group → mean) — no per-tag series splitting."""
    return (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {start})'
        f' |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")'
        f' |> filter(fn: (r) => r._field == "{field}")'
        f' |> group(columns: [])'          # collapse all tags into one series
        f' |> mean()'
        f' |> map(fn: (r) => ({{r with _field: "{field}"}}))'
    )


def flux_last(field, start="-5m"):
    """Single last value, collapsed."""
    return (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {start})'
        f' |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")'
        f' |> filter(fn: (r) => r._field == "{field}")'
        f' |> group(columns: [])'
        f' |> last()'
        f' |> map(fn: (r) => ({{r with _field: "{field}"}}))'
    )


def flux_timeseries(field, start="-5m"):
    """Time series: need to retain _time for Grafana to plot lines properly."""
    return (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {start})'
        f' |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")'
        f' |> filter(fn: (r) => r._field == "{field}")'
        f' |> group(columns: ["_measurement", "_field"])'  # keep single series but preserve time
        f' |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)'
    )


def flux_alerts(start="-30m"):
    return (
        f'from(bucket: "{BUCKET}")'
        f' |> range(start: {start})'
        f' |> filter(fn: (r) => r._measurement == "{ALERTS_M}")'
        f' |> filter(fn: (r) => r._field == "is_drifted" or r._field == "drift_score")'
        f' |> group(columns: ["_field"])'
        f' |> last()'
        f' |> drop(columns: ["_start", "_stop", "_measurement"])'
    )


# ── Panel builders ────────────────────────────────────────────────────────────
def stat_panel(pid, title, field, x, y, w=6, h=4,
               unit="short", color="#ff6d2b", decimals=1):
    return {
        "id": pid, "title": title, "type": "stat",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": INFLUX_DS,
        "targets": [{
            "refId": "A",
            "query": flux_last(field),
            "datasource": INFLUX_DS,
        }],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": decimals,
                "displayName": title,          # ← clean label, no tag strings
                "color": {"mode": "fixed", "fixedColor": color},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "mappings": [],
            }
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "textMode": "value",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "center",
        },
    }


def timeseries_panel(pid, title, field, x, y, w=12, h=8,
                     unit="short", color="#ff6d2b", thresholds=None,
                     legend_name=None):
    panel = {
        "id": pid, "title": title, "type": "timeseries",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": INFLUX_DS,
        "targets": [{
            "refId": "A",
            "query": flux_timeseries(field),
            "datasource": INFLUX_DS,
        }],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "displayName": legend_name or title,
                "color": {"mode": "fixed", "fixedColor": color},
                "custom": {"lineWidth": 2, "fillOpacity": 10, "spanNulls": True},
            }
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "single"},
        },
    }
    if thresholds:
        panel["fieldConfig"]["defaults"]["thresholds"] = {
            "mode": "absolute", "steps": thresholds,
        }
        panel["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
    return panel


def gauge_panel(pid, title, field, x, y, w=6, h=8,
                unit="ms", min_val=0, max_val=50):
    return {
        "id": pid, "title": title, "type": "gauge",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": INFLUX_DS,
        "targets": [{
            "refId": "A",
            "query": flux_mean(field),
            "datasource": INFLUX_DS,
        }],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": min_val,
                "max": max_val,
                "displayName": title,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green",  "value": None},
                        {"color": "yellow", "value": 10},
                        {"color": "red",    "value": 30},
                    ],
                },
                "color": {"mode": "thresholds"},
            }
        },
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def table_panel(pid, title, x, y, w=24, h=6):
    return {
        "id": pid, "title": title, "type": "table",
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": INFLUX_DS,
        "targets": [{
            "refId": "A",
            "query": flux_alerts(),
            "datasource": INFLUX_DS,
        }],
        "options": {
            "showHeader": True,
            "sortBy": [{"displayName": "Time", "desc": True}]
        },
        "fieldConfig": {
            "defaults": {"custom": {"filterable": True}},
            "overrides": [
                {"matcher": {"id": "byName", "options": "_value"},
                 "properties": [{"id": "displayName", "value": "Value"}]},
                {"matcher": {"id": "byName", "options": "_time"},
                 "properties": [{"id": "displayName", "value": "Time"}, {"id": "custom.width", "value": 200}]},
                {"matcher": {"id": "byName", "options": "_field"},
                 "properties": [{"id": "displayName", "value": "Metric"}]},
                # Hide all raw tags from the table
                {"matcher": {"id": "byRegexp", "options": "^(?!_value|_time|_field).*"},
                 "properties": [{"id": "custom.hidden", "value": True}]}
            ],
        },
    }


# ── Dashboard JSON ────────────────────────────────────────────────────────────
DASHBOARD = {
    "uid":   DASHBOARD_UID,
    "title": DASHBOARD_TITLE,
    "tags":  ["mlops", "bearing", "drift"],
    "timezone": "browser",
    "refresh": "5s",
    "schemaVersion": 37,
    "panels": [
        # ── Row 1: KPI stat panels ──────────────────────────────────────────
        stat_panel(1, "Confidence",       "confidence",           x=0,  y=0, unit="percentunit", color="#2dc937", decimals=1),
        stat_panel(2, "Drift Score",      "drift_metric",         x=6,  y=0, unit="none",        color="#ff6d2b", decimals=1),
        stat_panel(3, "Latency (mean)",   "inference_latency_ms", x=12, y=0, unit="ms",          color="#e6c229", decimals=2),
        stat_panel(4, "Throughput",       "throughput_wps",       x=18, y=0, unit="short",       color="#1f78c1", decimals=1),

        # ── Row 2: Time series ──────────────────────────────────────────────
        timeseries_panel(5, "Confidence Over Time", "confidence",
                         x=0, y=4, w=12, h=8,
                         unit="percentunit", legend_name="Confidence",
                         thresholds=[
                             {"color": "red",   "value": None},
                             {"color": "yellow","value": 0.7},
                             {"color": "green", "value": 0.9},
                         ]),
        timeseries_panel(6, "Drift Score Over Time", "drift_metric",
                         x=12, y=4, w=12, h=8,
                         unit="none", color="#ff6d2b", legend_name="Drift Score"),

        # ── Row 3: Latency + Throughput ────────────────────────────────────
        gauge_panel(7, "Inference Latency (ms)", "inference_latency_ms",
                    x=0, y=12, w=6, h=8, unit="ms", min_val=0, max_val=50),
        timeseries_panel(8, "Latency Over Time", "inference_latency_ms",
                         x=6, y=12, w=12, h=8,
                         unit="ms", color="#e6c229", legend_name="Latency (ms)"),
        timeseries_panel(9, "Throughput (windows/s)", "throughput_wps",
                         x=18, y=12, w=6, h=8,
                         unit="short", color="#1f78c1", legend_name="Throughput"),

        # ── Row 4: Drift alerts table ───────────────────────────────────────
        table_panel(10, "Drift Alerts (system_alerts)", x=0, y=20),
    ],
    "time": {"from": "now-5m", "to": "now"},
}


# ── Grafana API helpers ───────────────────────────────────────────────────────
def get_or_create_datasource():
    r = SESSION.get(f"{GRAFANA_URL}/api/datasources")
    if r.status_code == 200:
        for ds in r.json():
            if ds.get("name") == INFLUX_DS:
                print(f"✓ Datasource '{INFLUX_DS}' already exists (id={ds['id']})")
                return
    print(f"  Creating datasource '{INFLUX_DS}'...")
    payload = {
        "name": INFLUX_DS, "type": "influxdb",
        "url": "http://influxdb:8086", "access": "proxy", "isDefault": True,
        "jsonData": {"version": "Flux", "organization": ORG, "defaultBucket": BUCKET},
        "secureJsonData": {"token": INFLUXDB_TOKEN},
    }
    r2 = SESSION.post(f"{GRAFANA_URL}/api/datasources", json=payload)
    if r2.status_code in (200, 201):
        print("✓ Datasource created.")
    else:
        print(f"✗ Datasource creation failed: {r2.status_code} {r2.text}")


def upsert_dashboard():
    payload = {"dashboard": DASHBOARD, "overwrite": True, "folderId": 0}
    r = SESSION.post(f"{GRAFANA_URL}/api/dashboards/db", json=payload)
    if r.status_code in (200, 201):
        resp = r.json()
        print(f"✓ Dashboard upserted: {resp.get('url', '')}")
        print(f"  → http://localhost{resp.get('url', '')}")
    else:
        print(f"✗ Dashboard upsert failed: {r.status_code} {r.text}")


if __name__ == "__main__":
    print("=== Grafana MLOps Monitor Setup ===")
    get_or_create_datasource()
    upsert_dashboard()
    print("Done.")
