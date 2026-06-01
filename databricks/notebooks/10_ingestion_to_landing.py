# Databricks notebook source
# 10_ingestion_to_landing.py
# Replaces src/data_ingestion.py — same two sources (GitHub raw CSV, HF Datasets Server API),
# same retry/backoff/cache-fallback behavior; destination is now a UC Volume instead of
# the local data/raw/ directory.

# COMMAND ----------
# CELL 1 — assertion (run first; expect FileNotFoundError / empty listing)
landing_files = dbutils.fs.ls("/Volumes/churn_pilot/bronze/landing_volume/")
assert any(f.name.startswith("customer_churn_") and f.name.endswith(".csv") for f in landing_files), \
    "Expected at least one customer_churn_*.csv in the landing volume"
assert any(f.name.startswith("huggingface_churn_") and f.name.endswith(".json") for f in landing_files), \
    "Expected at least one huggingface_churn_*.json in the landing volume"

# COMMAND ----------
import json
import time
from datetime import datetime

import requests

LANDING_PATH = "/Volumes/churn_pilot/bronze/landing_volume"
CSV_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
HF_API_URL = "https://datasets-server.huggingface.co/rows?dataset=scikit-learn%2Fchurn-prediction&config=default&split=train&offset=0&length=100"


def ingest_csv_data() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = f"{LANDING_PATH}/customer_churn_{ts}.csv"
    resp = requests.get(CSV_URL, timeout=30)
    resp.raise_for_status()
    dbutils.fs.put(dest, resp.text, overwrite=True)
    print(f"Wrote CSV to {dest} ({len(resp.text)} bytes)")
    return dest


def ingest_huggingface_data(max_retries: int = 3, backoff_seconds: int = 2) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = f"{LANDING_PATH}/huggingface_churn_{ts}.json"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(HF_API_URL, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            dbutils.fs.put(dest, json.dumps(payload), overwrite=True)
            print(f"Wrote HF sample to {dest} (attempt {attempt})")
            return dest
        except requests.RequestException as exc:
            print(f"HF ingestion attempt {attempt}/{max_retries} failed: {exc}")
            if attempt == max_retries:
                # cache-fallback: reuse the most recent previously-landed file, mirroring
                # DataIngestionPipeline's local-cache fallback behavior
                existing = sorted(
                    (f for f in dbutils.fs.ls(LANDING_PATH) if f.name.startswith("huggingface_churn_")),
                    key=lambda f: f.name,
                    reverse=True,
                )
                if existing:
                    print(f"Falling back to cached file {existing[0].path}")
                    return existing[0].path
                raise
            time.sleep(backoff_seconds * attempt)


# COMMAND ----------
csv_path = ingest_csv_data()
hf_path = ingest_huggingface_data()
print(f"Ingestion complete: {csv_path}, {hf_path}")
