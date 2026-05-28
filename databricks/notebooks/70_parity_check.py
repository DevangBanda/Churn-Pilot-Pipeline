# Databricks notebook source
# 70_parity_check.py
# Compares one legacy main_pipeline.py / Airflow run's outputs against the
# Databricks Workflow's outputs for the same logical run — atlas Phase 8, step 23.
# This is also wired in as the final task ("pipeline_success" replacement) of the
# Asset Bundle job (Task 11), so it runs automatically at the end of every Workflow run.

# COMMAND ----------
# CELL 1 — assertion (run first, before legacy outputs have been exported for comparison;
# expect FileNotFoundError because the comparison CSVs don't exist yet)
import pandas as pd
legacy_gold = pd.read_csv("/Volumes/churn_prediction/bronze/landing_volume/legacy_export/training_set_latest.csv")
assert len(legacy_gold) > 0

# COMMAND ----------
# Operator prerequisite (manual, live-environment step — atlas Phase 8, step 23):
# Before the cells below can pass, run one full cycle of the legacy pipeline
# (main_pipeline.py, or trigger the Airflow DAG), locate its output at
# data/processed/training_sets/<set_id>.csv, and stage it for comparison with:
#   databricks fs cp data/processed/training_sets/<set_id>.csv \
#     dbfs:/Volumes/churn_prediction/bronze/landing_volume/legacy_export/training_set_latest.csv
# Only once that file exists will CELL 1 above stop raising FileNotFoundError and the
# parity comparisons below become runnable.

import pandas as pd
from pyspark.sql import functions as F

LEGACY_CSV = "/Volumes/churn_prediction/bronze/landing_volume/legacy_export/training_set_latest.csv"
GOLD_TABLE = "churn_prediction.gold.customer_features"

legacy_pdf = pd.read_csv(LEGACY_CSV)
gold_df = spark.table(GOLD_TABLE)

# COMMAND ----------
# 1. Record-count parity
legacy_count = len(legacy_pdf)
gold_count = gold_df.count()
count_diff_pct = abs(legacy_count - gold_count) / max(legacy_count, 1) * 100
print(f"Legacy rows: {legacy_count} | Gold rows: {gold_count} | diff: {count_diff_pct:.2f}%")
assert count_diff_pct < 1.0, "Row-count parity check failed (>1% difference)"

# COMMAND ----------
# 2. Churn-rate distribution parity
legacy_rate = legacy_pdf["Churn"].mean()
gold_rate = gold_df.select(F.avg("Churn")).first()[0]
rate_diff = abs(legacy_rate - gold_rate)
print(f"Legacy churn rate: {legacy_rate:.4f} | Gold churn rate: {gold_rate:.4f} | diff: {rate_diff:.4f}")
assert rate_diff < 0.01, "Churn-rate distribution parity check failed (>1pp difference)"

# COMMAND ----------
# 3. Model metric parity — compare latest UC-registered run vs. the metrics.json
#    that build_model.py logged into MLflow's file:///tmp/mlflow-runs store on the legacy box
from mlflow import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
latest_version = client.get_registered_model("churn_prediction.ml.churn_model").latest_versions[0]
new_run = client.get_run(latest_version.run_id)
new_f1 = new_run.data.metrics["f1"]

LEGACY_F1 = 0.58  # <- replace with the f1 value printed by the legacy build_model.py run's MLflow log
f1_diff = abs(new_f1 - LEGACY_F1)
print(f"Legacy F1: {LEGACY_F1:.4f} | Databricks F1: {new_f1:.4f} | diff: {f1_diff:.4f}")
assert f1_diff < 0.05, "Model-metric parity check failed (>0.05 F1 difference)"

print("PARITY CHECK PASSED — outputs match within tolerance across 1 cycle")
