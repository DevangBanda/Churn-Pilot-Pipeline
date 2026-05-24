# Databricks notebook source
# 60_model_serving.py
# Optional post-migration enhancement (atlas Phase 6, step 19). Stands up a
# Model Serving endpoint fronting the UC-registered churn_prediction.ml.churn_model —
# the assignment's static .joblib deliverable becomes a live REST endpoint.

# COMMAND ----------
# CELL 1 — assertion (run first; expect the endpoint lookup to raise / return NOT_FOUND)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
try:
    w.serving_endpoints.get("churn-prediction-endpoint")
    raise AssertionError("Expected the serving endpoint NOT to exist yet")
except Exception as e:
    assert "does not exist" in str(e).lower() or "not found" in str(e).lower()
print("OK: endpoint does not exist yet, as expected")

# COMMAND ----------
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

ENDPOINT_NAME = "churn-prediction-endpoint"
MODEL_NAME = "churn_prediction.ml.churn_model"
MODEL_VERSION = "1"  # bump after promoting a new champion via the UC Model Registry UI

w.serving_endpoints.create(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME,
                entity_version=MODEL_VERSION,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    ),
)
print(f"Creating endpoint {ENDPOINT_NAME} for {MODEL_NAME} v{MODEL_VERSION} (this takes a few minutes)")

# COMMAND ----------
# Re-verification (green phase): provisioning a serving endpoint is a long-running
# Databricks operation, so we poll rather than assert immediately. POLL_ATTEMPTS *
# POLL_INTERVAL_SECONDS gives a ~15-minute budget, mirroring the retry/backoff style
# used by ingest_huggingface_data in 10_ingestion_to_landing.py.
import time

POLL_ATTEMPTS = 30
POLL_INTERVAL_SECONDS = 30

for _ in range(POLL_ATTEMPTS):
    ep = w.serving_endpoints.get(ENDPOINT_NAME)
    if ep.state.ready == "READY":
        break
    time.sleep(POLL_INTERVAL_SECONDS)
assert ep.state.ready == "READY", f"Endpoint not ready: {ep.state}"
print(f"OK: {ENDPOINT_NAME} is READY and serving {MODEL_NAME} v{MODEL_VERSION}")
