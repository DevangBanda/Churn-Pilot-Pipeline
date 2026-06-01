# Databricks notebook source
# 11_autoloader_bronze.py
# Replaces src/raw_data_storage.py — Auto Loader reads everything landed in the volume
# by 10_ingestion_to_landing.py and appends it to a Bronze Delta table. UC's table
# metadata + Catalog Explorer lineage tab replaces create_data_catalog()'s JSON file.

# COMMAND ----------
# CELL 1 — assertion (run first; expect AnalysisException: Table or view not found)
df = spark.table("churn_pilot.bronze.customer_churn_raw")
assert df.count() > 0, "Expected at least one row in bronze.customer_churn_raw"
assert "_ingested_at" in df.columns, "Expected Auto Loader metadata column _ingested_at"

# COMMAND ----------
from pyspark.sql import functions as F

LANDING_PATH = "/Volumes/churn_pilot/bronze/landing_volume"
CHECKPOINT_PATH = "/Volumes/churn_pilot/bronze/landing_volume/_checkpoints/customer_churn_raw"
BRONZE_TABLE = "churn_pilot.bronze.customer_churn_raw"
LOG_TABLE = "churn_pilot.bronze.pipeline_logs"

# COMMAND ----------
# Structured logging table — replaces utils/logger.py's per-stage logs/<pipeline_name>.log
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
    pipeline_name STRING,
    level STRING,
    message STRING,
    logged_at TIMESTAMP
) USING DELTA
COMMENT 'Structured pipeline logs — replaces logs/*.log files written by utils/logger.py'
""")


def log(pipeline_name: str, level: str, message: str) -> None:
    spark.createDataFrame(
        [(pipeline_name, level, message)], "pipeline_name STRING, level STRING, message STRING"
    ).withColumn("logged_at", F.current_timestamp()).write.format("delta").mode("append").saveAsTable(LOG_TABLE)


# COMMAND ----------
# Auto Loader stream: GitHub CSV landing files (customer_churn_*.csv) -> Bronze table with source/type metadata columns
raw_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")
    .option("header", "true")
    .load(f"{LANDING_PATH}/customer_churn_*.csv")
    .withColumn("_source_file", F.input_file_name())
    .withColumn("_source", F.lit("github_csv"))
    .withColumn("_ingested_at", F.current_timestamp())
)

query = (
    raw_stream.writeStream.format("delta")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/data")
    .outputMode("append")
    .trigger(availableNow=True)
    .toTable(BRONZE_TABLE)
)
query.awaitTermination()

log("raw_data_storage", "INFO", f"Auto Loader appended batch into {BRONZE_TABLE}")
print(f"Bronze load complete. Row count: {spark.table(BRONZE_TABLE).count()}")

# COMMAND ----------
# Step 4 — verify lineage replaces the JSON catalog (UI check, not executable code)
# In Catalog Explorer UI: churn_pilot > bronze > customer_churn_raw > Lineage tab
# Expected: shows 10_ingestion_to_landing.py's landing volume as the upstream source —
# this is the UC-native equivalent of data/raw/data_catalog.json
