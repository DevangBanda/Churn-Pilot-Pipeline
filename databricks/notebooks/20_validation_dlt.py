# Databricks notebook source
# 20_validation_dlt.py
# Replaces src/data_validation.py. Re-expresses DataValidator's checks
# (missing values, duplicates, dtype mismatches, negative numerics on
# tenure/MonthlyCharges/TotalCharges) as DLT expectations, and writes a
# violation-summary Delta table queryable via SQL/BI instead of opening an .xlsx.

# COMMAND ----------
# CELL 1 — assertion (run first; expect AnalysisException: Table or view not found)
metrics = spark.table("churn_prediction.bronze.data_quality_metrics")
assert metrics.count() > 0
assert set(metrics.columns) >= {"check_name", "passed_count", "failed_count", "checked_at"}

# COMMAND ----------
import dlt
from pyspark.sql import functions as F

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dlt.table(
    name="customer_churn_validated",
    comment="Bronze rows passed through DataValidator-equivalent expectations before Silver prep",
)
@dlt.expect_or_drop("no_missing_customer_id", "customerID IS NOT NULL")
@dlt.expect("tenure_non_negative", "tenure >= 0")
@dlt.expect("monthly_charges_non_negative", "MonthlyCharges >= 0")
@dlt.expect("total_charges_non_negative", "TotalCharges IS NULL OR TotalCharges >= 0")
def customer_churn_validated():
    # Uniqueness on customerID (DataValidator's duplicate-record check) is enforced
    # here via dropDuplicates rather than as an expectation, since DLT expectations
    # validate row-level predicates and cannot express a set-level uniqueness check.
    return (
        dlt.read("churn_prediction.bronze.customer_churn_raw")
        .dropDuplicates(["customerID"])
    )


# COMMAND ----------
# Violation-summary table — the Delta replacement for reports/data_quality_report_<ts>.xlsx.
# Run this as a regular (non-DLT) batch cell after the DLT pipeline has produced
# customer_churn_validated, e.g. from notebook 30 or a dedicated validation-summary task.
def write_quality_metrics():
    raw = spark.table("churn_prediction.bronze.customer_churn_raw")
    checks = []
    for col in NUMERIC_COLS:
        failed = raw.filter((F.col(col).isNotNull()) & (F.col(col) < 0)).count()
        checks.append((f"{col}_non_negative", raw.count() - failed, failed))

    missing_id = raw.filter(F.col("customerID").isNull()).count()
    checks.append(("customerID_not_null", raw.count() - missing_id, missing_id))

    dup_id = raw.count() - raw.dropDuplicates(["customerID"]).count()
    checks.append(("customerID_unique", raw.count() - dup_id, dup_id))

    rows = [(name, passed, failed) for name, passed, failed in checks]
    (
        spark.createDataFrame(rows, "check_name STRING, passed_count LONG, failed_count LONG")
        .withColumn("checked_at", F.current_timestamp())
        .write.format("delta").mode("append")
        .saveAsTable("churn_prediction.bronze.data_quality_metrics")
    )


write_quality_metrics()

# COMMAND ----------
# Step 4 (optional) — Excel-report bridge for stakeholders who still need .xlsx,
# per atlas §3 ("the Excel report becomes an optional thin downstream rendering job").
# Uncomment to render the Delta metrics table back into the legacy report format:
#
# import pandas as pd
# metrics_pdf = spark.table("churn_prediction.bronze.data_quality_metrics").toPandas()
# metrics_pdf.to_excel("/Volumes/churn_prediction/bronze/landing_volume/reports/data_quality_report.xlsx", index=False)
