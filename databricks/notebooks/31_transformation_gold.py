# Databricks notebook source
# 31_transformation_gold.py
# Replaces src/data_transformation_storage.py (DataTransformationStorage). Builds the
# same aggregated + interaction features and writes a single Gold Delta table —
# replacing churn_data.db (SQLite: customer_features/feature_metadata/training_sets)
# and data/processed/training_sets/<set_id>.csv wholesale. UC table comments +
# `DESCRIBE HISTORY churn_prediction.gold.customer_features` replace feature_metadata
# and training_sets respectively.

# COMMAND ----------
# CELL 1 — assertion (run first; expect AnalysisException: Table or view not found)
gold = spark.table("churn_prediction.gold.customer_features")
expected_cols = {
    "total_services", "service_density", "customer_value_segment",
    "tenure_stability", "high_risk_payment", "tenure_monthly_interaction",
}
assert expected_cols.issubset(set(gold.columns)), f"Missing: {expected_cols - set(gold.columns)}"
assert gold.select("customerID").distinct().count() == gold.count(), "customerID must be unique in gold"

# COMMAND ----------
from pyspark.sql import functions as F

SOURCE_TABLE = "churn_prediction.silver.customer_churn_clean"
TARGET_TABLE = "churn_prediction.gold.customer_features"

silver = spark.table(SOURCE_TABLE)

SERVICE_BASE_COLS = {
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
}
SERVICE_COLS = [
    c for c in silver.columns
    if c.split("_")[0] in SERVICE_BASE_COLS
    and not c.endswith(("_No", "_No_phone_service", "_No_internet_service"))
]

# COMMAND ----------
# create_aggregated_features
high_risk_flag = (
    F.when(F.col("PaymentMethod_Electronic_check") == 1, 1).otherwise(0)
    if "PaymentMethod_Electronic_check" in silver.columns
    else F.lit(0)
)

gold = (
    silver
    .withColumn("total_services", sum(F.col(c) for c in SERVICE_COLS))
    .withColumn("service_density", F.col("total_services") / F.lit(max(len(SERVICE_COLS), 1)))
    .withColumn(
        "customer_value_segment",
        F.when(F.col("MonthlyCharges") >= 80, "high_value")
         .when(F.col("MonthlyCharges") >= 40, "mid_value")
         .otherwise("low_value"),
    )
    .withColumn(
        "tenure_stability",
        F.when(F.col("tenure") >= 48, "stable")
         .when(F.col("tenure") >= 12, "developing")
         .otherwise("new"),
    )
    .withColumn("high_risk_payment", high_risk_flag)
)

# COMMAND ----------
# create_feature_interactions
gold = (
    gold
    .withColumn("tenure_monthly_interaction", F.col("tenure") * F.col("MonthlyCharges"))
    .withColumn("tenure_total_interaction", F.col("tenure") * F.col("TotalCharges"))
    .withColumn("services_value_interaction", F.col("total_services") * F.col("MonthlyCharges"))
)

# COMMAND ----------
# apply_feature_scaling (MinMax, second pass) — interaction columns
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline as MLPipeline

interaction_cols = ["tenure_monthly_interaction", "tenure_total_interaction", "services_value_interaction"]
assembler = VectorAssembler(inputCols=interaction_cols, outputCol="_interactions_vec")
minmax = MinMaxScaler(inputCol="_interactions_vec", outputCol="_interactions_scaled")
scaled = MLPipeline(stages=[assembler, minmax]).fit(gold).transform(gold)

to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")
scaled = scaled.withColumn("_arr", to_array("_interactions_scaled"))
for i, col_name in enumerate(interaction_cols):
    scaled = scaled.withColumn(f"{col_name}_scaled", F.col("_arr")[i])
gold = scaled.drop("_interactions_vec", "_interactions_scaled", "_arr")

# COMMAND ----------
(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

# Table + column comments replace the feature_metadata SQLite table
spark.sql(f"COMMENT ON TABLE {TARGET_TABLE} IS "
          "'Gold feature table — aggregated + interaction features, replaces churn_data.db'")
spark.sql(f"ALTER TABLE {TARGET_TABLE} ALTER COLUMN total_services COMMENT "
          "'Count of active services per customer (sum of one-hot service indicator columns)'")
spark.sql(f"ALTER TABLE {TARGET_TABLE} ALTER COLUMN customer_value_segment COMMENT "
          "'high_value (>=80 MonthlyCharges) / mid_value (>=40) / low_value — replaces SQLite enum lookup'")

print(f"Wrote {gold.count()} rows to {TARGET_TABLE}")

# COMMAND ----------
# DESCRIBE HISTORY is the direct replacement for the training_sets SQLite table —
# every gold rewrite is an immutable, queryable, timestamped "training set version"
display(spark.sql(f"DESCRIBE HISTORY {TARGET_TABLE}").select("version", "timestamp", "operation"))
# expect at least one row: version 0, operation = WRITE/CREATE TABLE AS SELECT
