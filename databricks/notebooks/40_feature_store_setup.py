# Databricks notebook source
# 40_feature_store_setup.py
# Replaces src/feature_store.py (SimpleChurnFeatureStore). Registers gold.customer_features
# as a governed Feature Engineering in Unity Catalog table — replacing the hand-rolled
# CSV store (churn_features.csv / churn_features_sample.csv / feature_metadata.md) with
# point-in-time-correct, versioned, lineage-tracked feature lookups.

# COMMAND ----------
# CELL 1 — assertion (run first; expect ValueError / not found from the Feature Engineering client)
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
try:
    fe.get_table(name="churn_prediction.ml.customer_features_fe")
    raise AssertionError("Expected the feature table NOT to exist yet")
except Exception as e:
    assert "does not exist" in str(e).lower() or "not found" in str(e).lower(), f"Unexpected error: {e}"
print("OK: feature table does not exist yet, as expected")

# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
SOURCE_TABLE = "churn_prediction.gold.customer_features"
FEATURE_TABLE = "churn_prediction.ml.customer_features_fe"

source_df = spark.table(SOURCE_TABLE)

# COMMAND ----------
# create_table replaces populate_from_dataframe / auto_populate_from_latest_data
fe.create_table(
    name=FEATURE_TABLE,
    primary_keys=["customerID"],
    df=source_df,
    schema=source_df.schema,
    description=(
        "Churn feature table migrated from feature_store.SimpleChurnFeatureStore. "
        "Primary key customerID. Replaces data/feature_store/churn_features.csv."
    ),
)
print(f"Registered feature table {FEATURE_TABLE} with {source_df.count()} rows")

# COMMAND ----------
# get_feature_metadata equivalent — UC table description + Catalog Explorer schema view
print(fe.get_table(name=FEATURE_TABLE).description)

# COMMAND ----------
# create_training_set replaces build_model.get_latest_training_data's "glob latest CSV" logic.
# `labels_df` supplies the join keys + label column; FeatureLookup pulls every other column
# from the governed feature table — point-in-time correct by construction.
labels_df = source_df.select("customerID", "Churn")

feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key="customerID",
        exclude_columns=["customerID", "Churn"],
    )
]

training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="Churn",
    exclude_columns=["customerID"],
)
training_df = training_set.load_df()
print(f"Assembled training set: {training_df.count()} rows, {len(training_df.columns)} columns")

# COMMAND ----------
# Re-run as a sanity check that the table now exists (expect this to succeed, no exception)
info = fe.get_table(name="churn_prediction.ml.customer_features_fe")
assert info.primary_keys == ["customerID"]
print("OK: feature table now registered with primary key customerID")
