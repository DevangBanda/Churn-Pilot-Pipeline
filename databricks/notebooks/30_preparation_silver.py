# Databricks notebook source
# 30_preparation_silver.py
# Replaces src/data_preparation.py (DataPreparationPipeline). Same operations —
# median/mode imputation, one-hot encoding, Churn -> 0/1, derived features,
# IQR outlier capping, StandardScaler — rewritten against PySpark DataFrames,
# writing to silver.customer_churn_clean instead of data/processed/cleaned_data.csv.

# COMMAND ----------
# CELL 1 — assertion (run first; expect AnalysisException: Table or view not found)
silver = spark.table("churn_pilot.silver.customer_churn_clean")
expected_engineered_cols = {"tenure_group", "charges_per_tenure", "total_to_monthly_ratio", "avg_monthly_charges"}
assert expected_engineered_cols.issubset(set(silver.columns)), \
    f"Missing engineered columns: {expected_engineered_cols - set(silver.columns)}"
assert silver.filter("Churn IS NULL").count() == 0
assert silver.filter("Churn NOT IN (0, 1)").count() == 0, "Churn must be mapped to 0/1"

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

SOURCE_TABLE = "churn_pilot.bronze.customer_churn_validated"  # output of the DLT pipeline (Task 4)
TARGET_TABLE = "churn_pilot.silver.customer_churn_clean"

df = spark.table(SOURCE_TABLE)

# COMMAND ----------
# 1. handle_missing_values: median for numeric, mode for categorical
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("double"))

medians = df.approxQuantile(numeric_cols, [0.5], 0.001)
for col_name, median_list in zip(numeric_cols, medians):
    df = df.fillna({col_name: median_list[0]})

categorical_cols = [c for c, t in df.dtypes if t == "string" and c not in ("customerID",)]
for col_name in categorical_cols:
    mode_row = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
    if mode_row is not None:
        df = df.fillna({col_name: mode_row[col_name]})

# COMMAND ----------
# 2. encode_categorical: Churn -> 0/1, one-hot the rest
df = df.withColumn("Churn", F.when(F.col("Churn") == "Yes", 1).otherwise(0))

one_hot_cols = [c for c in categorical_cols if c != "Churn"]
# NOTE: original pandas used pd.get_dummies(..., drop_first=True), which drops one
# reference category per column to reduce dimensionality/multicollinearity. This
# PySpark port creates a dummy column for every distinct value (no drop_first
# equivalent), so it produces one extra column per categorical feature versus the
# original. Acceptable for tree-based models (robust to redundant correlated dummy
# columns), but worth noting when comparing feature counts in Task 12's parity check.
for col_name in one_hot_cols:
    distinct_vals = [r[0] for r in df.select(col_name).distinct().collect()]
    for val in distinct_vals:
        safe_val = "".join(ch if ch.isalnum() else "_" for ch in str(val))
        df = df.withColumn(f"{col_name}_{safe_val}", (F.col(col_name) == val).cast("int"))
    df = df.drop(col_name)

# COMMAND ----------
# 3. engineer_features: tenure_group, charges_per_tenure, total_to_monthly_ratio, avg_monthly_charges
df = (
    # NOTE: original pandas used pd.cut(bins=[0,12,24,48,72], labels=[...]).cat.codes,
    # which yields integer codes (0/1/2/3) and, due to default right=True bins
    # (interval (0,12]), assigns an undefined code of -1 to tenure=0 customers — an
    # edge-case bug in the original. This version deliberately uses readable string
    # labels (more interpretable for a Silver "clean" table) with corrected inclusive
    # "<=" boundaries so tenure=0 customers are correctly bucketed into "0-12".
    df.withColumn(
        "tenure_group",
        F.when(F.col("tenure") <= 12, "0-12")
         .when(F.col("tenure") <= 24, "13-24")
         .when(F.col("tenure") <= 48, "25-48")
         .otherwise("49+"),
    )
    .withColumn("charges_per_tenure", F.col("MonthlyCharges") / (F.col("tenure") + F.lit(1)))
    # NOTE: original pandas used a bare `TotalCharges / MonthlyCharges` with no
    # zero-guard, which yields inf/NaN when MonthlyCharges == 0. This PySpark port
    # adds F.greatest(MonthlyCharges, 0.01) as a safe-division floor to avoid that
    # division-by-zero edge case. Values will only differ from the original for the
    # rare rows where MonthlyCharges == 0 — worth knowing about for Task 12's parity check.
    .withColumn("total_to_monthly_ratio", F.col("TotalCharges") / F.greatest(F.col("MonthlyCharges"), F.lit(0.01)))
    .withColumn("avg_monthly_charges", F.col("TotalCharges") / (F.col("tenure") + F.lit(1)))
)

# COMMAND ----------
# 4. cap_outliers: IQR capping on numeric_cols
for col_name in numeric_cols:
    q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.001)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df.withColumn(col_name, F.when(F.col(col_name) < lower, lower)
                                  .when(F.col(col_name) > upper, upper)
                                  .otherwise(F.col(col_name)))

# COMMAND ----------
# 5. scale_features: StandardScaler over numeric_cols (+ engineered numerics), suffixed _scaled
scale_input_cols = numeric_cols + ["charges_per_tenure", "total_to_monthly_ratio", "avg_monthly_charges"]
assembler = VectorAssembler(inputCols=scale_input_cols, outputCol="_features_vec")
scaler = StandardScaler(inputCol="_features_vec", outputCol="_features_scaled", withMean=True, withStd=True)
scaled = Pipeline(stages=[assembler, scaler]).fit(df).transform(df)

to_array = F.udf(lambda v: v.toArray().tolist(), "array<double>")
scaled = scaled.withColumn("_scaled_arr", to_array("_features_scaled"))
for i, col_name in enumerate(scale_input_cols):
    scaled = scaled.withColumn(f"{col_name}_scaled", F.col("_scaled_arr")[i])
df = scaled.drop("_features_vec", "_features_scaled", "_scaled_arr")

# COMMAND ----------
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)
print(f"Wrote {df.count()} rows to {TARGET_TABLE}")

# COMMAND ----------
# Replaces save_eda_plots' pie/heatmap/histogram/boxplot PNGs with native Databricks visualizations
display(df.groupBy("Churn").count())                       # -> render as pie chart in the cell's chart UI
display(df.select(*numeric_cols, "Churn").summary())        # -> equivalent of summary_stats.csv
display(df.select(*[c + "_scaled" for c in scale_input_cols]))  # -> histogram/boxplot per scaled feature
