# Databricks notebook source
# 50_model_training.py
# Replaces src/build_model.py (TrainCustomModel). Training/evaluation logic is
# unchanged sklearn; only the MLflow tracking URI and model-registration target change —
# Databricks-managed MLflow + Unity Catalog Model Registry instead of
# file:///tmp/mlflow-runs + joblib.dump to data/models/.

# COMMAND ----------
# CELL 1 — assertion (run first; expect MlflowException / RestException: model does not exist)
from mlflow import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
try:
    client.get_registered_model("churn_prediction.ml.churn_model")
    raise AssertionError("Expected the UC registered model NOT to exist yet")
except Exception as e:
    assert "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower(), f"Unexpected error: {e}"
print("OK: churn_prediction.ml.churn_model not registered yet, as expected")

# COMMAND ----------
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Notebooks do not share Python variable state — whether run standalone, via %run, or
# as a job task in a multi-task workflow, each executes in its own scope. So, just like
# Task 7's 40_feature_store_setup.py, this notebook instantiates its own Feature
# Engineering Client rather than assuming `fe` exists from another notebook's context.
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/churn_prediction")

REGISTERED_MODEL_NAME = "churn_prediction.ml.churn_model"

# create_training_set replaces build_model.get_latest_training_data's "glob latest CSV"
# logic — pulls governed features from the Feature Engineering table registered in
# Task 7, point-in-time correct by construction, instead of reading an ad-hoc CSV snapshot.
training_set = fe.create_training_set(
    df=spark.table("churn_prediction.gold.customer_features").select("customerID", "Churn"),
    feature_lookups=[FeatureLookup(
        table_name="churn_prediction.ml.customer_features_fe",
        lookup_key="customerID",
        exclude_columns=["customerID", "Churn"],
    )],
    label="Churn",
    exclude_columns=["customerID"],
)
training_pdf = training_set.load_df().toPandas()

# COMMAND ----------
# load_and_split_data: drop customerID (already excluded), coerce TotalCharges, numeric-only,
# stratified 80/20 split — identical to TrainCustomModel.load_and_split_data
training_pdf["TotalCharges"] = training_pdf["TotalCharges"].apply(
    lambda v: float(v) if str(v).strip() not in ("", "nan") else 0.0
)
numeric_df = training_pdf.select_dtypes(include="number")
X = numeric_df.drop(columns=["Churn"])
y = numeric_df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


# COMMAND ----------
for model_name, model in [
    ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ("random_forest", RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)),
]:
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_metrics(metrics)

        # mlflow.sklearn.log_model(..., registered_model_name=...) replaces joblib.dump to
        # data/models/ — the model is versioned, lineage-tracked, and servable straight
        # from the Unity Catalog Model Registry instead of an ad-hoc local .joblib file.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(5),
        )
        print(f"{model_name}: {metrics}  (run_id={run.info.run_id})")

# COMMAND ----------
# Re-run as a sanity check that the model is now registered (expect this to succeed, no exception)
model_info = client.get_registered_model("churn_prediction.ml.churn_model")
assert len(model_info.latest_versions) >= 1
print(f"OK: {model_info.name} has {len(model_info.latest_versions)} version(s) registered")
