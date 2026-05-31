-- 01_unity_catalog_setup.sql
-- Creates the catalog and the four schemas used by every later migration stage.
-- Run as a metastore admin / user with CREATE CATALOG privilege.

CREATE CATALOG IF NOT EXISTS churn_prediction
  COMMENT 'Customer churn prediction lakehouse — migrated from local pandas/SQLite/CSV pipeline';

CREATE SCHEMA IF NOT EXISTS churn_prediction.bronze
  COMMENT 'Raw ingested data — Auto Loader landing tables, pipeline logs, data-quality metrics';

CREATE SCHEMA IF NOT EXISTS churn_prediction.silver
  COMMENT 'Cleaned/encoded/feature-engineered customer records (replaces data/processed/cleaned_data.csv)';

CREATE SCHEMA IF NOT EXISTS churn_prediction.gold
  COMMENT 'Aggregated + interaction features, training sets (replaces SQLite churn_data.db)';

CREATE SCHEMA IF NOT EXISTS churn_prediction.ml
  COMMENT 'Feature Engineering tables, MLflow-linked model artifacts metadata';
