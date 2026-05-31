-- 02_external_location_setup.sql
-- Replaces the AWS_ACCESS_KEY_ID/SECRET env vars in raw_data_storage.py / .env
-- with a governed Storage Credential + External Location.
-- <AWS_ROLE_ARN> = an IAM role the Databricks account can assume (set up once in AWS + account console).
-- <S3_BUCKET> = the bucket already referenced by data_versioning.setup_s3_remote(), e.g. churn-data-lake

CREATE STORAGE CREDENTIAL IF NOT EXISTS churn_s3_credential
  WITH (AWS_IAM_ROLE = '<AWS_ROLE_ARN>')
  COMMENT 'IAM role for churn pipeline S3 access — replaces embedded AWS keys in .env';

CREATE EXTERNAL LOCATION IF NOT EXISTS churn_landing_location
  URL 's3://<S3_BUCKET>/landing'
  WITH (STORAGE_CREDENTIAL churn_s3_credential)
  COMMENT 'Landing zone for raw CSV/JSON before Auto Loader ingestion into bronze';

-- UC Volume on top of the external location — the Databricks-native replacement
-- for data/raw/<ts>.csv and data/raw/sources/.../ partitioned local copies.
CREATE VOLUME IF NOT EXISTS churn_prediction.bronze.landing_volume
  COMMENT 'Landing volume for raw ingested files (replaces local data/raw/ tree)';
