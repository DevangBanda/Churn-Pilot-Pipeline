-- 03_grants.sql
-- Run as account/metastore admin. <SP_APPLICATION_ID> = the service principal's
-- Application ID, created once via Account Console > User management > Service principals.

GRANT USE CATALOG ON CATALOG churn_prediction TO `<SP_APPLICATION_ID>`;
GRANT USE SCHEMA ON SCHEMA churn_prediction.bronze TO `<SP_APPLICATION_ID>`;
GRANT USE SCHEMA ON SCHEMA churn_prediction.silver TO `<SP_APPLICATION_ID>`;
GRANT USE SCHEMA ON SCHEMA churn_prediction.gold   TO `<SP_APPLICATION_ID>`;
GRANT USE SCHEMA ON SCHEMA churn_prediction.ml     TO `<SP_APPLICATION_ID>`;

GRANT CREATE TABLE, MODIFY, SELECT ON SCHEMA churn_prediction.bronze TO `<SP_APPLICATION_ID>`;
GRANT CREATE TABLE, MODIFY, SELECT ON SCHEMA churn_prediction.silver TO `<SP_APPLICATION_ID>`;
GRANT CREATE TABLE, MODIFY, SELECT ON SCHEMA churn_prediction.gold   TO `<SP_APPLICATION_ID>`;
GRANT CREATE TABLE, CREATE MODEL, MODIFY, SELECT ON SCHEMA churn_prediction.ml TO `<SP_APPLICATION_ID>`;

GRANT READ FILES, WRITE FILES ON EXTERNAL LOCATION churn_landing_location TO `<SP_APPLICATION_ID>`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME churn_prediction.bronze.landing_volume TO `<SP_APPLICATION_ID>`;
