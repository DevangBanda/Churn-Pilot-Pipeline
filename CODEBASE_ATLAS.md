# Codebase Atlas — ChurnPilot ML Pipeline

> Reference doc for understanding this repo and planning its migration to Databricks.
> Generated from a full read of the source tree on 2026-06-07.

## 0. What this project is

A **local, end-to-end "Data Management for ML" pipeline** built for a BITS Pilani
DM4ML academic assignment (`docs/14_Assignment_Instructions.md`). It implements all
10 textbook stages — problem formulation → ingestion → raw storage → validation →
preparation → transformation/storage → feature store → versioning → model training →
orchestration — for the **IBM Telco Customer Churn dataset** (7,043 rows, 21 columns,
binary `Churn` target).

Everything currently runs **locally on a single machine** (or in Docker), wired
together by `main_pipeline.py`, with an equivalent **Apache Airflow DAG** as the
orchestration layer, **DVC + Git** for data versioning, **SQLite** as the "database",
CSV files as the "feature store", and **MLflow with a local file-based tracking URI**
for experiment tracking.

---

## 1. Folder & File Reference

### Root
| Path | Purpose |
|---|---|
| `main_pipeline.py` | Single entry point. Runs all 9 steps sequentially (ingestion → storage → validation → preparation → transformation → feature store → versioning → model training) and prints a summary report. |
| `README.md` | Project overview, dataset description, quick-start (venv / Docker), expected model performance targets. |
| `requirements.txt` | All Python deps: pandas/numpy/scikit-learn, mlflow, dvc/dvc-s3, apache-airflow, boto3, sqlalchemy, openpyxl, matplotlib/seaborn, etc. |
| `Dockerfile` | `python:3.12-slim` image; installs AWS CLI, sqlite3, DVC; seeds the SQLite DB from `database/init.sql`; symlinks `config/dvc/*` to project root; default command `python main_pipeline.py`. |
| `docker-compose.yml` | Defines 5 services: `pipeline` (runs the pipeline), `dvc-setup` (one-shot DVC init, profile `setup`), and an Airflow trio — `airflow-init`, `airflow-webserver`, `airflow-scheduler`. Bind-mounts `data/`, `logs/`, `reports/`, `config/`, `.dvc/`. |
| `.env.example` / `config/env/.env.example` | Template env vars: `STORAGE_TYPE`, `AWS_*`, `S3_BUCKET_NAME`, `DB_PATH`, MLflow URI, Airflow admin creds. |
| `dvc.yaml` / `dvc.lock` (and `config/dvc/*` mirrors) | DVC pipeline definition: a single `full_pipeline` stage (`python main_pipeline.py`) declaring `src/*.py` as deps and `data/raw|processed|feature_store|eda|models`, `reports/` as outs. |
| `.dvc/`, `.dvcignore` | DVC repo metadata/config (cache type = `copy`). |

### `src/` — pipeline source code (imported by `main_pipeline.py` and the Airflow DAG)
| File | Role | Reads | Writes |
|---|---|---|---|
| `data_ingestion.py` → `DataIngestionPipeline` | **Step 2.** Pulls the Telco CSV from a public GitHub raw URL and a sample from the Hugging Face Datasets Server API (JSON, with retry/backoff and cached-file fallback). | Public internet (GitHub raw, HF API) | `data/raw/customer_churn_<ts>.csv`, `data/raw/huggingface_churn_<ts>.json` |
| `raw_data_storage.py` → `RawDataStorage` | **Step 3.** Copies ingested files into a partitioned layout `data/raw/sources/{source}/{data_type}/YYYY/MM/DD/...`; optionally mirrors to S3 when `STORAGE_TYPE=cloud` (boto3); writes a JSON file catalog. | files from Step 2 | partitioned copies + `data/raw/data_catalog.json` (+ optional S3 mirror) |
| `data_validation.py` → `DataValidator` | **Step 4.** Checks missing values, duplicates, dtypes, negative numerics on the latest CSV/JSON; emits a multi-sheet Excel quality report. | latest `data/raw/*` files | `reports/data_quality_report_<ts>.xlsx` |
| `data_preparation.py` → `DataPreparationPipeline` | **Step 5.** Median/mode imputation, one-hot encoding, `Churn`→0/1 mapping, derived features (`tenure_group`, `charges_per_tenure`, `total_to_monthly_ratio`, `avg_monthly_charges`), IQR outlier capping, `StandardScaler`, EDA plots (pie/heatmap/histograms/boxplots). | latest raw CSV | `data/eda/{raw,cleaned}/*.png` + `summary_stats.csv`, `data/processed/cleaned_data.csv` (+ `_scaled` variant) |
| `data_transformation_storage.py` → `DataTransformationStorage` | **Step 6.** Builds aggregated features (`total_services`, `service_density`, `customer_value_segment`, `tenure_stability`, `high_risk_payment`), interaction features (`tenure_monthly_interaction`, etc.), applies `StandardScaler`/`MinMaxScaler`, persists everything to a **SQLite** DB (`customer_features`, `feature_metadata`, `training_sets` tables + indexes) and writes a versioned training-set CSV. | latest cleaned/processed CSV | `data/processed/churn_data.db` (SQLite), `data/processed/training_sets/<set_id>.csv` |
| `feature_store.py` → `SimpleChurnFeatureStore` | **Step 7.** A hand-rolled, CSV-based "feature store": auto-discovers latest training set, writes `churn_features.csv` (+ 100-row sample), exposes `get_features(entity_id)`, `get_training_dataset()`, `get_feature_metadata()` (df/Markdown), and a feature-summary helper. | latest `data/processed/training_sets/*.csv` | `data/feature_store/churn_features.csv`, `churn_features_sample.csv`, `feature_metadata.md` |
| `data_versioning.py` → `DVCVersioning`, `version_pipeline_step()` | **Step 8.** Wraps the `dvc` and `git` CLIs via `subprocess`: init/configure DVC, add data to tracking, commit + tag versions, list/checkout versions, push/pull to an S3 remote (`s3://<bucket>/dvc-storage`). `version_pipeline_step()` is the convenience hook called after each pipeline stage to create a tagged Git commit. | git/DVC state, `.env` (S3 creds) | Git commits/tags, `.dvc` files, optional S3 remote pushes |
| `build_model.py` → `TrainCustomModel` | **Step 9.** Loads the latest training set, drops `customerID`, coerces `TotalCharges`, keeps numeric features, stratified 80/20 split, trains `LogisticRegression` or `RandomForestClassifier` (`class_weight='balanced'`), computes accuracy/precision/recall/F1, **logs params/metrics/model to MLflow** (`mlflow.set_tracking_uri(...)`, default `file:///tmp/mlflow-runs`, experiment `"ChurnPilot"`), and saves a `.joblib` artifact. | latest training set CSV | `data/models/logreg_model_<ts>.joblib`, MLflow run (local file store) |
| `utils/logger.py` | Centralized `PipelineLogger` — caches one logger per pipeline name, dual file+console handlers, writes to `logs/<pipeline_name>.log`. `PIPELINE_NAMES` maps step → log filename. | — | `logs/*.log` |
| `setup_dvc.sh`, `startup.sh`, `airflow_docker/start_airflow_docker.sh` | Shell helpers: install/init DVC + configure S3 remote from `.env`; Docker container entrypoint dispatcher (`dvc-setup` / `airflow-init` / `airflow-webserver` / `airflow-scheduler` / default pipeline run); convenience script to (re)build & launch the Airflow Docker stack. | — | — |

### `airflow/`
| File | Purpose |
|---|---|
| `dags/churn_pilot_pipeline.py` | Airflow DAG `churn_pilot_pipeline` — **9 sequential `PythonOperator` tasks** (`data_ingestion → raw_data_storage → data_validation → data_preparation → data_transformation → feature_store → data_versioning → model_building → pipeline_success`), each spawning a subprocess that imports and calls the matching `src/*` class. Scheduled every 6 hours, `catchup=False`, 1 retry. This is a **near 1:1 mirror of `main_pipeline.py`**, just orchestrated task-by-task instead of as one script. |
| `setup_airflow.py` | Initializes the Airflow metadata DB and creates the admin user (`AIRFLOW_WWW_USER_USERNAME/PASSWORD`, default `admin`/`admin`). |
| `airflow.cfg`, `simple_auth_manager_passwords.json.generated`, `logs/` | Standard Airflow runtime config/state (local SQLite-backed Airflow metastore by default). |

### `database/`
| File | Purpose |
|---|---|
| `init.sql` | DDL for the SQLite schema (`customer_features`, `feature_metadata`, `training_sets`, indexes, seed rows for `feature_metadata`). Mirrors what `DataTransformationStorage.setup_database()` creates programmatically — used to pre-seed the DB image at Docker build time. |

### `data/` (DVC-tracked outputs — mostly empty/sample in the checked-in tree)
`raw/`, `cleaned/`, `processed/` (incl. `training_sets/`, `churn_data.db`), `feature_store/`,
`eda/{raw,cleaned}/*.png`, `models/`. These are pipeline **outputs**, regenerated on each run
and tracked by DVC (declared as `outs:` in `dvc.yaml`).

### `docs/`
Numbered write-ups mapping 1:1 to assignment tasks (`01_Problem_Formulation` … `14_Assignment_Instructions`),
plus `project_guide.md` (file-by-file overview) and `build_model_service.md` (model-training docs).
`12_Cloud_Setup.md` documents a manual **EC2 + Airflow + S3** deployment recipe (not automated).
`13_Assignment_Hints.md` / `14_Assignment_Instructions.md` are the **course assignment brief** —
useful for understanding *why* the pipeline is shaped the way it is, not for migration planning.

### `logs/`, `reports/`
Runtime artifacts: per-stage `.log` files (via `utils/logger.py`) and the Excel data-quality
report from the validation stage.

---

## 2. End-to-End Pipeline Flow

```
 ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
 │ 2. Ingestion│──▶│ 3. Raw Storage│──▶│4. Validation│──▶│5. Preparation│
 │ (CSV + HF   │   │ (partitioned  │   │ (Excel QA   │   │ (impute,     │
 │  API → CSV/ │   │  copy + JSON  │   │  report)    │   │  encode, FE, │
 │  JSON files)│   │  catalog +    │   │             │   │  outliers,   │
 │             │   │  optional S3) │   │             │   │  scale, EDA) │
 └─────────────┘   └──────────────┘   └────────────┘   └──────┬───────┘
                                                                │ data/processed/cleaned_data.csv
                                                                ▼
 ┌────────────────┐   ┌───────────────┐   ┌──────────────────────────────┐
 │ 6. Transform & │──▶│ 7. Feature    │──▶│ 9. Model Training            │
 │   Store        │   │   Store       │   │ (LogReg / RandomForest,      │
 │ (agg/interact  │   │ (CSV-based,   │   │  MLflow tracking + registry, │
 │  features →    │   │  get_features,│   │  joblib artifact)            │
 │  SQLite DB +   │   │  metadata)    │   │                              │
 │  training set) │   │               │   │                              │
 └────────────────┘   └───────────────┘   └──────────────────────────────┘
        ▲                                            ▲
        └──────────── 8. Versioning (DVC + Git tag after every stage) ───┘

 Orchestration: main_pipeline.py (single script) == Airflow DAG (9 sequential tasks)
```

Concretely, the **filesystem hand-off chain** every run relies on (`glob` + "latest by
mtime/ctime" lookups — there is no explicit run-ID threading between stages):

1. `data/raw/customer_churn_*.csv` + `huggingface_churn_*.json` (Ingestion)
2. → `data/raw/sources/.../*.csv|json` + `data_catalog.json` (Raw storage)
3. → `reports/data_quality_report_*.xlsx` (Validation — reads raw, doesn't gate downstream)
4. → `data/processed/cleaned_data.csv` (+ `_scaled`) and `data/eda/**` (Preparation)
5. → `data/processed/churn_data.db` (SQLite) + `data/processed/training_sets/<set_id>.csv` (Transformation)
6. → `data/feature_store/churn_features.csv` (+ sample, metadata) (Feature Store)
7. → Git commit/tag per stage, optional `dvc push` to S3 (Versioning — interleaved, not a final stage)
8. → `data/models/logreg_model_<ts>.joblib` + MLflow run under `file:///tmp/mlflow-runs` (Model Training)

Both `main_pipeline.py` and the Airflow DAG execute **the exact same stage classes** —
the DAG just wraps each one in a subprocess + `PythonOperator` for scheduling/retries/observability.

---

## 3. What should stay local (or local-equivalent) regardless of migration

- **Developer inner loop**: editing `src/*.py`, unit-level debugging, EDA notebooks (Jupyter is
  in `requirements.txt`), and quick local repro via `docker-compose up`.
- **Docker/Compose stack** for local integration testing of the *whole* pipeline before pushing
  changes — keep it as a CI/dev sanity check even after the prod path moves to Databricks.
- **Secrets/config templates**: `.env.example`, `config/env/.env.example` — useful scaffolding
  for local dev regardless of where prod runs (never the actual secrets).
- **`requirements.txt`-driven local venv** for fast iteration on transformation/feature logic
  before porting it into notebooks/jobs.
- **Course-assignment artifacts** (`docs/13_*`, `docs/14_*`, `README.md` "Educational purposes
  only" framing) — these describe *why* the repo looks the way it does and won't migrate; keep
  them as historical context.
- **Excel-based validation report generation** (`openpyxl` output) *if* the deliverable format
  must remain a human-readable `.xlsx` for stakeholders — though the underlying checks themselves
  migrate cleanly (see below).

## 4. What can migrate to Databricks

Almost the entire **data + ML path** is a strong candidate — it's already modular,
stage-based, and the "glue" (glob-the-latest-file, SQLite, CSV feature store, subprocess-driven
DVC/Airflow) is exactly the kind of local-only scaffolding Databricks primitives replace outright:

| Local component | Databricks replacement |
|---|---|
| `data_ingestion.py` (requests → CSV/JSON on local disk) | Notebook/job task writing to a **Bronze Delta table** via Spark `read` + Auto Loader (for recurring/incremental loads) |
| `raw_data_storage.py` (partitioned local copy + JSON catalog + optional S3 mirror) | **Unity Catalog Volumes** for any raw files that must stay file-shaped, or directly **Delta tables** registered in UC (catalog = the "data catalog" requirement, natively, with lineage) |
| `data_validation.py` (pandas checks → Excel report) | **Lakehouse Monitoring** / **Delta Live Tables expectations** (`@dlt.expect_*`) for declarative quality gates, with violations queryable as Delta tables (Excel export, if still required, becomes a thin downstream job) |
| `data_preparation.py` (pandas clean/encode/FE/scale + matplotlib EDA) | **PySpark / pandas-on-Spark** transformations writing a **Silver Delta table**; EDA via Databricks notebooks + built-in visualizations or `dbutils.data.summarize` |
| `data_transformation_storage.py` (SQLite + manual schema/indexes) | **Gold Delta tables** in Unity Catalog — schema, indexing (Z-ORDER/liquid clustering), and metadata (table/column comments, UC lineage) replace `feature_metadata`/`training_sets` tables wholesale |
| `feature_store.py` (hand-rolled CSV store) | **Databricks Feature Engineering in Unity Catalog** (feature tables + `FeatureEngineeringClient`, point-in-time lookups, online/offline serving) |
| `data_versioning.py` (DVC + Git subprocess wrapper, S3 remote) | **Delta Lake time travel** (`VERSION AS OF` / `TIMESTAMP AS OF`) + **Unity Catalog lineage** — eliminates the entire DVC/S3/Git-tag dance for data; Git remains for *code* versioning only |
| `build_model.py` (sklearn + local MLflow file store + joblib) | **Databricks-managed MLflow** (`databricks` tracking URI), **Unity Catalog Model Registry** (`models:/catalog.schema.model`), optionally distributed training via **Spark ML** / `mlflow.pyfunc` for serving |
| `airflow/dags/churn_pilot_pipeline.py` (Airflow DAG, subprocess-per-task) | **Databricks Workflows** (Jobs with task graphs, retries, alerting) — same 9-node DAG shape, but tasks become notebook/Python-wheel/SQL tasks instead of `subprocess.run([sys.executable, '-c', ...])` |
| `database/init.sql` (SQLite DDL) | UC table DDL (`CREATE TABLE catalog.schema.table ... USING DELTA`), generated/managed via notebooks or Databricks Asset Bundles |
| S3 mirror logic in `raw_data_storage.py` / `data_versioning.setup_s3_remote()` | **External Locations + Storage Credentials** in Unity Catalog (governed access to the same S3 buckets, no embedded AWS keys) |

---

## 5. Step-by-Step Migration Plan → Databricks (Delta Lake + MLflow + Unity Catalog + Workflows)

### Phase 0 — Foundations
1. Stand up/confirm a **Unity Catalog metastore**; create a catalog (e.g. `churn_pilot`)
   with `bronze`, `silver`, `gold`, and `ml` schemas.
2. Create **External Locations + Storage Credentials** pointing at the existing S3 bucket
   (`churn-data-lake`) so raw landing files can be read without embedding AWS keys (replaces
   `AWS_ACCESS_KEY_ID/SECRET` env-var plumbing in `raw_data_storage.py` / `.env`).
3. Set up a **Databricks Repo** synced to this Git repo so `src/*.py` logic can be imported
   into notebooks/jobs as a package (minimal rewrite — keep the transformation *logic*,
   replace the pandas/SQLite/file-glob *plumbing*).
4. Create a dedicated **service principal** for job runs; grant it `USE CATALOG`, `USE SCHEMA`,
   `CREATE TABLE`, `MODIFY` on the new schemas, and `READ FILES`/`WRITE FILES` on the external
   location.

### Phase 1 — Ingestion → Bronze (replaces `data_ingestion.py` + `raw_data_storage.py`)
5. Port `ingest_csv_data`/`ingest_huggingface_data` into a notebook/task that fetches the
   same two sources and writes raw payloads to a **landing Volume** (UC Volume backed by the
   external S3 location) instead of `data/raw/*.csv|json` on local disk.
6. Add an **Auto Loader** (`cloudFiles`) stream or scheduled batch read over that Volume,
   writing append-only to `bronze.customer_churn_raw` (Delta). This single step subsumes both
   "ingestion" and "raw storage/cataloging" — UC's table metadata + lineage *is* the catalog
   that `create_data_catalog()` hand-built as JSON.
7. Carry over the retry/backoff and logging *behavior* (re-implement with Python `logging` →
   Databricks job/task logs, or structured logs to a `bronze.pipeline_logs` Delta table —
   replacing `utils/logger.py`'s per-file logs).

### Phase 2 — Validation (replaces `data_validation.py`)
8. Re-express the missing/duplicate/dtype/negative-value checks as **Delta Live Tables
   expectations** (`@dlt.expect`, `@dlt.expect_or_drop`, `@dlt.expect_or_fail`) on the
   Bronze→Silver boundary, or as a **Lakehouse Monitoring** profile on `bronze.customer_churn_raw`.
9. Persist violation summaries to a `bronze.data_quality_metrics` Delta table (queryable via
   SQL/BI instead of opening an `.xlsx`); keep an optional notebook that renders the same
   Excel format from that table if the Excel deliverable is still required by stakeholders.

### Phase 3 — Preparation & Transformation → Silver/Gold (replaces `data_preparation.py` + `data_transformation_storage.py`)
10. Rewrite the pandas transformations (`handle_missing_values`, `encode_categorical`,
    `engineer_features`, `cap_outliers`, `scale_features`, aggregated/interaction features,
    scaling) using **PySpark DataFrame** ops or **pandas-on-Spark**, preserving the exact
    feature-engineering logic (this is the most copy-paste-friendly part — the *math* doesn't
    change, only the dataframe API).
11. Write the cleaned/encoded result to `silver.customer_churn_clean`, and the
    aggregated/interaction/scaled feature set to `gold.customer_features` — both Delta tables
    in Unity Catalog. This eliminates `data/processed/churn_data.db` (SQLite),
    `feature_metadata`/`training_sets` tables (replaced by UC table comments + Delta history),
    and the manual `database/init.sql` DDL.
12. Reproduce EDA outputs (`save_eda_plots`) as Databricks notebook cells with native
    visualizations, or persist summary stats to `gold.eda_summary` for dashboarding.

### Phase 4 — Feature Store (replaces `feature_store.py`)
13. Register `gold.customer_features` as a **Feature Engineering in Unity Catalog** feature
    table (`FeatureEngineeringClient.create_table(...)`) with `customerID` as the primary key —
    this directly replaces `populate_from_dataframe`, `get_features`, `get_training_dataset`,
    and `get_feature_metadata` with governed, versioned, point-in-time-correct equivalents.
14. Use `FeatureEngineeringClient.create_training_set(...)` to assemble training data via
    feature lookups (replacing the "find latest training_sets CSV" glob logic in
    `build_model.py.get_latest_training_data`).

### Phase 5 — Versioning (replaces `data_versioning.py` / DVC)
15. Drop DVC entirely for *data*: rely on **Delta Lake transaction log + time travel**
    (`DESCRIBE HISTORY`, `VERSION AS OF`) for reproducibility, and **Unity Catalog lineage**
    (Catalog Explorer → Lineage tab) for the source→feature→model audit trail that
    `version_pipeline_step()` approximated with Git commits/tags.
16. Keep Git/Repos for **code** versioning only (DAG/notebook/package source) — this is the
    one place the existing tool (Git) stays, just scoped down from "data + code" to "code only".

### Phase 6 — Model Training (replaces `build_model.py`)
17. Point `mlflow.set_tracking_uri("databricks")` (or omit — it's the default in a Databricks
    notebook) and `mlflow.set_experiment("/Shared/churn_pilot")`; keep the
    `LogisticRegression`/`RandomForestClassifier` training + `accuracy/precision/recall/f1`
    evaluation logic as-is (pure sklearn, portable verbatim).
18. Replace `joblib.dump` + local `data/models/` with `mlflow.sklearn.log_model(...,
    registered_model_name="churn_pilot.ml.churn_model")` targeting the **Unity Catalog
    Model Registry**, giving you governed model versions, aliases (`@champion`/`@challenger`),
    and approval workflows out of the box.
19. (Optional, post-migration enhancement) swap to `mlflow.pyfunc` + **Model Serving** for a
    real-time inference endpoint — the assignment's "deployable model" deliverable becomes a
    living REST endpoint instead of a `.joblib` file.

### Phase 7 — Orchestration (replaces `main_pipeline.py` + Airflow DAG)
20. Recreate the 9-node dependency chain
    (`ingestion → raw_storage/bronze → validation → preparation → transformation → feature_store
    → versioning(n/a) → model_building → success`) as a **Databricks Workflow (Job)** with one
    task per stage (notebook or Python-wheel tasks), `depends_on` edges mirroring the existing
    `>>` chain in `churn_pilot_pipeline.py`.
21. Carry over the DAG's operational settings 1:1 — `retries=1`, `retry_delay=5min`,
    `schedule=timedelta(hours=6)`, `catchup=False`, `max_active_runs=1`, failure
    notifications — using native Workflows job-level retry/schedule/alert config (drop the
    Airflow webserver/scheduler/Postgres stack and `airflow-init`/`docker-compose` services
    entirely).
22. Optionally manage the Job + UC objects as code via a **Databricks Asset Bundle** (`databricks.yml`)
    so the whole target environment (catalog, schemas, feature tables, job, permissions) is
    reproducible from source control — the spiritual successor to `dvc.yaml` + `docker-compose.yml`.

### Phase 8 — Cutover & validation
23. Run the new Workflow side-by-side with the legacy `main_pipeline.py`/Airflow path for one
    or two cycles; diff record counts, churn-rate distributions, and model metrics between the
    SQLite/CSV outputs and the new Delta/UC outputs to confirm parity.
24. Decommission Airflow containers, SQLite DB, CSV feature store, and DVC/S3 remote once the
    Databricks path has matched outputs for N consecutive runs; keep `docker-compose.yml`
    around only as a local dev/test harness for `src/*` unit-level changes (per §3).
