# Final Whole-Implementation Review — Databricks Migration (12-task)

Reviewed: all 3 setup SQL files, all 9 notebooks (10→70), `databricks.yml`, `churn_pipeline_job.yml`.

## Cross-file naming/lineage chain verified

| Stage | Table/object | Created by | Read by |
|---|---|---|---|
| Landing volume | `/Volumes/churn_pilot/bronze/landing_volume` | `02_external_location_setup.sql` | `10_ingestion_to_landing.py`, `11_autoloader_bronze.py` |
| Bronze | `churn_pilot.bronze.customer_churn_raw` | `11_autoloader_bronze.py` | `20_validation_dlt.py` (DLT `dlt.read(...)` and `write_quality_metrics`) |
| Bronze (DQ) | `churn_pilot.bronze.data_quality_metrics` | `20_validation_dlt.py` | (terminal — BI/report bridge) |
| Bronze (DLT) | `churn_pilot.bronze.customer_churn_validated` | DLT table decorator `name="customer_churn_validated"` in `20_validation_dlt.py` | `30_preparation_silver.py` (`SOURCE_TABLE`) |
| Silver | `churn_pilot.silver.customer_churn_clean` | `30_preparation_silver.py` | `31_transformation_gold.py` (`SOURCE_TABLE`) |
| Gold | `churn_pilot.gold.customer_features` | `31_transformation_gold.py` | `40_feature_store_setup.py` (`SOURCE_TABLE`), `50_model_training.py`, `70_parity_check.py` (`GOLD_TABLE`) |
| FE table | `churn_pilot.ml.customer_features_fe` | `40_feature_store_setup.py` | `50_model_training.py` (`FeatureLookup.table_name`) |
| UC Model | `churn_pilot.ml.churn_model` | `50_model_training.py` (`REGISTERED_MODEL_NAME`) | `60_model_serving.py` (`MODEL_NAME`), `70_parity_check.py` |
| Serving endpoint | `churn-pilot-endpoint` | `60_model_serving.py` | (terminal, optional) |

All names match exactly end-to-end — no drift, no typos. Catalog/schema prefixes (`bronze`/`silver`/`gold`/`ml`) are consistent with `01_unity_catalog_setup.sql`.

## Issues found

### CRITICAL — DLT table `customer_churn_validated` is wired as a plain notebook task, not a DLT pipeline
`20_validation_dlt.py` is a **DLT** notebook (`import dlt`, `@dlt.table(name="customer_churn_validated", ...)`, `@dlt.expect*`). DLT-decorated functions only execute and materialize a table when run **inside a DLT pipeline** (a `pipelines` resource with `target_catalog`/`target_schema`, referenced from a job via `pipeline_task`). 
However:
- There is **no `pipelines:` resource block** anywhere in `databricks/bundle/` (only `churn_pipeline_job.yml` exists in `resources/`).
- `churn_pipeline_job.yml`'s `validation` task uses a plain `notebook_task: notebook_path: ../../notebooks/20_validation_dlt.py` — running this notebook as an ordinary notebook task will not materialize `churn_pilot.bronze.customer_churn_validated` (the `@dlt.table` decorator is a no-op outside a pipeline context; `dlt.read`/`dlt.table` calls will raise outside DLT runtime).
- Consequently `30_preparation_silver.py`'s `SOURCE_TABLE = "churn_pilot.bronze.customer_churn_validated"` will not resolve when the job chain runs end-to-end — the `preparation_silver` task will fail with "table not found," breaking the entire downstream chain (gold, feature store, model training, parity).
- The `write_quality_metrics()` batch cell in the same notebook (which produces `bronze.data_quality_metrics`, the table CELL 1's assertion checks) is fine running as a plain notebook task — but it's bundled in the same file as the DLT table definition, compounding the orchestration mismatch.

This is a genuine end-to-end coherence break that is invisible when reviewing `20_validation_dlt.py` or `churn_pipeline_job.yml` individually — it only surfaces when tracing the full chain "how does the DLT table actually get created when the job runs."

**Fix options**: (a) add a `pipelines:` resource (with `target_catalog: churn_pilot`, `target_schema: bronze`, `libraries: [{notebook: {path: ../../notebooks/20_validation_dlt.py}}]`) and change the `validation` task to a `pipeline_task: {pipeline_id: ...}`, or (b) split the notebook into a DLT-pipeline-only file plus a separate plain-notebook quality-metrics file and wire the pipeline + the metrics notebook as two tasks.

### Minor — header comment slightly overstates orchestration reality
`20_validation_dlt.py`'s header says "writes a violation-summary Delta table queryable via SQL/BI" and frames the whole notebook as the validation stage, but doesn't disclose (the way other notebooks do, e.g. `31_transformation_gold.py`'s "Task 12 parity" notes) that the `@dlt.table` portion requires a DLT pipeline wrapper that the bundle doesn't yet define. A one-line note here (or in the job YAML) would have made the gap self-documenting.

### Minor — `write_quality_metrics` comment says "Run this as a regular (non-DLT) batch cell ... e.g. from notebook 30" 
This is honest about the DLT/non-DLT split inside the file, but it underlines that the file mixes two execution models (DLT pipeline table + plain batch cell) inside one notebook that the job then runs as a single plain `notebook_task` — neither model is fully satisfied by that wiring.

## Things confirmed correct (no issues)

- **CELL 1 assertion convention**: present and structurally consistent across all 9 notebooks — each asserts on the exact table/object that notebook is responsible for creating (landing files / bronze.customer_churn_raw / bronze.data_quality_metrics / silver.customer_churn_clean / gold.customer_features / ml.customer_features_fe / ml.churn_model / serving endpoint / legacy_export CSV). Comment style ("CELL 1 — assertion (run first; expect ...)") and red-phase exception types are consistent and plausible per API.
- **Asset Bundle `notebook_path` resolution**: all 8 paths (`../../notebooks/10_..._landing.py` ... `70_parity_check.py`) correctly resolve from `databricks/bundle/resources/` to `databricks/notebooks/*.py`. All 9 notebook files exist; `60_model_serving.py` is correctly the only one NOT referenced — consistent with its "optional" status.
- **Job task chain vs. notebook order**: `ingestion(10) → raw_storage_bronze(11) → validation(20) → preparation_silver(30) → transformation_gold(31) → feature_store(40) → model_building(50) → pipeline_success(70)` — order and substance match the Airflow DAG's task_ids (`data_ingestion → raw_data_storage → data_validation → data_preparation → data_transformation → feature_store → [data_versioning omitted, documented] → model_building → pipeline_success`) and the notebook numbering. The `data_versioning` omission is explicitly documented inline in the YAML with sound reasoning (Delta time travel/UC lineage are passive).
- **`60_model_serving.py` optionality**: header explicitly states "Optional post-migration enhancement (atlas Phase 6, step 19)" and it correctly does not appear in the job YAML — consistent with the plan.
- **`70_parity_check.py` header claim** ("wired in as the final task ... of the Asset Bundle job") is TRUE — it is `pipeline_success`, last task, `notebook_path: .../70_parity_check.py`. `LEGACY_F1 = 0.58` placeholder confirmed intentional per task instructions.
- **No duplicated/orphaned table or model names** found anywhere in the set.

## Verdict

**Changes needed** — one Critical issue: the `validation` stage's DLT table (`churn_pilot.bronze.customer_churn_validated`) cannot actually be produced by the current job-task wiring (plain `notebook_task` instead of a DLT `pipeline_task` backed by a `pipelines:` bundle resource), which breaks the entire downstream chain that `30_preparation_silver.py` depends on. This must be fixed (add a `pipelines` resource + `pipeline_task`, or restructure the validation stage) before the branch can be considered a coherent, runnable end-to-end migration. All naming/lineage, CELL 1 conventions, bundle path resolution, and documentation cross-references are otherwise consistent and correct.
