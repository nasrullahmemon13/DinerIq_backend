# DineIQ Analytics — PySpark Big Data Processing Layer

## Executive Overview

This dedicated `pyspark/` directory contains the complete, production-grade **Apache Spark / PySpark** Big Data architecture for **DineIQ Analytics**, fully satisfying the Big Data processing layer requirements specified in the project Software Requirements Specification (SRS).

The architecture processes **1,195,503+ rows** across 11 relational operational tables, orchestrating distributed schema validation, automated data quality remediation, relational multi-way joins, a 22-feature analytical feature store, 8 production Spark SQL business queries, and a multi-algorithm MLlib machine learning benchmark.

```
RAW OPERATIONAL DATA (11 Tables, 1.2M+ Rows)
        ↓
[pyspark/ingestion]    → Schema Validation (11 Explicit StructType Schemas)
        ↓
[pyspark/cleaning]     → 15 SRS Data-Quality Rules + Quarantine Partitioning
        ↓
[pyspark/joins]        → Relational Joins & Master Analytical Cube (904,586 fact rows)
        ↓
[pyspark/features]     → 22 Analytical Features (Menu, Customer, RFM Quintiles)
        ↓
[pyspark/queries]      → 8 Production Spark SQL Queries (Registered TempViews)
        ↓
[pyspark/mllib]        → Multi-Model Benchmark (LR, DT, RF, GBT) & Champion Selection
        ↓
PARQUET DATA LAKE      → Snappy-compressed Columnar Storage in processed_data/ & parquet_data/
```

---

## Directory Structure & Component Map

```
pyspark/
├── README.md                      # Comprehensive Big Data documentation (this file)
├── session.py                     # Centralized PySpark session provider (AQE, Windows 8.3 short-paths)
├── schemas.py                     # Explicit PySpark StructType schemas for all 11 tables
├── run_pipeline.py                # Master End-to-End CLI Pipeline Runner (7 stages in ~36s)
├── verify_pyspark.py              # Rapid Health & Verification Audit (<15s)
├── sql_runner.py                  # Production Spark SQL execution engine
├── benchmark_mllib.py             # Multi-algorithm MLlib benchmark (Accuracy, F1, ROC-AUC, Throughput)
├── customer_segmentation.py       # SRS Steps 15 & 16: RFM quintile scoring & 10-factor segmentation
├── menu_classification.py         # SRS Steps 9, 10, 11: 10-dimension profitability & 4-category classification
│
├── ingestion/                     # Distributed Ingestion & Schema Contracts
│   ├── schema_definitions.py      # StructType schema declarations
│   ├── schema_validation.py       # Pre-ingestion schema conformity checker
│   ├── load_csv.py                # Strict schema-enforced CSV reader
│   ├── load_csv_inferred.py       # Dynamic schema inference utility
│   └── spark_compat.py            # High-performance dual-engine bridge
│
├── cleaning/                      # Data Quality & Remediation Subsystem
│   ├── cleaning_rules.py          # Remediation rules addressing ALL 15 SRS data-quality defects
│   ├── quarantine_handler.py      # Automated quarantine isolation partitioned by entity & failure reason
│   └── data_quality_report.py     # Markdown & JSON data quality audit generator
│
├── joins/                         # Relational Integration & Joins
│   └── join_pipeline.py           # Executes ALL 10 SRS entity relationships & builds Master Analytical Cube
│
├── features/                      # Feature Engineering Subsystem
│   ├── menu_features.py           # Menu profitability & performance metrics
│   ├── customer_features.py       # Behavioral, cadence, and lifetime spend metrics
│   ├── rfm_features.py            # Statistical quintile scoring (R_Score, F_Score, M_Score: 1-5)
│   └── save_to_parquet.py         # Columnar Snappy-compressed feature store persistence
│
├── mllib/                         # Machine Learning Benchmarking Subsystem
│   ├── model_pipeline.py          # PySpark MLlib Pipeline assembler (VectorAssembler, Scaler)
│   ├── train_models.py            # Model trainer for LR, Decision Tree, Random Forest, GBT
│   └── evaluate_models.py         # Evaluator generating accuracy, precision, recall, F1, and ROC-AUC
│
└── queries/                       # Production Spark SQL Analytical Query Suite
    ├── 01_sales_and_revenue_trends.sql
    ├── 02_category_profitability_margins.sql
    ├── 03_menu_item_performance.sql
    ├── 04_customer_spend_quartiles.sql
    ├── 05_location_benchmarks.sql
    ├── 06_kitchen_wastage_by_cause.sql
    ├── 07_promotion_effectiveness_lift.sql
    └── 08_channel_margin_economics.sql
```

---

## Technical Highlights & Windows Environment Compatibility

1. **Windows 8.3 Short-Path Resolution:**
   Apache Spark worker processes on Windows can fail when file paths contain whitespace (e.g. `C:\Users\HP 250 G9\...`). `session.py` invokes the Windows Win32 API (`kernel32.GetShortPathNameW`) to convert Python executable paths to DOS 8.3 format (e.g. `C:\Users\HP250G~1\...`), ensuring worker sockets initialize seamlessly without process crashes.

2. **Adaptive Query Execution (AQE):**
   `session.py` initializes Spark with `spark.sql.adaptive.enabled = true` and `spark.sql.adaptive.coalescePartitions.enabled = true`, dynamically tuning shuffle partitions at runtime and reducing execution latency across large aggregations.

3. **Zero Python Namespace Collisions:**
   The `pyspark/` folder deliberately omits a root `__init__.py`. This prevents Python from shadowing the installed `pyspark` library in `site-packages`, allowing scripts to freely import both official PySpark submodules (`from pyspark.sql import SparkSession`) and project-specific modules (`from session import get_spark_session`).

4. **Direct Parquet Columnar Loading:**
   To prevent Windows Hadoop `NativeIO$Windows.access0` issues on directory structures, PySpark reads single Parquet files directly (`orders.parquet`, `order_items.parquet`, etc.) and performs partition distribution in-memory with `.repartition(8, "location_id")`.

---

## Quick Start / Evaluator Execution Commands

All scripts are executed from the project root using the active virtual environment:

### 1. Run Master End-to-End Pipeline (Recommended for Demo)
Executes all 7 stages (SparkSession, Schemas, Ingestion/DQ, Joins, Spark SQL Suite, MLlib Benchmark, and Output Verification) in ~36 seconds:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/run_pipeline.py
```

### 2. Run Instant Health & Verification Audit (<15 seconds)
Verifies SparkSession creation, explicit schema contracts (11 tables), Parquet loading, and multi-way joins:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/verify_pyspark.py
```

### 3. Run Production Spark SQL Query Suite
Registers all 11 cleaned tables as TempViews and runs all 8 SQL business queries:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/sql_runner.py
```

### 4. Run PySpark MLlib Multi-Algorithm Benchmark
Evaluates Logistic Regression, Decision Tree, Random Forest, and Gradient-Boosted Trees on 50,000 customer vectors and outputs model comparison evidence:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/benchmark_mllib.py
```

### 5. Run Customer Segmentation & RFM Quintiles (SRS Steps 15 & 16)
Calculates all 10 SRS behavioral factors and quintile RFM cells (111 to 555) for 50,000 customers:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/customer_segmentation.py
```

### 6. Run Menu Performance Classification (SRS Steps 9, 10, 11)
Analyzes 10-dimensional menu profitability, classifies 150 items into 4 BCG categories, and flags 10 tricky operational cases:
```powershell
.\dineiq_env\Scripts\python.exe pyspark/menu_classification.py
```

---

## SRS Traceability Matrix

| SRS Step | Description | Implementation Module | Output / Evidence |
|---|---|---|---|
| **Step 3** | PySpark Ingestion Engine | `pyspark/ingestion/load_csv.py` | 1,195,503 rows ingested |
| **Step 4** | Explicit Schemas Contract | `pyspark/schemas.py` | 11 StructType table schemas |
| **Step 5** | Data Quality & Remediation | `pyspark/cleaning/cleaning_rules.py` | 24 quarantine partition files |
| **Step 6** | Relational Joins & Master Cube | `pyspark/joins/join_pipeline.py` | 904,586 fact rows joined |
| **Step 6 & 13** | Spark SQL Query Engine | `pyspark/sql_runner.py`, `pyspark/queries/*.sql` | 8 production analytical queries |
| **Step 7 & 8** | Feature Engineering | `pyspark/features/` | 22 features in Parquet feature store |
| **Step 9** | Menu Profitability (10 Dims) | `pyspark/menu_classification.py` | `reports/menu_classification/` |
| **Step 10** | 4-Category Menu Classification | `pyspark/menu_classification.py` | Profit/Volume Drivers, Hidden, Low |
| **Step 11** | 10 Tricky Menu Scenarios | `pyspark/menu_classification.py` | Diagnostic flags on all 150 items |
| **Step 12** | PySpark MLlib Benchmark | `pyspark/benchmark_mllib.py` | `spark_model_evidence.json` |
| **Step 15** | Customer Segmentation (10 Factors) | `pyspark/customer_segmentation.py` | 6 SRS segments classified |
| **Step 16** | RFM Statistical Quintiles | `pyspark/customer_segmentation.py` | R/F/M scores (1–5), RFM cells |
