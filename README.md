# DineIQ backend

FastAPI API and supporting analytics modules extracted from the current DineIQ workspace.

Install `requirements.txt`, configure variables from `.env.example`, initialize the PostgreSQL schema/data, and run `uvicorn app:app --host 0.0.0.0 --port 8000`.

## Deployment status

The root `app.py` exports the ASGI application. This repository is not yet verified for full Vercel operation. Pipeline/report features write to local storage, Spark requires Java, and authentication sessions are currently held in process memory. These require persistent external storage/session handling and an appropriate Spark runtime for reliable serverless deployment. Analytics dependencies may also exceed hosting bundle limits.

Real environment files and local SQLite databases are intentionally excluded.

Runtime datasets, sample records, generated reports and trained model artifacts are excluded from this source repository. Provision required analytics data separately before running data-dependent endpoints.
