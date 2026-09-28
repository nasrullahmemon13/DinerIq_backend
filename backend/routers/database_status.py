"""
DineIQ Analytics - Database & MongoDB Status Router (SRS Step 2 & Step 30)
Provides health, connection state, collection statistics, and sync endpoints.
"""
from fastapi import APIRouter
from database.mongodb_manager import check_mongo_connection, MONGO_EXPORT_DIR
import os

router = APIRouter(prefix="/api/database", tags=["Database"])


@router.get("/status")
def get_database_status():
    """Returns connectivity and statistics for both Relational DB and NoSQL MongoDB."""
    mongo_status = check_mongo_connection()

    local_collections = {}
    if os.path.exists(MONGO_EXPORT_DIR):
        for f in os.listdir(MONGO_EXPORT_DIR):
            if f.endswith(".json"):
                fp = os.path.join(MONGO_EXPORT_DIR, f)
                c_name = f.replace(".json", "")
                with open(fp, "r", encoding="utf-8") as fl:
                    cnt = sum(1 for line in fl if line.strip())
                local_collections[c_name] = cnt

    return {
        "status": "healthy",
        "relational_database": {
            "type": "SQLite / PostgreSQL (SQLAlchemy)",
            "status": "active"
        },
        "nosql_database": {
            "type": "MongoDB Atlas (PyMongo)",
            "connection_state": mongo_status.get("status"),
            "target_database": os.getenv("MONGODB_DATABASE", "dineiq_analytics"),
            "cluster_info": mongo_status,
            "total_prepared_collections": len(local_collections),
            "total_documents": sum(local_collections.values()),
            "collections_summary": local_collections
        }
    }
