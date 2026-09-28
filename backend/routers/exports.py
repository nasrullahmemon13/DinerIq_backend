"""
DineIQ Analytics - Data Export Router (SRS Step 50)
Enforces permission checks and exports datasets in CSV or Excel format.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Query
from src.routes import require_roles
from src.data_export_engine import DataExportEngine, PermissionDeniedError

router = APIRouter(prefix="/api/v1/export", tags=["Data Export"])


@router.get("/datasets")
def list_exportable_datasets(user=Depends(require_roles(["admin", "analyst", "regional_manager", "manager"]))):
    """Returns available analytical datasets for CSV/Excel export."""
    return {
        "srs_step": 50,
        "datasets": DataExportEngine.list_exportable_datasets()
    }


@router.get("/{dataset_key}")
def export_dataset(
    dataset_key: str,
    format: str = Query("csv", pattern="^(csv|excel|xlsx)$", description="Format: csv or excel"),
    location_id: Optional[str] = None,
    category_id: Optional[str] = None,
    item_id: Optional[str] = None,
    user=Depends(require_roles(["admin", "analyst", "regional_manager", "manager"]))
):
    """
    Exports the specified analytical dataset in CSV or Excel format.
    Checks permissions: requires can_export_data permission (roles: admin, analyst, executive, manager).
    """
    try:
        content_bytes, media_type, filename = DataExportEngine.export_dataset(
            dataset_key=dataset_key,
            export_format=format,
            user_role=user["role_id"],
            filters={"location_id": location_id, "category_id": category_id, "item_id": item_id}
        )
        return Response(
            content=content_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except PermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
