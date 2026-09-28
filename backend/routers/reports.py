"""
DineIQ Analytics - Downloadable Reports Router (SRS Step 49)
"""

from fastapi import APIRouter, HTTPException, Response
from src.report_generator import ReportGenerator

router = APIRouter(prefix="/api/v1/reports", tags=["Downloadable Reports"])
generator = ReportGenerator()


@router.get("")
def list_downloadable_reports():
    """Returns catalog of all 12 SRS Step 49 downloadable reports."""
    return {
        "srs_step": 49,
        "total_reports": 12,
        "reports": generator.list_reports()
    }


@router.get("/{report_key}/download")
def download_report(report_key: str):
    """
    Downloads the selected analytical report as a formatted Markdown/Text document.
    """
    content = generator.get_report_content(report_key)
    if not content:
        raise HTTPException(status_code=404, detail=f"Report '{report_key}' not found.")

    filename = f"dineiq_{report_key}_report.md"
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
