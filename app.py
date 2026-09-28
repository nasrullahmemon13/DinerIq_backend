"""ASGI entry point for hosting providers."""
from backend.main import app


@app.get("/", include_in_schema=False)
def api_index():
    return {
        "service": "DineIQ Analytics API",
        "docs": "/docs",
        "health": "/api/health",
    }
