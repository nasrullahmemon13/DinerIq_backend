"""
DineIQ Analytics - Backend Routers
"""
from backend.routers.executive import router as executive_router
from backend.routers.menu_intelligence import router as menu_intelligence_router
from backend.routers.customer_intelligence import router as customer_intelligence_router
from backend.routers.wastage import router as wastage_router
from backend.routers.search_filter import router as search_filter_router
from backend.routers.reports import router as reports_router
from backend.routers.exports import router as exports_router

__all__ = [
    "executive_router",
    "menu_intelligence_router",
    "customer_intelligence_router",
    "wastage_router",
    "search_filter_router",
    "reports_router",
    "exports_router"
]
