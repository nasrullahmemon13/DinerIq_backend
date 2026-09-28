"""
DineIQ Analytics - Enterprise FastAPI Backend
Serving REST APIs for DineIQ Executive & Analytical Dashboards (SRS Steps 42-47).
"""

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

class SPAFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code == 404 and not path.startswith('api/') and '.' not in path.rsplit('/', 1)[-1]:
                return await super().get_response('index.html', scope)
            raise

from backend.routers.executive import router as executive_router
from backend.routers.menu_intelligence import router as menu_intelligence_router
from backend.routers.customer_intelligence import router as customer_intelligence_router
from backend.routers.wastage import router as wastage_router
from backend.routers.search_filter import router as search_filter_router
from backend.routers.reports import router as reports_router
from backend.routers.exports import router as exports_router
from backend.routers.data_pipeline import router as data_pipeline_router
from backend.routers.database_status import router as database_status_router
from backend.routers.restaurants import router as restaurants_router
from backend.routers.locations_channels import router as locations_channels_router
from backend.routers.orders import router as orders_router
from backend.routers.promotions import router as promotions_router
from backend.routers.ratings import router as ratings_router
from backend.routers.inventory import router as inventory_router
from backend.routers.market_basket import router as market_basket_router
from backend.routers.peak_periods import router as peak_periods_router
from backend.routers.forecasting import router as forecasting_router
from backend.routers.pricing_intelligence import router as pricing_intelligence_router
from backend.routers.anomalies_churn import router as anomalies_churn_router
from backend.routers.recommendations import router as recommendations_router
from backend.routers.what_if import router as what_if_router
from backend.routers.data_governance import router as data_governance_router
from backend.routers.admin_management import router as admin_management_router
from src.routes import router as crud_router, get_current_user
from src.error_handlers import register_error_handlers
from backend.routers.verified_models import router as verified_models_router

app = FastAPI(
    title="DineIQ Analytics Platform API",
    description="Backend API powering the DineIQ Analytics Dashboard Suite (SRS Steps 42-50, 61-66)",
    version="1.0.0"
)

register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(executive_router, dependencies=[Depends(get_current_user)])
app.include_router(menu_intelligence_router, dependencies=[Depends(get_current_user)])
app.include_router(customer_intelligence_router, dependencies=[Depends(get_current_user)])
app.include_router(wastage_router, dependencies=[Depends(get_current_user)])
app.include_router(search_filter_router, dependencies=[Depends(get_current_user)])
app.include_router(reports_router, dependencies=[Depends(get_current_user)])
app.include_router(exports_router, dependencies=[Depends(get_current_user)])
app.include_router(data_pipeline_router, dependencies=[Depends(get_current_user)])
app.include_router(database_status_router, dependencies=[Depends(get_current_user)])
app.include_router(restaurants_router, dependencies=[Depends(get_current_user)])
app.include_router(locations_channels_router, dependencies=[Depends(get_current_user)])
app.include_router(orders_router, dependencies=[Depends(get_current_user)])
app.include_router(promotions_router, dependencies=[Depends(get_current_user)])
app.include_router(ratings_router, dependencies=[Depends(get_current_user)])
app.include_router(inventory_router, dependencies=[Depends(get_current_user)])
app.include_router(market_basket_router, dependencies=[Depends(get_current_user)])
app.include_router(peak_periods_router, dependencies=[Depends(get_current_user)])
app.include_router(forecasting_router, dependencies=[Depends(get_current_user)])
app.include_router(pricing_intelligence_router, dependencies=[Depends(get_current_user)])
app.include_router(anomalies_churn_router, dependencies=[Depends(get_current_user)])
app.include_router(recommendations_router, dependencies=[Depends(get_current_user)])
app.include_router(what_if_router, dependencies=[Depends(get_current_user)])
app.include_router(data_governance_router, dependencies=[Depends(get_current_user)])
app.include_router(admin_management_router, dependencies=[Depends(get_current_user)])
app.include_router(crud_router)
app.include_router(verified_models_router)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "DineIQ Analytics API",
        "active_dashboards": [
            "Executive Dashboard (Step 42)",
            "Menu Intelligence Dashboard (Step 43)",
            "Customer Intelligence Dashboard (Step 44)",
            "Wastage Dashboard (Step 45)"
        ]
    }


FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(FRONTEND_DIST):
    app.mount("/", SPAFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
