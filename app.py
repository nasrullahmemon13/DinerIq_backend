"""ASGI entry point for hosting providers."""
from backend.main import app
from fastapi.middleware.cors import CORSMiddleware


@app.get("/", include_in_schema=False)
def api_index():
    return {
        "service": "DineIQ Analytics API",
        "docs": "/docs",
        "health": "/api/health",
    }


# Wrap the entire app so unhandled-error responses also include CORS headers.
app = CORSMiddleware(
    app,
    allow_origins=["https://diner-iq-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
