"""Minimal CityLab health API and static frontend."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR

app = FastAPI(title="CityLab")


@app.get("/api/health")
def health() -> dict[str, bool]:
    """Report whether the CityLab application is running."""
    return {"ok": True}


# Keep API routes above the root static mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# TODO: Add research endpoints when the agent and tools are implemented.
