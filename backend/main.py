"""CityLab study API and static frontend, connected to the merged analysis tools."""

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import JsonValue, ValidationError

from backend.config import FRONTEND_DIR, PROJECT_ROOT
from backend.schemas import AskRequest, AskResponse, Building, StudyResult
from backend.agent.study import StudyUnavailable, preview_buildings, run_study
from backend.agent.agent_api import router as agent_router

SHOWCASE_DIR = PROJECT_ROOT / "data" / "showcase"

app = FastAPI(title="CityLab")


def _load_json(name: str) -> dict[str, JsonValue]:
    """Read a named, internal showcase fixture without exposing local file details."""
    path = SHOWCASE_DIR / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{name} not found") from None
    except (OSError, ValueError):
        raise HTTPException(status_code=500, detail="Unable to read showcase data") from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Invalid showcase data")
    return data


def _load_study() -> StudyResult:
    """Validate the saved mock study shared by the preview and ask endpoints."""
    try:
        study = StudyResult.model_validate(_load_json("mock_study.json"))
    except ValidationError:
        raise HTTPException(status_code=500, detail="Invalid showcase study") from None
    if not study.is_mock:
        raise HTTPException(status_code=500, detail="Showcase study must be marked as mock")
    return study


@app.get("/api/health")
def health() -> dict[str, bool]:
    """Report whether the CityLab application is running."""
    return {"ok": True}


@app.get("/api/questions")
def questions() -> dict[str, JsonValue]:
    """Return the existing three showcase questions without modifying their metadata."""
    return _load_json("questions.json")


@app.get("/api/mock-study", response_model=StudyResult)
def mock_study() -> StudyResult:
    """Return the validated synthetic StudyResult for a frontend preview."""
    return _load_study()


@app.get("/api/buildings", response_model=list[Building])
def buildings(
    market_id: int = 14, property_type: int = 11,
    limit: int = Query(default=50, ge=1, le=500), use_mock_tools: bool = False,
) -> list[Building]:
    """Preview real or synthetic city records without making paid API/model calls."""
    try:
        return preview_buildings(market_id, property_type, limit, use_mock_tools)
    except ValueError:
        raise HTTPException(status_code=422, detail="Choose Houston (14) or Los Angeles (20).") from None
    except StudyUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        raise HTTPException(status_code=503, detail="City sample data could not be loaded.") from None


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run the requested research or retrofit workflow using real or explicit mock tools."""
    try:
        return run_study(request)
    except ValueError:
        raise HTTPException(status_code=422, detail="Enter a question and choose Houston (14) or Los Angeles (20).") from None
    except StudyUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        raise HTTPException(status_code=503, detail="The study could not complete. Check local data, tool settings and cached results.") from None


app.include_router(agent_router)

# Keep API routes above the root static mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
