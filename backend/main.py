"""CityLab study API and static frontend, connected to the merged analysis tools."""

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import JsonValue, ValidationError

from backend.config import CACHE_DIR, FRONTEND_DIR, MP_SDK_KEY, PROJECT_ROOT
from backend.schemas import AskRequest, AskResponse, Building, Measurement, StudyResult
from backend.agent.study import StudyUnavailable, preview_buildings, run_study
from backend.agent.agent_api import router as agent_router
from backend.agent.solar_city import router as solar_city_router
from backend.samples import sample_buildings
from backend.tools import costar_data, geometry, solar

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


@app.get("/api/config")
def frontend_config() -> dict[str, JsonValue]:
    """Public, non-secret settings the frontend needs; never a credential pair.

    The Matterport Showcase SDK key is a publishable, domain-restricted key (used to
    embed a model viewer client-side), unlike MP_TOKEN_ID/MP_TOKEN_SECRET, which stay
    server-only and are never returned here.
    """
    return {"matterport_sdk_key": MP_SDK_KEY}


@app.get("/api/questions")
def questions() -> dict[str, JsonValue]:
    """Return the existing three showcase questions without modifying their metadata."""
    return _load_json("questions.json")


@app.get("/api/mock-study", response_model=StudyResult)
def mock_study() -> StudyResult:
    """Return the validated synthetic StudyResult for a frontend preview."""
    return _load_study()


def _has_measurement_with_boxes(building_id: int) -> bool:
    """True when a cached measurement and its boxed aerial photo both exist for this id."""
    measurement_path = CACHE_DIR / "vision" / f"m-{building_id}.json"
    boxes_path = CACHE_DIR / "vexcel" / "detect" / f"property_{building_id}_boxes.jpg"
    return measurement_path.is_file() and boxes_path.is_file()


def _has_roof_view(building_id: int) -> bool:
    return (CACHE_DIR / "roof_view" / f"property_{building_id}_report.jpg").is_file()


@app.get("/api/showcase-strip")
def showcase_strip() -> dict[str, JsonValue]:
    """One real, fully-measured building for the landing-page walkthrough.

    Picks the first sampled Houston building that already has a cached measurement,
    a boxed aerial photo and a roof-sun report (all produced by scripts/measure_sample.py
    and scripts/roof_solar_view.py, never computed on request), so the page never
    triggers a live Vexcel, vision or NREL call. Returns 404 if none qualifies yet.
    """
    for sample in sample_buildings("houston", 11, n=50):
        if not (_has_measurement_with_boxes(sample.id) and _has_roof_view(sample.id)):
            continue
        # The committed sample CSV is a lean join key; the full CoStar record (name,
        # address, size) lives in the local kit, same as backend.agent.study's cohort.
        building = costar_data.get_property(sample.id) or sample
        measurement = Measurement.model_validate_json(
            (CACHE_DIR / "vision" / f"m-{building.id}.json").read_text(encoding="utf-8"))
        roof = geometry.roof_footprint_sq_ft(building)
        estimate = solar.estimate_solar(building, measurement)
        return {
            "property_id": building.id,
            "name": building.name,
            "address": (building.primary_address or {}).get("city"),
            "building_size_sq_ft": building.building_size_sq_ft,
            "year_built": building.year_built,
            "roof_area_sq_ft": roof,
            "payback_years": estimate.payback_years,
            "capacity_kw": estimate.capacity_kw,
        }
    raise HTTPException(status_code=404, detail="No fully-measured showcase building is cached yet.")


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
app.include_router(solar_city_router)



@app.get("/classic", include_in_schema=False)
def classic_page() -> FileResponse:
    """The original question-box app (its assets are absolute paths served by the root mount)."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Keep API routes above the root static mount. The solar page is the home page: the root
# mount looks in frontend/solar/ first, then frontend/ (app.js, style.css, and /solar/ itself).
frontend_files = StaticFiles(directory=FRONTEND_DIR, html=True)
frontend_files.all_directories = [FRONTEND_DIR / "solar", FRONTEND_DIR]
app.mount("/", frontend_files, name="frontend")
