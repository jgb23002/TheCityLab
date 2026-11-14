"""Run the merged tools as a typed, question-specific workflow for the web UI.

The web workflow uses the committed city cohorts, retains the full Building
records for regression controls, and writes briefs directly from tool outputs.
The Deep Agents CLI remains available for conversational tool planning.
"""

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from backend import config, samples
from backend.cache import CacheMiss
from backend.schemas import (
    AgentEvent, AskRequest, AskResponse, Building, Measurement, OutreachDraft,
    SolarResult, StatsResult, StudyResult,
)
from backend.tools import costar_data, mock, outreach, solar, stats, vexcel, vision

CITIES = {14: "Houston", 20: "Los Angeles"}


class StudyUnavailable(RuntimeError):
    """A user-actionable study problem with a credential-free public message."""


def preview_buildings(
    market_id: int, property_type: int = 11, limit: int = 50, use_mock_tools: bool = False,
) -> list[Building]:
    """Return the selected cohort without imagery, model calls or statistical analysis."""
    if market_id not in CITIES:
        raise ValueError("Choose Houston or Los Angeles.")
    request = AskRequest(question="Preview selected cohort", market_id=market_id,
                         property_type=property_type, limit=limit, use_mock_tools=use_mock_tools)
    return _cohort(request, market_id)


@lru_cache(maxsize=2)
def _kit_properties(data_dir: Path) -> dict[int, Building]:
    """Enrich committed cohort ids with the existing loader once per local kit."""
    return {building.id: building for building in costar_data.load_properties()}


def _cohort(request: AskRequest, market: int) -> list[Building]:
    """Use synthetic tools explicitly, otherwise the merged PR's real city samples."""
    property_type = request.property_type if request.property_type is not None else 11
    if request.use_mock_tools:
        return mock.find_buildings(market, property_type, request.limit)
    city = "houston" if market == 14 else "los_angeles"
    try:
        buildings = samples.sample_buildings(city, property_type, n=request.limit)
        if (config.COSTAR_DATA_DIR / "commercial-property.jsonl").exists():
            properties = _kit_properties(config.COSTAR_DATA_DIR)
            buildings = [properties.get(building.id, building) for building in buildings]
    except (OSError, ValueError, KeyError):
        raise StudyUnavailable("City sample data is unavailable. Restore data/samples/ or rebuild the city samples.") from None
    return buildings


def _cached_measurement(building: Building) -> Measurement | None:
    """Reuse the vision cache format already written by scripts/accuracy_check.py."""
    path = config.CACHE_DIR / "vision" / f"m-{building.id}.json"
    try:
        measurement = Measurement.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
    observations = measurement.observations or {}
    if measurement.property_id != building.id or observations.get("error") or observations.get("is_mock"):
        return None
    return measurement


def _measure(building: Building, use_mock: bool) -> tuple[Measurement, bool]:
    """Measure one property through the merged tools, respecting cache-only mode."""
    if use_mock:
        image = mock.get_aerial(building.latitude, building.longitude, building.id)
        parcel = mock.get_property_info(building.latitude, building.longitude)
        result = mock.measure_building(image.image_path, building.id)
    else:
        cached = _cached_measurement(building)
        if cached is not None:
            return cached, True
        if config.USE_CACHE_ONLY:
            raise CacheMiss("No cached vision measurement")
        if (config.LLM_PROVIDER or "").strip().lower() not in {"bedrock", "google", "anthropic"} or not (config.VISION_MODEL or "").strip():
            raise StudyUnavailable("Configure LLM_PROVIDER and VISION_MODEL for fresh aerial analysis, or use cached results or mock demo.")
        image = vexcel.get_aerial(building.latitude, building.longitude, building.id)
        parcel = vexcel.get_property_info(building.latitude, building.longitude)
        if not image.image_path:
            raise StudyUnavailable("The aerial tool returned no saved image.")
        result = vision.measure_building(image.image_path, building.id)
        if (result.observations or {}).get("error"):
            raise StudyUnavailable("Vision analysis was unavailable. Check the vision model and provider settings, then retry.")
    if result.property_id != building.id:
        raise StudyUnavailable("The measurement does not match the sampled building.")
    result = result.model_copy(update={"observations": {
        **(result.observations or {}), "vexcel_property_info_id": parcel.id,
    }})
    if not use_mock and not (result.observations or {}).get("error"):
        path = config.CACHE_DIR / "vision" / f"m-{building.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(), encoding="utf-8")
    return result, False


def _research_brief(question: str, result: StatsResult, is_mock: bool) -> str:
    """Describe only returned statistics, with explicit missing results and provenance."""
    label = "**MOCK / SYNTHETIC DATA**\n\n" if is_mock else ""
    key = result.predictor
    coefficient = (result.coefficients or {}).get(key)
    interval = (result.confidence_intervals or {}).get(key)
    p_value = (result.p_values or {}).get(key)
    finding = result.interpretation or "The tools could not estimate an association for this sample."
    numbers = f"Analyzed buildings: {result.sample_size if result.sample_size is not None else 'unavailable'}. "
    numbers += f"Effect: {coefficient if coefficient is not None else 'unavailable'}. "
    numbers += f"95% confidence interval: {list(interval) if interval is not None else 'unavailable'}. "
    numbers += f"p-value: {p_value if p_value is not None else 'unavailable'}."
    controls = ", ".join(result.controls or []) or "not estimable"
    return (f"{label}**Question:** {question}\n\n{finding}\n\n{numbers}\n\n"
            f"Outcome: {result.outcome or 'unavailable'}. Predictor: {key or 'unavailable'}. "
            f"Controls: {controls}. Source: run_study tool output; observational association only.")


def _draft(building: Building, estimate: SolarResult, use_mock: bool) -> OutreachDraft:
    """Reuse the merged outreach tool, using its local template in cache-only mode."""
    if use_mock:
        return mock.draft_outreach(building, estimate)
    if config.USE_CACHE_ONLY:
        subject, body = outreach.template_draft(building, estimate)
        return OutreachDraft(id=f"outreach-{building.id}", property_id=building.id,
                             subject=subject, body_markdown=body,
                             caveats=["Template draft for review; no model call or message was sent."])
    return outreach.draft_outreach(building, estimate)


def run_study(request: AskRequest) -> AskResponse:
    """Run research or retrofit tools and return the shared study/events API contract."""
    question = request.question.strip()
    if not question:
        raise ValueError("Enter a nonempty question.")
    market = request.market_id
    if market is None:
        market = 20 if re.search(r"los angeles|\bla\b", question, re.I) else 14
    if market not in CITIES:
        raise ValueError("Choose Houston (14) or Los Angeles (20).")
    mode = request.mode
    if re.search(r"solar|payback", question, re.I) and not re.search(r"rent|vacan", question, re.I):
        mode = "retrofit"
    run_id = f"study-{uuid4().hex}"
    events: list[AgentEvent] = []

    def event(kind: str, message: str, **payload: int | bool) -> None:
        """Record a completed workflow step with its real timestamp and source mode."""
        events.append(AgentEvent(
            id=f"{run_id}-{len(events) + 1}", run_id=run_id, event_type=kind,
            message=message, timestamp=datetime.now(timezone.utc).isoformat(),
            payload={"is_mock": request.use_mock_tools, **payload},
        ))

    event("plan", f"Preparing {mode} study in {CITIES[market]}.")
    buildings = _cohort(request, market)
    if not buildings:
        raise StudyUnavailable("No buildings match this city's committed sample and property type.")
    buildings = [building for building in buildings
                 if building.latitude is not None and building.longitude is not None]
    if not buildings:
        raise StudyUnavailable("No sampled buildings have latitude and longitude.")
    event("sample", f"Selected {len(buildings)} {CITIES[market]} buildings.", count=len(buildings))
    caveats = ["Mock: every measurement and statistic is synthetic."] if request.use_mock_tools else [
        "Building attributes come from the committed CoStar cohort, enriched from the local kit when available.",
        "Cached aerial measurements may reflect an earlier capture; an image square may not cover a whole community.",
    ]
    if len(buildings) < request.limit:
        caveats.append(f"Requested {request.limit} buildings; the available cohort contains {len(buildings)}.")
    measurements = []
    usable = 0
    for index, building in enumerate(buildings, 1):
        try:
            measurement, cached = _measure(building, request.use_mock_tools)
        except CacheMiss:
            measurement = Measurement(id=f"m-{building.id}", property_id=building.id,
                                      observations={"error": "No cached vision measurement"})
            cached = False
        except StudyUnavailable:
            raise
        except Exception:
            raise StudyUnavailable("Aerial analysis failed. Check the Vexcel/vision settings and connection, or choose mock demo.") from None
        if not (measurement.observations or {}).get("error") and any(value is not None for value in (
            measurement.green_cover_percent, measurement.paved_surface_percent, measurement.roof_area_sq_ft,
            measurement.parking_space_count,
        )):
            usable += 1
        measurements.append(measurement)
        event("measure", f"Checked aerial evidence for {index}/{len(buildings)} buildings.",
              property_id=building.id, cached=cached)
    event("inspect", f"Usable vision measurements: {usable}/{len(buildings)}.", count=usable)
    if not usable:
        raise StudyUnavailable("No usable aerial measurements. Populate the vision cache or check Vexcel/vision settings; mock demo is also available.")
    missing = len(buildings) - usable
    if missing:
        caveats.append(f"{missing} buildings lack usable aerial measurements; missing values remain unknown.")

    result: StatsResult | None = None
    estimates: list[SolarResult] | None = None
    drafts: list[OutreachDraft] | None = None
    if mode == "research":
        tool = mock.run_study if request.use_mock_tools else stats.run_study
        result = tool(question, buildings, measurements)
        caveats.extend(result.caveats or [])
        event("stats", f"Statistical analysis completed; analyzed sample: {result.sample_size or 0}.")
        brief = _research_brief(question, result, request.use_mock_tools)
    else:
        tools = mock if request.use_mock_tools else solar
        estimates = tools.rank_buildings([
            tools.estimate_solar(building, measurement)
            for building, measurement in zip(buildings, measurements)
        ])
        shortlist = [estimate for estimate in estimates if estimate.payback_years is not None][:10]
        event("solar", f"Ranked {len(shortlist)} shortlisted solar candidates by payback.", count=len(shortlist))
        for estimate in estimates:
            caveats.extend(estimate.caveats or [])
        label = "**MOCK / SYNTHETIC DATA**\n\n" if request.use_mock_tools else ""
        brief = f"{label}**Question:** {question}\n\nScreened {len(buildings)} {CITIES[market]} buildings using measured roof areas. "
        if shortlist:
            best = shortlist[0]
            building = next(building for building in buildings if building.id == best.property_id)
            brief += (f"The shortest returned payback is {best.payback_years} years for {building.name or building.id}, "
                      f"with {best.capacity_kw} kW capacity and {best.annual_energy_kwh} kWh annual energy. "
                      "Source: estimate_solar and rank_buildings tool outputs.")
            drafts = [_draft(building, best, request.use_mock_tools)]
            caveats.extend(drafts[0].caveats or [])
            event("outreach", "Prepared an owner-email draft for the top candidate; no email was sent.")
        else:
            brief += "No candidate has a known payback; no ranking recommendation or owner email was produced."
        brief += "\n\nThese are screening estimates. Review roof condition, tariffs and financial assumptions before making a decision."
    event("write", "Built the brief, evidence table and caveats from tool outputs.")
    return AskResponse(study=StudyResult(
        id=run_id, question=question, buildings=buildings, measurements=measurements,
        stats=result, brief_markdown=brief, caveats=list(dict.fromkeys(caveats)),
        solar_results=estimates, outreach_drafts=drafts, is_mock=request.use_mock_tools,
    ), events=events)
