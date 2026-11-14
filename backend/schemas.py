"""CityLab output models; TODO: confirm source mappings before implementation."""

from pydantic import BaseModel, JsonValue


class Building(BaseModel):
    """A CoStar building with optional location and model metadata."""

    id: int
    primary_address: dict[str, JsonValue] | None = None
    property_type: int | None = None
    market_id: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    matterport_model_id: str | None = None


class AerialImage(BaseModel):
    """Metadata for cached aerial imagery; image bytes stay on disk."""

    id: str
    property_id: int | None = None
    image_path: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    captured_at: str | None = None
    metadata: dict[str, JsonValue] | None = None


class VisionResult(BaseModel):
    """Optional measurements and observations from an aerial image."""

    id: str
    property_id: int | None = None
    aerial_image_id: str | None = None
    roof_area_sq_ft: float | None = None
    confidence: float | None = None
    observations: dict[str, JsonValue] | None = None


class Study(BaseModel):
    """A research question and its eventual statistical findings."""

    id: str
    question: str | None = None
    property_ids: list[int] | None = None
    method: str | None = None
    findings: dict[str, JsonValue] | None = None


class SolarEstimate(BaseModel):
    """Optional rooftop solar estimates with explicit units."""

    id: str
    property_id: int | None = None
    usable_roof_area_sq_ft: float | None = None
    capacity_kw: float | None = None
    annual_energy_kwh: float | None = None
    rank: int | None = None
    assumptions: dict[str, JsonValue] | None = None


class AgentEvent(BaseModel):
    """An optional progress message and JSON payload from an agent run."""

    id: str
    run_id: str | None = None
    event_type: str | None = None
    message: str | None = None
    timestamp: str | None = None
    payload: dict[str, JsonValue] | None = None


class RunResult(BaseModel):
    """Optional research outputs associated with one agent run."""

    id: str
    question: str | None = None
    buildings: list[Building] | None = None
    studies: list[Study] | None = None
    solar_estimates: list[SolarEstimate] | None = None
    events: list[AgentEvent] | None = None
    summary: str | None = None
