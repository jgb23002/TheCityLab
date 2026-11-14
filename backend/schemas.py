"""CityLab tool and API contracts; source mappings are documented in CONTRACTS.md."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Building(BaseModel):
    """A CoStar building with optional location and model metadata."""

    id: int
    primary_address: dict[str, JsonValue] | None = None
    property_type: int | None = None
    market_id: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    matterport_model_id: str | None = None
    name: str | None = None
    building_size_sq_ft: float | None = None
    year_built: int | None = None
    building_class: int | None = None
    total_num_units: int | None = None
    max_rent: float | None = None
    vacancy_percent: float | None = None
    surface_parking_spaces: int | None = None


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


class Detection(BaseModel):
    """One labeled image detection; bbox is [left, top, right, bottom] in pixels."""

    id: str
    label: str
    property_id: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    bbox: tuple[float, float, float, float] | None = None
    geometry_wkt: str | None = None


class Measurement(VisionResult):
    """Image-derived measurements; existing VisionResult fields retain their names."""

    green_cover_percent: float | None = Field(default=None, ge=0, le=100)
    paved_surface_percent: float | None = Field(default=None, ge=0, le=100)
    parking_space_count: int | None = Field(default=None, ge=0)
    detections: list[Detection] | None = None
    is_corrected: bool = False


class StatsResult(Study):
    """Statistical output extending the existing Study contract."""

    sample_size: int | None = Field(default=None, ge=0)
    outcome: str | None = None
    predictor: str | None = None
    controls: list[str] | None = None
    coefficients: dict[str, float] | None = None
    p_values: dict[str, float] | None = None
    confidence_intervals: dict[str, tuple[float, float]] | None = None
    r_squared: float | None = None
    interpretation: str | None = None
    caveats: list[str] | None = None


class SolarResult(SolarEstimate):
    """Solar output extending SolarEstimate without renaming its fields or units."""

    score: float | None = None
    caveats: list[str] | None = None


class VexcelPropertyInfo(BaseModel):
    """Cached Vexcel parcel attributes; provider keys remain unchanged in raw_response."""

    id: str
    latitude: float | None = None
    longitude: float | None = None
    attributes: dict[str, JsonValue] | None = None
    structures: list[dict[str, JsonValue]] | None = None
    raw_response: dict[str, JsonValue] | None = None


class MatterportModel(BaseModel):
    """Optional Matterport model payload, keyed by its model id."""

    id: str
    metadata: dict[str, JsonValue] | None = None
    raw_response: dict[str, JsonValue] | None = None


class OutreachDraft(BaseModel):
    """A reviewable message draft; this contract never implies sending a message."""

    id: str
    property_id: int | None = None
    subject: str | None = None
    body_markdown: str | None = None
    caveats: list[str] | None = None


class StudyResult(BaseModel):
    """Complete study payload shared by research, correction and mock UI flows."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    buildings: list[Building]
    measurements: list[Measurement]
    stats: StatsResult | None
    brief_markdown: str
    caveats: list[str]
    solar_results: list[SolarResult] | None = None
    is_mock: bool = False


class AskRequest(BaseModel):
    """Reserved JSON request body for POST /api/ask."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    market_id: int | None = None
    property_type: int | None = None
    limit: int = Field(default=50, ge=1, le=500)
    mode: Literal["research", "retrofit"] = "research"


class AskResponse(BaseModel):
    """Reserved POST /api/ask response containing a study and progress events."""

    study: StudyResult
    events: list[AgentEvent] = Field(default_factory=list)


class MeasurementCorrection(BaseModel):
    """Editable measurement fields; omitted fields remain unchanged in future handlers."""

    model_config = ConfigDict(extra="forbid")

    roof_area_sq_ft: float | None = Field(default=None, ge=0)
    green_cover_percent: float | None = Field(default=None, ge=0, le=100)
    paved_surface_percent: float | None = Field(default=None, ge=0, le=100)
    parking_space_count: int | None = Field(default=None, ge=0)
    detections: list[Detection] | None = None
    note: str | None = None


class CorrectRequest(BaseModel):
    """Reserved JSON request body for POST /api/correct."""

    model_config = ConfigDict(extra="forbid")

    study_id: str
    measurement_id: str
    corrections: MeasurementCorrection


class CorrectResponse(AskResponse):
    """Reserved correction response with the updated study and measurement."""

    corrected_measurement: Measurement
