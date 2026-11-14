"""Deterministic, synthetic implementations of every documented CityLab tool.

These functions use only the committed mock fixture and fixed demonstration
values. They never load credentials, inspect real imagery, call an API, fit a
regression, or send a message. Return schema objects, not serialized strings.
"""

from pathlib import Path

from backend.schemas import (
    AerialImage,
    Building,
    MatterportModel,
    Measurement,
    OutreachDraft,
    SolarResult,
    StatsResult,
    StudyResult,
    VexcelPropertyInfo,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "data" / "showcase" / "mock_study.json"
_LA_COORDINATES = (
    (34.0522, -118.2437), (34.0610, -118.2830), (34.0760, -118.3040),
    (34.0440, -118.2710), (34.0850, -118.3310), (34.0280, -118.2930),
    (34.1030, -118.3210), (34.0700, -118.3530), (34.0160, -118.3090),
    (34.0950, -118.2910), (34.0360, -118.2520), (34.0780, -118.2600),
)


def _study() -> StudyResult:
    """Load a fresh validated fixture so callers cannot mutate shared mock state."""
    return StudyResult.model_validate_json(_FIXTURE.read_text(encoding="utf-8"))


def _required_building(property_id: int) -> Building:
    """Find a known synthetic property or fail instead of fabricating an identity."""
    building = get_property(property_id)
    if building is None:
        raise ValueError("Property id is not in the synthetic mock cohort.")
    return building


def load_properties() -> list[Building]:
    """Return twelve fictional Houston and twelve fictional Los Angeles communities."""
    houston = _study().buildings
    los_angeles = [
        building.model_copy(deep=True, update={
            "id": 990101 + index, "market_id": 20,
            "latitude": lat, "longitude": lon,
            "name": f"Mock Los Angeles Community {index + 1:02d}",
            "primary_address": {"city": "Los Angeles", "state": "CA", "researchMarketId": 20},
        })
        for index, (building, (lat, lon)) in enumerate(zip(houston, _LA_COORDINATES))
    ]
    return [*houston, *los_angeles]


def get_property(id: int) -> Building | None:
    """Return a synthetic Building by its id, or None for an unknown id."""
    return next((building for building in load_properties() if building.id == id), None)


def find_buildings(market_id: int, property_type: int, limit: int = 20) -> list[Building]:
    """Return up to limit synthetic buildings matching the documented market/type."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    return [building for building in load_properties()
            if building.market_id == market_id and building.property_type == property_type][:limit]


def get_aerial(lat: float, lon: float, property_id: int) -> AerialImage:
    """Return a synthetic image reference; no real file or imagery request is made."""
    _required_building(property_id)
    return AerialImage(
        id=f"mock-aerial-{property_id}", property_id=property_id,
        latitude=lat, longitude=lon, image_path=f"mock://aerial/{property_id}.jpg",
        metadata={"is_mock": True, "image_available": False, "source": "fixed synthetic reference"},
    )


def get_property_info(lat: float, lon: float) -> VexcelPropertyInfo:
    """Return fixed fake parcel/structure evidence for a known synthetic location."""
    building = next((b for b in load_properties()
                     if abs(b.latitude - lat) < 0.000001 and abs(b.longitude - lon) < 0.000001), None)
    if building is None:
        raise ValueError("Coordinates are not in the synthetic mock cohort.")
    attributes = {"is_mock": True, "structure.count": 1, "parcel-tree-cover.percent": 22.0,
                  "pool.detected": False, "pool.total-area": 0}
    structures = [{"primary-structure": True, "footprint.area": 18000.0,
                   "roof-characteristics.condition": 3.0}]
    return VexcelPropertyInfo(
        id=f"mock-parcel-{building.id}", latitude=lat, longitude=lon,
        attributes=attributes, structures=structures,
        raw_response={"is_mock": True, "source": "fixed demonstration, not a Vexcel response"},
    )


def measure_building(image_path: str, property_id: int) -> Measurement:
    """Return fixed fixture measurements without reading image_path or calling a model."""
    building = _required_building(property_id)
    index = building.id - (990001 if building.market_id == 14 else 990101)
    measurement = _study().measurements[index]
    return measurement.model_copy(deep=True, update={
        "id": f"mock-measurement-{property_id}", "property_id": property_id,
        "aerial_image_id": f"mock-aerial-{property_id}",
        "detections": [d.model_copy(deep=True, update={"id": f"mock-detection-{property_id}-{i}",
                                                    "property_id": property_id})
                       for i, d in enumerate(measurement.detections or [])],
        "observations": {**(measurement.observations or {}), "is_mock": True,
                         "image_reference": image_path, "image_read": False},
    })


def run_study(
    question: str, buildings: list[Building], measurements: list[Measurement] | None = None,
) -> StatsResult:
    """Return fixed illustrative statistics, explicitly not an actual regression fit."""
    if not buildings:
        raise ValueError("Provide at least one sampled building.")
    result = _study().stats.model_copy(deep=True, update={
        "id": "mock-stats", "question": question, "property_ids": [b.id for b in buildings],
        "sample_size": len(buildings),
        "findings": {"is_mock": True, "regression_fitted": False,
                     "measurement_count": len(measurements or [])},
        "caveats": ["MOCK: fixed synthetic numbers; no regression was fitted.",
                    "No causal inference; imagery and rent comparability are unverified."],
    })
    if "vacan" in question.lower():
        result = result.model_copy(update={
            "outcome": "vacancy_percent", "predictor": "paved_surface_percent",
            "coefficients": {"paved_surface_percent": 0.14},
            "p_values": {"paved_surface_percent": 0.13},
            "confidence_intervals": {"paved_surface_percent": (-0.06, 0.34)},
            "r_squared": 0.31,
            "interpretation": "MOCK: illustrative vacancy percentage-point association only.",
        })
    return result


def estimate_solar(building: Building, vision_result: Measurement | None = None) -> SolarResult:
    """Return fixed synthetic solar capacity, energy and simple payback in years."""
    if vision_result is not None and vision_result.property_id != building.id:
        raise ValueError("Measurement property_id must match the sampled building.")
    slot = building.id % 3
    return SolarResult(
        id=f"mock-solar-{building.id}", property_id=building.id,
        usable_roof_area_sq_ft=(3500.0, 4500.0, 5500.0)[slot],
        capacity_kw=(48.0, 60.0, 72.0)[slot], annual_energy_kwh=(72000.0, 90000.0, 108000.0)[slot],
        payback_years=(6.4, 7.1, 8.2)[slot], score=(90.0, 80.0, 70.0)[slot],
        assumptions={"is_mock": True, "source": "fixed demo assumptions, not a solar calculation"},
        caveats=["MOCK: no irradiance, roof engineering, tariff or financial analysis was performed."],
    )


def rank_buildings(estimates: list[SolarResult]) -> list[SolarResult]:
    """Rank the supplied mock estimates by known payback, with missing paybacks last."""
    ordered = sorted(estimates, key=lambda e: (e.payback_years is None,
                     e.payback_years if e.payback_years is not None else float("inf"), e.property_id or 0))
    return [estimate.model_copy(deep=True, update={"rank": index}) for index, estimate in enumerate(ordered, 1)]


def draft_outreach(building: Building, solar_estimate: SolarResult) -> OutreachDraft:
    """Return a synthetic owner-email draft for review without sending anything."""
    if solar_estimate.property_id != building.id:
        raise ValueError("Solar estimate property_id must match the sampled building.")
    return OutreachDraft(
        id=f"mock-draft-{building.id}", property_id=building.id,
        subject=f"MOCK DRAFT: rooftop solar screening for {building.name or building.id}",
        body_markdown=f"Hello property owner,\n\nThis synthetic demonstration for {building.name or building.id} "
        f"shows {solar_estimate.capacity_kw} kW of capacity and {solar_estimate.payback_years} years of "
        "simple payback. These are mock values, not an assessment of your property. "
        "Please arrange professional roof and financial review before considering any installation.",
        caveats=["MOCK DRAFT only; no owner identity was looked up and no message was sent."],
    )


def get_model(model_id: str) -> MatterportModel:
    """Return fixed fake Matterport metadata without authentication or network access."""
    return MatterportModel(id=model_id, metadata={"is_mock": True, "content_available": False},
                           raw_response={"source": "synthetic Matterport placeholder"})
