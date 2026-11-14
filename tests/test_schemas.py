"""Validate the minimal contracts for CityLab output models."""

import pytest
from pydantic import BaseModel, ValidationError

from backend.schemas import (
    AerialImage,
    AgentEvent,
    Building,
    RunResult,
    SolarEstimate,
    Study,
    VisionResult,
)


@pytest.mark.parametrize(
    ("model", "identifier"),
    [
        (Building, 1),
        (AerialImage, "image-1"),
        (VisionResult, "vision-1"),
        (Study, "study-1"),
        (SolarEstimate, "solar-1"),
        (AgentEvent, "event-1"),
        (RunResult, "run-1"),
    ],
)
def test_models_require_only_an_id(
    model: type[BaseModel], identifier: int | str
) -> None:
    """Each output accepts an id alone, requires it, and serializes to JSON."""
    with pytest.raises(ValidationError):
        model()

    instance = model(id=identifier)
    assert instance.model_dump(exclude_unset=True) == {"id": identifier}
    assert model.model_validate_json(instance.model_dump_json()) == instance
