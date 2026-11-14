"""Planning interfaces for the CityLab research agent."""

from pydantic import JsonValue


def plan_research(question: str) -> dict[str, JsonValue]:
    """TODO: Plan building measurement, statistical studies, and solar ranking."""
    ...
