"""Outreach draft interfaces for future building research results."""

from pydantic import JsonValue


def draft_outreach(
    building: dict[str, JsonValue], solar_estimate: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """TODO: Draft an outreach message for review without sending it."""
    ...
