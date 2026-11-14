"""Outreach draft interfaces for future building research results."""

from backend.schemas import Building, OutreachDraft, SolarResult


def draft_outreach(
    building: Building, solar_estimate: SolarResult
) -> OutreachDraft:
    """TODO: Draft an outreach message for review without sending it."""
    raise NotImplementedError
