"""Local CoStar JSON Lines interfaces; there is no CoStar API."""

from pydantic import JsonValue


def load_properties() -> list[dict[str, JsonValue]]:
    """TODO: Use pandas.read_json(path, lines=True) on commercial-property.jsonl."""
    # TODO: Read from COSTAR_DATA_DIR and return JSON-serializable records.
    ...


def get_property(id: int) -> dict[str, JsonValue] | None:
    """TODO: Find a local commercial property by its documented integer id."""
    ...


def find_buildings(
    market_id: int, property_type: int, limit: int = 20
) -> list[dict[str, JsonValue]]:
    """TODO: Filter primaryAddress.researchMarketId and propertyType; Houston is 14."""
    # TODO: Check models/commercial-property.md and schemas/enums.json first.
    ...
