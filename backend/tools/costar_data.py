"""Local CoStar JSON Lines interfaces; there is no CoStar API."""

from backend.schemas import Building


def load_properties() -> list[Building]:
    """TODO: Use pandas.read_json(path, lines=True) on commercial-property.jsonl."""
    # TODO: Read from COSTAR_DATA_DIR and return JSON-serializable records.
    raise NotImplementedError


def get_property(id: int) -> Building | None:
    """TODO: Find a local commercial property by its documented integer id."""
    raise NotImplementedError


def find_buildings(
    market_id: int, property_type: int, limit: int = 20
) -> list[Building]:
    """TODO: Filter primaryAddress.researchMarketId and propertyType; Houston is 14."""
    # TODO: Check models/commercial-property.md and schemas/enums.json first.
    raise NotImplementedError
