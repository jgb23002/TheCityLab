"""Optional Matterport GraphQL integration using HTTP Basic authentication."""

from pydantic import JsonValue


def get_model(model_id: str) -> dict[str, JsonValue]:
    """TODO: Query MATTERPORT_GRAPHQL_URL using MP_TOKEN_ID and MP_TOKEN_SECRET."""
    # TODO: Resolve PropertyId to MatterportModelId via commercial-properties.csv.
    ...
