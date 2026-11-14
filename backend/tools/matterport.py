"""Optional Matterport GraphQL integration using HTTP Basic authentication."""

from backend.schemas import MatterportModel


def get_model(model_id: str) -> MatterportModel:
    """TODO: Query MATTERPORT_GRAPHQL_URL using MP_TOKEN_ID and MP_TOKEN_SECRET."""
    # TODO: Resolve PropertyId to MatterportModelId via commercial-properties.csv.
    raise NotImplementedError
