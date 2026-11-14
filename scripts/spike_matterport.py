"""Throwaway spike: test whether Matterport's GraphQL API is reachable and what it returns.

Not a contract implementation. Run manually with a real model id:
    python -m scripts.spike_matterport <model_id>

Prints the raw response shape so a human can judge whether Matterport exposes
anything usable for building height/elevation before wiring it into the real
backend.tools.matterport module.
"""

import argparse
import json
import sys

import requests

from backend import config

QUERY = """
query SpikeModel($modelId: ID!) {
  model(id: $modelId) {
    id
    name
    visibility
    geocoordinates {
      latitude
      longitude
      altitude
      rotation { x y z w }
      source
    }
    dimensions { height width depth areaFloor areaCeiling areaWall volume units }
    dimensionEstimates { height width depth area units }
    floors {
      id
      sequence
      dimensions { height areaFloor volume units }
      dimensionEstimates { height area lowerElevation units }
    }
  }
}
"""

INTROSPECTION_QUERY = """
query SpikeIntrospectType($name: String!) {
  __type(name: $name) {
    name
    fields {
      name
      type { name kind ofType { name kind } }
    }
  }
}
"""


def _post(query: str, variables: dict[str, object]) -> dict[str, object]:
    """Send one GraphQL request and return the parsed JSON body."""
    if not config.MP_TOKEN_ID or not config.MP_TOKEN_SECRET:
        raise ValueError("Set MP_TOKEN_ID and MP_TOKEN_SECRET in .env before running this spike.")
    response = requests.post(
        config.MATTERPORT_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        auth=(config.MP_TOKEN_ID, config.MP_TOKEN_SECRET),
        timeout=30,
    )
    print(f"HTTP {response.status_code}", file=sys.stderr)
    try:
        return response.json()
    except ValueError:
        return {"error": "non-JSON response", "status_code": response.status_code, "text": response.text[:500]}


def run(model_id: str) -> dict[str, object]:
    """Fetch the model query with real height/geocoordinate/elevation fields."""
    return {"model": _post(QUERY, {"modelId": model_id})}


def main(argv: list[str] | None = None) -> int:
    """Query one Matterport model id and print the raw response, no secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="A real Matterport model id to test with.")
    args = parser.parse_args(argv)
    try:
        body = run(args.model_id)
    except Exception:
        print("Matterport spike failed; check MP_TOKEN_ID/MP_TOKEN_SECRET and the model id.", file=sys.stderr)
        return 1
    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
