"""Statistical study interfaces for questions about CoStar buildings."""

from pydantic import JsonValue


def run_study(
    question: str, buildings: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    """TODO: Test a research question with documented fields and assumptions."""
    ...
