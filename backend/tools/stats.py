"""Statistical study interfaces for questions about CoStar buildings."""

from backend.schemas import Building, Measurement, StatsResult


def run_study(
    question: str, buildings: list[Building], measurements: list[Measurement] | None = None
) -> StatsResult:
    """TODO: Test a research question with documented fields and assumptions."""
    raise NotImplementedError
