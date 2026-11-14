"""Each agent prompt exists and states its key rule."""

import pytest

from backend.agent import prompts

NAMES = ("ORCHESTRATOR_PROMPT", "SAMPLER_PROMPT", "INSPECTOR_PROMPT", "STATISTICIAN_PROMPT", "WRITER_PROMPT")


@pytest.mark.parametrize("name", NAMES)
def test_prompt_exists(name: str) -> None:
    """All five prompts are nonempty strings."""
    assert isinstance(getattr(prompts, name), str) and getattr(prompts, name).strip()


def test_writer_prompt_states_the_citation_rule() -> None:
    """The writer must tag every number with [E#] and use only ledger numbers."""
    text = prompts.WRITER_PROMPT
    assert "[E#]" in text and "[E7]" in text and "ledger" in text
    for section in ("Question", "Answer", "Method", "Result", "Confidence and caveats"):
        assert section in text


def test_orchestrator_order_and_demo_label() -> None:
    """The orchestrator delegates in order and labels mock results."""
    text = prompts.ORCHESTRATOR_PROMPT
    positions = [text.index(name) for name in ("sampler", "inspector", "statistician", "writer")]
    assert positions == sorted(positions)
    assert "(demo data)" in text and "Never invent numbers" in text


def test_orchestrator_always_runs_statistician_and_keeps_rent_from_inspector() -> None:
    """The statistician always runs; rent comes from sample_buildings, not the inspector."""
    text = prompts.ORCHESTRATOR_PROMPT
    assert "ALWAYS call it after the inspector, even when n is small" in text
    assert "Never ask the inspector for rent" in text and "max_rent" in text


def test_inspector_returns_only_measurements_and_never_rent() -> None:
    """The inspector reports image measurements and must not mention or request rent."""
    text = prompts.INSPECTOR_PROMPT
    assert "Return only measurements" in text
    for field in ("parking_space_count", "green_cover_percent", "paved_surface_percent"):
        assert field in text
    assert "Do not mention or request rent" in text
    assert "max_rent" not in text


def test_statistician_copies_every_reported_value_exactly() -> None:
    """Outcome, predictor, controls, coefficient, CI, p-value and n are copied exactly."""
    text = prompts.STATISTICIAN_PROMPT
    assert "EXACTLY" in text
    for item in ("outcome", "predictor", "controls", "coefficient", "confidence interval",
                 "p-value", "sample_size"):
        assert item in text
    assert "building size" not in text  # the tool chooses the controls


def test_writer_controls_power_image_area_and_start_rules() -> None:
    """Writer lists only returned controls, flags n < 15, avoids area numbers, starts at the heading."""
    text = prompts.WRITER_PROMPT
    assert 'list ONLY the controls in the run_regression\nresult\'s "controls" field' in text
    assert 'Never say "building size" unless it is listed' in text
    assert "below 15" in text and "underpowered" in text
    assert "Never give the image area size as\na number" in text
    assert "a small square around each address point" in text
    assert 'Start your reply at "## Question"' in text
    assert "80" not in text


def test_writer_never_concludes_without_a_regression_result() -> None:
    """With no successful regression the writer must report an incomplete study."""
    text = prompts.WRITER_PROMPT
    assert "NEVER state a conclusion" in text and "study is incomplete" in text
    assert "statistician" in prompts.ORCHESTRATOR_PROMPT and "call it\n   ONCE" in prompts.ORCHESTRATOR_PROMPT
    assert "Never pass buildings or measurements" in prompts.STATISTICIAN_PROMPT
