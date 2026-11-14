"""verify_brief: every number needs a real [E#] tag whose evidence contains it."""

import pytest

from backend.agent.citations import verify_brief
from backend.agent.evidence import Ledger


@pytest.fixture
def ledger() -> Ledger:
    """E1 is a sample of 12 buildings; E2 is a regression result."""
    ledger = Ledger()
    ledger.add("sample_buildings", {"market_id": 14, "property_type": 11, "limit": 40},
               {"count": 12, "buildings": [{"id": 990001, "max_rent": 1950.0}]}, "mock")
    ledger.add("run_regression", {"question": "parking and rent"},
               {"sample_size": 12, "coefficients": {"parking_space_count": 1.6372},
                "p_values": {"parking_space_count": 0.0934},
                "confidence_intervals": {"parking_space_count": [-0.9, 4.1]}, "r_squared": 0.29}, "mock")
    return ledger


def test_correct_brief_passes(ledger: Ledger) -> None:
    """Cited numbers that match their evidence produce no problems."""
    brief = (
        "## Question\nDo parking spaces raise rent?\n\n"
        "## Method\nWe sampled n = 12 [E1] buildings, requesting 40 [E1].\n\n"
        "## Result\nThe coefficient is 1.6372 [E2] (95% CI -0.9 to 4.1 [E2]), R² 0.29 [E2]. "
        "Building 990001 [E1] asks $1,950 [E1] per month.\n"
    )
    assert verify_brief(brief, ledger) == []


def test_uncited_number_fails(ledger: Ledger) -> None:
    """A number in a sentence without any tag is a problem."""
    problems = verify_brief("We sampled 12 buildings.\n\nThe coefficient is 1.6372 [E2].", ledger)
    assert len(problems) == 1 and problems[0].startswith("12 has no evidence tag")


def test_number_not_matching_its_evidence_fails(ledger: Ledger) -> None:
    """A tagged number absent from that entry's JSON is a problem, even if another entry has it."""
    assert verify_brief("The coefficient is 12.4 [E2].", ledger) == ["12.4 not found in E2"]
    # 1.6372 exists in E2 but not in E1, and the governing tag is E1.
    assert verify_brief("The coefficient is 1.6372 [E1].", ledger) == ["1.6372 not found in E1"]


def test_unknown_tag_fails(ledger: Ledger) -> None:
    """Tags must refer to ledger entries."""
    assert verify_brief("n = 12 [E9].", ledger) == ["[E9] is not in the evidence ledger"]


def test_years_tags_list_markers_and_ci_phrase_are_ignored(ledger: Ledger) -> None:
    """Years, E-tag digits, list/heading numbering and '95% CI' are not checked."""
    brief = ("## 1. Question\nSince 1984 and through 2025, rents changed.\n"
             "1. First caveat.\n2) Second caveat.\n- The 95% CI is wide [E2].\nSee [E2].")
    assert verify_brief(brief, ledger) == []


def test_enumerators_are_not_numbers(ledger: Ledger) -> None:
    """'(1)', '1)' and '1.' at the start of an item are numbering, also after a bullet."""
    brief = ("(1) Correlation is not causation.\n2) The sample is small.\n3. Imagery is partial.\n"
             "- (4) Rents are asking rents.\n## (5) Caveats\n")
    assert verify_brief(brief, ledger) == []
    # The same digits mid-sentence are still checked.
    assert verify_brief("We saw (7) outliers.", ledger)[0].startswith("7 has no evidence tag")


@pytest.mark.parametrize("phrase", ["an 80 m square", "an 80-m square", "an 80m square",
                                    "an 80 meter square", "an 80-meter square", "80 metres across"])
def test_method_constant_80_m_is_exempt(ledger: Ledger, phrase: str) -> None:
    """The documented 80 m image square is a methodology constant, not a result."""
    assert verify_brief(f"Measurements cover {phrase} around each point.", ledger) == []


def test_bare_80_and_wrong_numbers_still_fail(ledger: Ledger) -> None:
    """Without a unit, 80 is an ordinary number; wrong numbers still fail."""
    assert verify_brief("About 80 buildings were sampled.", ledger)[0].startswith("80 has no evidence tag")
    assert verify_brief("About 80 [E1] buildings, 180 m apart [E1].", ledger) == [
        "80 not found in E1", "180 not found in E1"]
    assert verify_brief("An 80 m square held 13 [E1] buildings.", ledger) == ["13 not found in E1"]


def test_rounding_and_percent_forms_are_accepted(ledger: Ledger) -> None:
    """Rounded values and x100 percentage forms of evidence values match."""
    brief = ("The coefficient is about 1.64 [E2] or 1.6 [E2]. "
             "The p-value is 0.09 [E2], or 9.3% [E2]. R² is 29% [E2]. Rent is $1,950.00 [E1].")
    assert verify_brief(brief, ledger) == []
    assert verify_brief("The p-value is 0.5 [E2].", ledger) == ["0.5 not found in E2"]


def test_each_number_uses_the_next_tag_in_its_sentence(ledger: Ledger) -> None:
    """With several tags in one sentence, a number is checked against the tag after it."""
    assert verify_brief("n = 12 [E1], coefficient 1.6372 [E2].", ledger) == []
    assert verify_brief("coefficient 1.6372 [E1], n = 12 [E2].", ledger) == ["1.6372 not found in E1"]
