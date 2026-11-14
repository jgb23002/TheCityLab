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


# ---- markdown tables (retrofit ranking) ------------------------------------------------

@pytest.fixture
def rank_ledger() -> Ledger:
    """E1 is a two-building solar ranking."""
    ledger = Ledger()
    ledger.add("rank_buildings", {"estimates": 2}, {"ranked": [
        {"property_id": 990102, "rank": 1, "usable_roof_area_sq_ft": 4500.0, "capacity_kw": 60.0,
         "annual_energy_kwh": 90000.0, "payback_years": 7.1},
        {"property_id": 990104, "rank": 2, "usable_roof_area_sq_ft": 3512.0, "capacity_kw": 42.1,
         "annual_energy_kwh": 54730.0, "payback_years": 8.2}]}, "mock")
    return ledger


TABLE = ("| Rank | Property id | Usable roof area (sq ft) | Capacity (kW) | Annual energy (kWh) | Payback (years) |\n"
         "|---|---|---|---|---|---|\n"
         "| 1 | 990102 [E1] | 4,500 [E1] | 60.0 [E1] | 90,000 [E1] | 7.1 [E1] |\n"
         "| 2 | 990104 | 3512 | 42.1 | 54,730 | 8.2 [E1] |\n")


def test_table_row_tag_covers_its_numbers_and_rank_column_is_ignored(rank_ledger: Ledger) -> None:
    """Each row is one unit: one [E#] anywhere in the row covers it; ranks 1-10 are skipped."""
    assert verify_brief(TABLE, rank_ledger) == []
    # Ranks 3..10 in the Rank column are ignored even though no evidence contains them.
    for rank in (3, 9, 10):
        assert verify_brief(TABLE.replace("| 2 |", f"| {rank} |"), rank_ledger) == []


def test_table_rows_are_checked_individually(rank_ledger: Ledger) -> None:
    """A row without a tag fails even if a neighbouring row has one; wrong values fail."""
    untagged = TABLE.replace("8.2 [E1]", "8.2")
    assert {p.split(" ")[0] for p in verify_brief(untagged, rank_ledger)} == {"990104", "3512", "42.1", "54,730", "8.2"}
    wrong = TABLE.replace("60.0 [E1]", "66.0 [E1]")
    assert verify_brief(wrong, rank_ledger) == ["66.0 not found in E1"]


def test_rank_exemption_is_limited_to_the_rank_column_and_1_to_10(rank_ledger: Ledger) -> None:
    """Rank 11, or a bare small number in another column, is still checked."""
    assert verify_brief(TABLE.replace("| 2 |", "| 11 |"), rank_ledger) == ["11 not found in E1"]
    assert verify_brief(TABLE.replace("7.1 [E1]", "3 [E1]"), rank_ledger) == ["3 not found in E1"]
    # Without a Rank header, the first column is ordinary data.
    no_header = TABLE.replace("| Rank |", "| Order |")
    assert verify_brief(no_header.replace("| 2 |", "| 5 |"), rank_ledger) == ["5 not found in E1"]


def test_ids_and_exact_counts_must_match_exactly(rank_ledger: Ledger) -> None:
    """Property ids are not matched within 1%: 990103 is not 990102."""
    assert verify_brief("Property 990103 [E1] ranks first.", rank_ledger) == ["990103 not found in E1"]
    assert verify_brief("Property 990102 [E1] ranks first.", rank_ledger) == []


def test_question_section_restating_the_question_is_not_checked(ledger: Ledger) -> None:
    """Numbers in the "## Question" body are the user's words; later sections are checked."""
    brief = ("## Question\nWhich 10 Los Angeles apartment buildings are the best candidates?\n\n"
             "## Answer\nWe ranked 12 [E1] buildings.\n")
    assert verify_brief(brief, ledger) == []
    assert verify_brief(brief.replace("12 [E1]", "10"), ledger)[0].startswith("10 has no evidence tag")
