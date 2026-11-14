"""Accuracy-check math tests (no data files, no model calls)."""

from scripts import accuracy_check as ac


def test_surface_spaces_counts_only_surface_types() -> None:
    parking = [{"parkingType": 1, "totalSpaces": 20}, {"parkingType": 2, "totalSpaces": 5},
               {"parkingType": 3, "totalSpaces": 100}, {"parkingType": 1, "totalSpaces": None}]
    assert ac.surface_spaces(parking) == 25
    assert ac.surface_spaces([{"parkingType": 3, "totalSpaces": 9}]) is None
    assert ac.surface_spaces(None) is None


def test_summary_and_sentence() -> None:
    s = ac.summarize([(12, 10), (8, 10), (30, 20)])
    assert s["n"] == 3 and s["mean_abs_error"] == 4.7 and s["median_abs_error"] == 2
    assert s["within_20_pct"] == 2 and s["mean_signed_error"] == 3.3
    assert "within 2 spaces" in ac.pitch_sentence(s) and "overcount" in ac.pitch_sentence(s)
    assert ac.summarize([]) == {"n": 0}
