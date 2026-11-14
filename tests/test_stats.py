"""Audit tests for stats.run_study: HC3 OLS, control selection, listwise deletion, failures."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from backend.schemas import Building, Measurement, StatsResult
from backend.tools import stats

Q = "In Houston, do apartment communities with more green cover have higher asking rents?"


def synthetic(n: int = 40, seed: int = 1, **overrides) -> tuple[list[Building], list[Measurement]]:
    """Independent size/year/units/class plus green cover, with a known rent model."""
    rng = np.random.default_rng(seed)
    buildings, measurements = [], []
    for i in range(n):
        fields = dict(
            id=i + 1,
            building_size_sq_ft=float(rng.uniform(20_000, 400_000)),
            year_built=int(rng.integers(1960, 2020)),
            total_num_units=int(rng.integers(20, 600)),
            building_class=int(rng.integers(1, 4)),
        )
        green = float(rng.uniform(0, 40))
        fields["max_rent"] = (900 + 12 * green + 0.001 * fields["building_size_sq_ft"]
                              + 2 * (fields["year_built"] - 1960) + 150 * (3 - fields["building_class"])
                              + float(rng.normal(0, 40)))
        for key, fn in overrides.items():
            fields[key] = fn(i, fields)
        buildings.append(Building(**fields))
        measurements.append(Measurement(id=f"m{i}", property_id=i + 1, green_cover_percent=green))
    return buildings, measurements


def direct_fit(buildings: list[Building], measurements: list[Measurement]) -> sm.regression.linear_model.RegressionResultsWrapper:
    """The same model fitted directly with statsmodels (class C = 3 is the baseline)."""
    green = {m.property_id: m.green_cover_percent for m in measurements}
    df = pd.DataFrame([{**b.model_dump(), "green_cover_percent": green[b.id]} for b in buildings])
    X = df[["green_cover_percent", "building_size_sq_ft", "year_built", "total_num_units"]].astype(float)
    X["building_class_A"] = (df["building_class"] == 1).astype(float)
    X["building_class_B"] = (df["building_class"] == 2).astype(float)
    X = sm.add_constant(X).rename(columns={"const": "intercept"})
    return sm.OLS(df["max_rent"].astype(float), X).fit(cov_type="HC3")


def test_matches_direct_statsmodels_hc3_to_6_decimals() -> None:
    """Coefficients, HC3 p-values and 95% CIs equal a direct statsmodels fit."""
    buildings, measurements = synthetic()
    result = stats.run_study(Q, buildings, measurements)
    direct = direct_fit(buildings, measurements)
    ci = direct.conf_int(alpha=0.05)
    assert result.sample_size == 40 and result.method == "OLS with HC3 robust standard errors"
    assert set(result.coefficients) == set(direct.params.index)
    for term in direct.params.index:
        assert result.coefficients[term] == pytest.approx(direct.params[term], abs=1e-6)
        assert result.p_values[term] == pytest.approx(direct.pvalues[term], abs=1e-6)
        low, high = result.confidence_intervals[term]
        assert round(low, 6) == round(ci.loc[term, 0], 6) and round(high, 6) == round(ci.loc[term, 1], 6)
    assert direct.cov_type == "HC3"


def test_reports_predictor_and_every_control_and_lists_only_used_controls() -> None:
    """Every control term has a coefficient, p-value and CI; controls lists exactly those used."""
    buildings, measurements = synthetic()
    result = stats.run_study(Q, buildings, measurements)
    assert result.controls == ["building_size_sq_ft", "year_built", "total_num_units",
                               "building_class (categorical; C baseline)"]
    terms = {"intercept", "green_cover_percent", "building_size_sq_ft", "year_built", "total_num_units",
             "building_class_A", "building_class_B"}
    assert set(result.coefficients) == set(result.p_values) == set(result.confidence_intervals) == terms
    assert not any("was dropped" in c for c in result.caveats)
    assert set(result.findings["vif"]) == terms - {"intercept"}
    StatsResult.model_validate_json(result.model_dump_json())


def test_collinear_control_is_dropped_and_named() -> None:
    """units = size / 1000 exactly: the later control (units) is dropped, the earlier kept."""
    buildings, measurements = synthetic(total_num_units=lambda i, f: int(f["building_size_sq_ft"] // 1000) * 1)
    for b in buildings:
        b.building_size_sq_ft = float(b.total_num_units * 1000)  # make it exactly collinear
    result = stats.run_study(Q, buildings, measurements)
    assert "total_num_units" not in result.controls and "building_size_sq_ft" in result.controls
    assert "total_num_units" not in result.coefficients
    assert any(c.startswith("Control total_num_units was dropped: collinear") for c in result.caveats)


def test_mostly_missing_control_is_dropped_not_used_for_deletion() -> None:
    """Size known for only 3 of 40 buildings (>90% missing): dropped, and no rows lost to it."""
    buildings, measurements = synthetic(building_size_sq_ft=lambda i, f: f["building_size_sq_ft"] if i < 3 else None)
    result = stats.run_study(Q, buildings, measurements)
    assert result.sample_size == 40
    assert result.controls == ["year_built", "total_num_units", "building_class (categorical; C baseline)"]
    assert any(c == "Control building_size_sq_ft was dropped: 92% of buildings have no value." for c in result.caveats)
    assert not any("listwise" in c for c in result.caveats)


def test_missing_rows_are_counted_and_sample_size_is_rows_analyzed() -> None:
    """3 rows without rent and 2 (other) rows without year: 5 dropped, n = 35."""
    buildings, measurements = synthetic()
    for b in buildings[:3]:
        b.max_rent = None
    for b in buildings[3:5]:
        b.year_built = None
    buildings[5].building_class = 0  # unknown class also counts as missing
    result = stats.run_study(Q, buildings, measurements)
    assert result.sample_size == 34 and len(result.findings["chart_data"]["scatter"]) == 34
    assert result.findings["n_dropped"] == 6
    assert ("6 of 40 buildings were dropped (listwise deletion) because a value was unknown "
            "(max_rent 3, year_built 2, building_class 1).") in result.caveats


def test_too_few_rows_fails_cleanly_with_no_estimates() -> None:
    """With 7 parameters, 11 rows is a clear failure and 12 rows fits."""
    buildings, measurements = synthetic(11)
    result = stats.run_study(Q, buildings, measurements)
    assert result.coefficients is None and result.p_values is None and result.confidence_intervals is None
    assert result.sample_size == 11 and result.interpretation is None
    assert result.findings["error"].startswith("not enough complete rows (n=11; need at least 12 for 7 parameters)")
    assert any(c.startswith("Regression not run: not enough complete rows") for c in result.caveats)
    assert stats.run_study(Q, *synthetic(12)).coefficients is not None


def test_dropped_rows_can_push_below_the_minimum() -> None:
    """Listwise deletion happens before the minimum-rows rule; both are reported.

    With class unknown everywhere (dropped as missing) there are 5 parameters, so at
    least 10 complete rows are needed: 12 - 3 = 9 fails, 13 - 3 = 10 fits.
    """
    def run(n: int) -> StatsResult:
        buildings, measurements = synthetic(n, building_class=lambda i, f: None)
        for b in buildings[:3]:
            b.max_rent = None
        return stats.run_study(Q, buildings, measurements)

    short = run(12)
    assert short.coefficients is None and short.sample_size == 9
    assert short.findings["error"].startswith("not enough complete rows (n=9; need at least 10 for 5 parameters)")
    assert any("3 of 12 buildings were dropped" in c for c in short.caveats)
    enough = run(13)
    assert enough.coefficients is not None and enough.sample_size == 10
    assert len(enough.coefficients) == 5


def test_single_building_class_is_dropped() -> None:
    """A class control with one class present cannot be estimated and is dropped with a caveat."""
    buildings, measurements = synthetic(building_class=lambda i, f: 2)
    result = stats.run_study(Q, buildings, measurements)
    assert not any(c.startswith("building_class") for c in result.controls)
    assert "Control building_class was dropped: only one building class is present." in result.caveats
