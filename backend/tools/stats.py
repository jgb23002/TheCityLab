"""Statistical study tool: OLS regression with HC3 robust errors.

``run_study`` reads the outcome (rent or vacancy) and the predictor (green
cover, paved surface, parking, roof area) from the question's words, joins
image measurements to buildings by ``property_id``, and fits

    outcome ~ predictor + total_num_units + year_built + C(building_class)

Findings always say "associated with", never "causes". Missing results stay
null. ``_regress`` is the reusable core for the correction loop.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from backend.schemas import Building, Measurement, StatsResult

# Question words -> field names. First match in list order wins.
_OUTCOME_WORDS = [
    (("rent",), "max_rent"),
    (("vacanc",), "vacancy_percent"),
]
_PREDICTOR_WORDS = [
    (("green", "tree", "vegetation"), "green_cover_percent"),
    (("paved", "pavement", "hardscape", "asphalt"), "paved_surface_percent"),
    (("parking",), "parking_space_count"),
    (("roof",), "roof_area_sq_ft"),
]
_MEASUREMENT_FIELDS = {"green_cover_percent", "paved_surface_percent",
                       "parking_space_count", "roof_area_sq_ft"}
_NUMERIC_CONTROLS = ["total_num_units", "year_built"]
_CLASS_LABELS = {1: "A", 2: "B", 3: "C", 4: "F"}
_MIN_N = 8


def _infer(question: str, words: list[tuple[tuple[str, ...], str]]) -> str | None:
    """Return the field for the first keyword group found in *question*."""
    q = question.lower()
    hits = [(min(q.find(w) for w in ws if w in q), field)
            for ws, field in words if any(w in q for w in ws)]
    return min(hits)[1] if hits else None


def _study_id(question: str, ids: list[int]) -> str:
    raw = question + "|" + ",".join(map(str, sorted(ids)))
    return "stats-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _table(buildings: list[Building], measurements: list[Measurement] | None) -> pd.DataFrame:
    """One row per building with Building fields plus its measurement fields."""
    rows = {b.id: b.model_dump(exclude={"primary_address"}) for b in buildings}
    for m in measurements or []:
        if m.property_id in rows:
            for field in _MEASUREMENT_FIELDS:
                rows[m.property_id][field] = getattr(m, field)
    return pd.DataFrame(list(rows.values()))


def _regress(
    df: pd.DataFrame, outcome: str, predictor: str, controls: list[str],
    class_control: bool = True,
) -> dict[str, Any]:
    """Fit OLS with HC3 errors; return coefficient tables and chart data.

    *controls* are numeric columns. With *class_control*, ``building_class``
    becomes dummies (worst class present as baseline) when 2+ classes appear.
    Returns ``{"error": ...}`` instead of raising when the data cannot fit.
    """
    numeric = [c for c in controls if c in df.columns and df[c].notna().any()]
    cols = ["id", outcome, predictor, *numeric]
    if class_control and "building_class" in df.columns:
        cols.append("building_class")
    if outcome not in df.columns or predictor not in df.columns:
        return {"error": f"missing column {outcome if outcome not in df.columns else predictor}"}

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    n_before = len(data)
    data = data.dropna()
    if "building_class" in data.columns:
        data = data[data["building_class"].isin(list(_CLASS_LABELS))]
    n_dropped = n_before - len(data)

    X = data[[predictor, *numeric]].astype(float)
    used_controls = list(numeric)
    if "building_class" in data.columns and data["building_class"].nunique() >= 2:
        classes = sorted(int(c) for c in data["building_class"].unique())
        base = classes[-1]
        for c in classes[:-1]:
            X[f"building_class_{_CLASS_LABELS[c]}"] = (data["building_class"] == c).astype(float)
        used_controls.append(f"building_class (categorical; {_CLASS_LABELS[base]} baseline)")

    n = len(data)
    if n < max(_MIN_N, X.shape[1] + 3) or data[predictor].nunique() < 2:
        return {"error": f"not enough complete rows (n={n})", "n": n, "n_dropped": n_dropped}

    X = sm.add_constant(X, has_constant="add").rename(columns={"const": "intercept"})
    y = data[outcome].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit(cov_type="HC3")
    ci = model.conf_int(alpha=0.05)

    terms = list(model.params.index)
    pearson_r, pearson_p = stats.pearsonr(data[predictor], y)

    lo, hi = float(data[predictor].min()), float(data[predictor].max())
    at_means = X.mean()
    fit_line = []
    for x in np.linspace(lo, hi, 25):
        row = at_means.copy()
        row[predictor] = x
        fit_line.append({"x": float(x), "y": float(np.dot(model.params.values, row[terms].values))})

    return {
        "n": n,
        "n_dropped": n_dropped,
        "controls": used_controls,
        "coefficients": {t: float(model.params[t]) for t in terms},
        "std_errors": {t: float(model.bse[t]) for t in terms},
        "p_values": {t: float(model.pvalues[t]) for t in terms},
        "confidence_intervals": {t: (float(ci.loc[t, 0]), float(ci.loc[t, 1])) for t in terms},
        "r_squared": float(model.rsquared),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "chart_data": {
            "scatter": [{"id": int(i), "x": float(px), "y": float(py)}
                        for i, px, py in zip(data["id"], data[predictor], y)],
            "fit_line": fit_line,
        },
    }


def _sentence(outcome: str, predictor: str, coef: float, ci: tuple[float, float], p: float, n: int) -> str:
    """One plain-English finding that never claims causation."""
    pretty_p = predictor.replace("_", " ").replace("percent", "(percentage points)")
    pretty_o = outcome.replace("_", " ")
    direction = "higher" if coef > 0 else "lower"
    strength = "is" if p <= 0.05 else "is weakly and not significantly"
    return (
        f"Across {n} buildings, each additional unit of {pretty_p} {strength} associated with "
        f"{abs(coef):.3g} {direction} {pretty_o} (95% CI {ci[0]:.3g} to {ci[1]:.3g}, p={p:.3f}), "
        f"holding the controls fixed. This is an association, not a cause."
    )


def run_study(
    question: str, buildings: list[Building], measurements: list[Measurement] | None = None
) -> StatsResult:
    """Test a research question with OLS + HC3 errors over the given buildings.

    Outcome and predictor come from the question's words; image-derived
    predictors come from *measurements* (joined on ``property_id``). Without
    measurements a parking question falls back to CoStar's
    ``surface_parking_spaces``. Controls: units, year built and building
    class (categorical). If the study cannot run, statistics stay null and
    ``caveats`` says why.
    """
    ids = [b.id for b in buildings]
    outcome = _infer(question, _OUTCOME_WORDS)
    predictor = _infer(question, _PREDICTOR_WORDS)
    caveats = [
        "Observational association only; this does not show cause and effect.",
        "Image measurements cover the Vexcel parcel or 80 m square, which may not match the whole community.",
    ]

    if predictor == "parking_space_count" and not measurements:
        predictor = "surface_parking_spaces"

    base = dict(id=_study_id(question, ids), question=question, property_ids=ids,
                method="OLS with HC3 robust standard errors", outcome=outcome, predictor=predictor)
    if outcome is None or predictor is None:
        missing = "outcome (rent or vacancy)" if outcome is None else "predictor (green, paved, parking or roof)"
        return StatsResult(**base, sample_size=0, caveats=[f"Could not tell the {missing} from the question.", *caveats])

    fit = _regress(_table(buildings, measurements), outcome, predictor, _NUMERIC_CONTROLS)
    if "error" in fit:
        return StatsResult(**base, sample_size=fit.get("n", 0), findings={"error": fit["error"]},
                           caveats=[f"Regression not run: {fit['error']}.", *caveats])

    coef = fit["coefficients"][predictor]
    p = fit["p_values"][predictor]
    ci = fit["confidence_intervals"][predictor]
    n = fit["n"]
    if n < 20:
        caveats.append(f"Small sample (n={n}); treat the estimate with caution.")
    if fit["n_dropped"]:
        caveats.append(f"{fit['n_dropped']} buildings were left out because a value was unknown.")

    return StatsResult(
        **base,
        sample_size=n,
        controls=fit["controls"],
        coefficients=fit["coefficients"],
        p_values=fit["p_values"],
        confidence_intervals=fit["confidence_intervals"],
        r_squared=fit["r_squared"] if math.isfinite(fit["r_squared"]) else None,
        interpretation=_sentence(outcome, predictor, coef, ci, p, n),
        findings={
            "std_errors": fit["std_errors"],
            "pearson_r": fit["pearson_r"],
            "pearson_p": fit["pearson_p"],
            "n_dropped": fit["n_dropped"],
            "chart_data": fit["chart_data"],
        },
        caveats=caveats,
    )
