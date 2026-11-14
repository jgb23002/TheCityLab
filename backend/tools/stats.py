"""Statistical study tool: OLS regression with HC3 robust errors (statsmodels).

``run_study`` reads the outcome (rent or vacancy) and the predictor (green
cover, paved surface, parking, roof area) from the question's words, joins
image measurements to buildings by ``property_id``, and fits

    outcome ~ predictor + building_size_sq_ft + year_built
              [+ total_num_units] [+ C(building_class)]

with ``sm.OLS(...).fit(cov_type="HC3")``. Control selection, in this order:

1. A control more than 90% missing across the given buildings is dropped.
2. Listwise deletion: rows missing the outcome, predictor or a kept control are
   dropped; ``sample_size`` is the number of rows actually analyzed.
3. Collinearity: controls are added one at a time (building class as one block of
   dummies); a control whose addition gives any term a VIF above 10 is dropped (the
   later one is dropped, the earlier ones kept). Rows are then re-selected for the
   final controls.
4. Fewer rows than parameters + 5 is a clear failure, not an estimate.

Every dropped control and the listwise-deletion count are stated in ``caveats``;
``controls`` lists only the controls used. Coefficient, p-value and 95% CI are
reported for the predictor and every control term. Findings say "associated with",
never "causes". Missing results stay null. ``_regress`` is the reusable core.
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
# Numeric controls in priority order: when two are collinear, the later is dropped.
_NUMERIC_CONTROLS = ["building_size_sq_ft", "year_built", "total_num_units"]
_CLASS_CONTROL = "building_class"
_CLASS_LABELS = {1: "A", 2: "B", 3: "C", 4: "F"}
_MAX_MISSING_SHARE = 0.90  # drop a control missing for more than 90% of buildings
_VIF_LIMIT = 10.0
_EXTRA_ROWS = 5  # need at least (number of parameters + 5) complete rows


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


def _missing_share(df: pd.DataFrame, column: str) -> float:
    """Share of rows where a control is unknown (building class 0/unknown counts as missing)."""
    if column not in df.columns or len(df) == 0:
        return 1.0
    values = pd.to_numeric(df[column], errors="coerce")
    known = values.isin(list(_CLASS_LABELS)) if column == _CLASS_CONTROL else values.notna()
    return float(1 - known.mean())


def _complete_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Listwise deletion over *columns* (numeric), keeping only A/B/C/F building classes."""
    data = df[["id", *columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if _CLASS_CONTROL in columns:
        data = data[data[_CLASS_CONTROL].isin(list(_CLASS_LABELS))]
    return data


def _class_dummies(classes: pd.Series) -> tuple[pd.DataFrame, str]:
    """Dummies for building class with the worst class present as the baseline."""
    present = sorted(int(c) for c in classes.unique())
    base = present[-1]
    dummies = pd.DataFrame({f"building_class_{_CLASS_LABELS[c]}": (classes == c).astype(float)
                            for c in present[:-1]}, index=classes.index)
    return dummies, _CLASS_LABELS[base]


def _max_vif(X: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """VIF of every column of X (computed with an intercept); inf/NaN mean collinear."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if X.shape[1] < 2:
        return 1.0, {c: 1.0 for c in X.columns}
    design = sm.add_constant(X.astype(float), has_constant="add").to_numpy()
    vifs: dict[str, float] = {}
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, column in enumerate(X.columns, start=1):
            value = float(variance_inflation_factor(design, i))
            vifs[column] = value if math.isfinite(value) else math.inf
    return max(vifs.values()), vifs


def _design(data: pd.DataFrame, predictor: str, numeric: list[str], use_class: bool) -> tuple[pd.DataFrame, str | None]:
    """Predictor + numeric controls (+ class dummies); returns (X, class baseline label)."""
    X = data[[predictor, *numeric]].astype(float)
    if use_class:
        dummies, base = _class_dummies(data[_CLASS_CONTROL])
        return pd.concat([X, dummies], axis=1), base
    return X, None


def _regress(
    df: pd.DataFrame, outcome: str, predictor: str, controls: list[str] | None = None,
    class_control: bool = True,
) -> dict[str, Any]:
    """Select controls, fit OLS with HC3 errors, and return tables and chart data.

    *controls* are numeric control columns in priority order (default
    ``_NUMERIC_CONTROLS``); with *class_control*, ``building_class`` is added as a
    categorical block. Returns ``{"error": ...}`` (with ``n``, ``n_dropped`` and
    ``caveats``) instead of raising or estimating when the data cannot support a fit.
    """
    controls = list(_NUMERIC_CONTROLS if controls is None else controls)
    candidates = [*controls, *([_CLASS_CONTROL] if class_control else [])]
    caveats: list[str] = []
    for column in (outcome, predictor):
        if column not in df.columns:
            return {"error": f"missing column {column}", "n": 0, "n_dropped": len(df), "caveats": caveats}

    # 1) Drop controls that are mostly missing.
    kept: list[str] = []
    for column in candidates:
        share = _missing_share(df, column)
        if share > _MAX_MISSING_SHARE:
            caveats.append(f"Control {column} was dropped: {share:.0%} of buildings have no value.")
        else:
            kept.append(column)

    # 2) Listwise deletion, then 3) collinearity, adding controls one at a time.
    data = _complete_rows(df, [outcome, predictor, *kept])
    numeric: list[str] = []
    use_class = False
    for column in kept:
        if column == _CLASS_CONTROL:
            if data[_CLASS_CONTROL].nunique() < 2:
                caveats.append("Control building_class was dropped: only one building class is present.")
                continue
            trial, _ = _design(data, predictor, numeric, True)
        else:
            trial, _ = _design(data, predictor, [*numeric, column], use_class)
        worst, _ = _max_vif(trial) if len(data) > trial.shape[1] + 1 else (1.0, {})
        if worst > _VIF_LIMIT:
            shown = "infinite" if math.isinf(worst) else f"{worst:.2f}"
            caveats.append(f"Control {column} was dropped: collinear with the predictor or an earlier "
                           f"control (VIF {shown} > {_VIF_LIMIT:g}).")
            continue
        if column == _CLASS_CONTROL:
            use_class = True
        else:
            numeric.append(column)

    final_columns = [outcome, predictor, *numeric, *([_CLASS_CONTROL] if use_class else [])]
    data = _complete_rows(df, final_columns)
    if use_class and data[_CLASS_CONTROL].nunique() < 2:
        use_class = False
        caveats.append("Control building_class was dropped: only one building class is present.")
        data = _complete_rows(df, final_columns[:-1])
    n, n_dropped = len(data), len(df) - len(data)
    if n_dropped:
        missing_by = {c: int(pd.to_numeric(df[c], errors="coerce").isna().sum()) for c in final_columns
                      if c != _CLASS_CONTROL}
        if use_class:
            missing_by[_CLASS_CONTROL] = int((~pd.to_numeric(df[_CLASS_CONTROL], errors="coerce")
                                              .isin(list(_CLASS_LABELS))).sum())
        detail = ", ".join(f"{c} {k}" for c, k in missing_by.items() if k)
        caveats.append(f"{n_dropped} of {len(df)} buildings were dropped (listwise deletion) because a "
                       f"value was unknown ({detail}).")

    X, base = _design(data, predictor, numeric, use_class) if n else (pd.DataFrame(), None)
    used_controls = [*numeric, *([f"building_class (categorical; {base} baseline)"] if use_class else [])]
    n_params = X.shape[1] + 1  # plus the intercept
    if n < n_params + _EXTRA_ROWS:
        return {"error": f"not enough complete rows (n={n}; need at least {n_params + _EXTRA_ROWS} for "
                         f"{n_params} parameters)", "n": n, "n_dropped": n_dropped, "caveats": caveats}
    if data[predictor].nunique() < 2:
        return {"error": f"predictor {predictor} does not vary (n={n})", "n": n, "n_dropped": n_dropped,
                "caveats": caveats}

    X = sm.add_constant(X, has_constant="add").rename(columns={"const": "intercept"})
    y = data[outcome].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit(cov_type="HC3")
    ci = model.conf_int(alpha=0.05)
    terms = list(model.params.index)
    _, vifs = _max_vif(X.drop(columns="intercept"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        "caveats": caveats,
        "coefficients": {t: float(model.params[t]) for t in terms},
        "std_errors": {t: float(model.bse[t]) for t in terms},
        "p_values": {t: float(model.pvalues[t]) for t in terms},
        "confidence_intervals": {t: (float(ci.loc[t, 0]), float(ci.loc[t, 1])) for t in terms},
        "r_squared": float(model.rsquared),
        "vif": {t: (v if math.isfinite(v) else None) for t, v in vifs.items()},
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
    ``surface_parking_spaces``. Controls: building size and year built, plus
    units and building class (categorical) when present; mostly-missing or
    collinear controls are dropped and named in ``caveats`` (see the module
    docstring). If the study cannot run (including fewer rows than parameters
    + 5), statistics stay null and ``caveats`` says why.
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
        return StatsResult(**base, sample_size=fit.get("n", 0),
                           findings={"error": fit["error"], "n_dropped": fit.get("n_dropped", 0)},
                           caveats=[f"Regression not run: {fit['error']}.", *fit.get("caveats", []), *caveats])

    coef = fit["coefficients"][predictor]
    p = fit["p_values"][predictor]
    ci = fit["confidence_intervals"][predictor]
    n = fit["n"]
    caveats.extend(fit["caveats"])
    if n < 20:
        caveats.append(f"Small sample (n={n}); treat the estimate with caution.")

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
            "vif": fit["vif"],
            "chart_data": fit["chart_data"],
        },
        caveats=caveats,
    )
