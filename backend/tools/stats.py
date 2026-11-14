"""Statistical study tool: OLS regression with HC3 robust errors.

Runs a simple OLS regression of *outcome* on *predictor* plus *controls*
(building size and year built), reports the coefficient, 95 % confidence
interval, p-value, Pearson r, and writes a one-sentence plain-English finding.

The finding always says "associated with", never "causes".
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def run_study(
    question: str,
    buildings: list[dict[str, JsonValue]],
    outcome: str = "vacancy_percent",
    predictor: str = "parking_spaces",
    controls: list[str] | None = None,
) -> dict[str, JsonValue]:
    """Run an OLS regression and return a Study-shaped result dict.

    The regression model is::

        outcome ~ predictor [+ controls] + const

    with HC3 (heteroscedasticity-consistent) robust standard errors.

    Args:
        question: The plain-English research question being tested.
        buildings: List of building dicts, each containing *outcome*,
            *predictor*, and any *controls* as numeric fields.
        outcome: Dependent variable name (a key present in each building dict).
        predictor: Independent variable of interest (created by vision model).
        controls: Additional regressors (default: ``["gross_building_area_sq_ft",
            "year_built"]``).

    Returns:
        A JSON-serialisable Study dict with keys:

        - ``question`` – the input question
        - ``outcome``, ``predictor``, ``controls``
        - ``n`` – number of rows used after dropping missing values
        - ``n_dropped`` – rows excluded due to missing values
        - ``coefficient`` – OLS coefficient on *predictor*
        - ``std_error`` – HC3 robust standard error
        - ``p_value`` – two-sided p-value
        - ``conf_int_95`` – [low, high] 95 % confidence interval
        - ``r_squared`` – R² of the full model
        - ``pearson_r`` – bivariate Pearson r between outcome and predictor
        - ``pearson_p`` – p-value for the bivariate correlation
        - ``finding`` – one plain-English sentence
        - ``warnings`` – list of caution strings (e.g. small n, non-significant)
        - ``chart_data`` – ``{scatter: [{id, x, y}], fit_line: [{x, y}]}``
    """
    if controls is None:
        controls = ["gross_building_area_sq_ft", "year_built"]

    needed_cols = {outcome, predictor, *controls, "id"}
    df = pd.DataFrame(buildings)

    # Keep only columns that exist in the data
    present_controls = [c for c in controls if c in df.columns]
    all_cols = [c for c in [outcome, predictor] + present_controls if c in df.columns]

    df_model = df[["id"] + all_cols].copy()
    df_model = df_model.apply(pd.to_numeric, errors="coerce")

    n_before = len(df_model)
    df_model = df_model.dropna(subset=all_cols)
    n_dropped = n_before - len(df_model)
    n = len(df_model)

    result_warnings: list[str] = []

    if n < 10:
        result_warnings.append(f"Only {n} complete observations — results are unreliable.")

    if n < 2:
        return {
            "question": question,
            "outcome": outcome,
            "predictor": predictor,
            "controls": present_controls,
            "n": n,
            "n_dropped": n_dropped,
            "coefficient": None,
            "std_error": None,
            "p_value": None,
            "conf_int_95": None,
            "r_squared": None,
            "pearson_r": None,
            "pearson_p": None,
            "finding": "Insufficient data to run the regression.",
            "warnings": result_warnings,
            "chart_data": {"scatter": [], "fit_line": []},
        }

    # Bivariate correlation
    pearson_r, pearson_p = stats.pearsonr(df_model[predictor], df_model[outcome])

    # OLS regression
    X = df_model[[predictor] + present_controls]
    X = sm.add_constant(X)
    y = df_model[outcome]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit(cov_type="HC3")

    coef = float(model.params[predictor])
    std_err = float(model.bse[predictor])
    p_val = float(model.pvalues[predictor])
    conf_int = model.conf_int(alpha=0.05).loc[predictor].tolist()
    r_sq = float(model.rsquared)

    if p_val > 0.05:
        result_warnings.append(
            f"The result is not statistically significant at p=0.05 (p={p_val:.3f})."
        )
    if n < 20:
        result_warnings.append(
            f"Small sample (n={n}); treat results with caution."
        )

    # One-sentence finding
    direction = "higher" if coef > 0 else "lower"
    sig_phrase = "significantly " if p_val <= 0.05 else ""
    finding = (
        f"Each additional unit of {predictor.replace('_', ' ')} is {sig_phrase}associated "
        f"with {abs(coef):.3g} {direction} {outcome.replace('_', ' ')} "
        f"(coefficient={coef:.3g}, p={p_val:.3f}, n={n})."
    )

    # Chart data
    scatter = [
        {"id": int(row["id"]), "x": float(row[predictor]), "y": float(row[outcome])}
        for _, row in df_model.iterrows()
    ]

    x_min, x_max = float(df_model[predictor].min()), float(df_model[predictor].max())
    # Fitted line: hold controls at their mean
    mean_controls = {c: float(df_model[c].mean()) for c in present_controls}
    fit_line = []
    for x_val in np.linspace(x_min, x_max, 50):
        row_data = {"const": 1.0, predictor: x_val, **mean_controls}
        x_row = pd.Series(row_data)[[col for col in model.params.index if col in row_data]]
        y_fit = float(model.predict(x_row.values.reshape(1, -1))[0])
        fit_line.append({"x": x_val, "y": y_fit})

    return {
        "question": question,
        "outcome": outcome,
        "predictor": predictor,
        "controls": present_controls,
        "n": n,
        "n_dropped": n_dropped,
        "coefficient": coef,
        "std_error": std_err,
        "p_value": p_val,
        "conf_int_95": conf_int,
        "r_squared": r_sq,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "finding": finding,
        "warnings": result_warnings,
        "chart_data": {"scatter": scatter, "fit_line": fit_line},
    }


def run_regression(
    df: pd.DataFrame | list[dict[str, JsonValue]],
    outcome: str,
    predictor: str,
    controls: list[str] | None = None,
    question: str = "",
) -> dict[str, JsonValue]:
    """Team-plan name for :func:`run_study`: OLS with HC3 errors on a table.

    Args:
        df: A DataFrame or list of building dicts; must have an ``id`` column.
        outcome: Dependent variable, e.g. ``"max_rent"``.
        predictor: Variable of interest, e.g. ``"green_cover_percent"``.
        controls: Control columns. Defaults to size (``total_num_units`` or
            ``gross_building_area_sq_ft``, whichever the table has) and
            ``year_built``; columns the table lacks are skipped.
        question: Optional plain-English question carried into the result.

    Returns:
        The same dict as :func:`run_study` (coefficient, ``conf_int_95``,
        ``p_value``, ``n``, ``finding``, ``chart_data`` ...).
    """
    buildings = df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else df
    if controls is None:
        controls = ["total_num_units", "gross_building_area_sq_ft", "year_built"]
    return run_study(question, buildings, outcome=outcome, predictor=predictor, controls=controls)
