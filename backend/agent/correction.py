"""Correction loop: patch one measurement, re-run the regression, explain what changed.

No LLM or network call happens here. The regression is re-run with the same question
(so the same outcome and predictor) over the same property ids as the original
run_regression entry, using the same implementation as before (stats.run_study for
real studies, mock.run_study for demo studies, so before and after are comparable).
Every number in the generated text is cited to a ledger entry: the correction itself
(tool "correct_measurement", with before/after values), the original regression and
the new one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from backend.agent.citations import verify_brief
from backend.agent.evidence import Ledger
from backend.agent.registry import regression_ok
from backend.agent.study_store import StudyRun, next_corrected_id
from backend.schemas import Measurement, MeasurementCorrection, SolarResult, StatsResult

CORRECTION_HEADING = "## What changed after your correction"
_RESULT_SECTION = re.compile(r"(^## Result[^\n]*\n)(.*?)(?=^## |\Z)", re.M | re.S)
_CORRECTION_SECTION = re.compile(rf"\n*^{re.escape(CORRECTION_HEADING)}\n.*?(?=^## |\Z)", re.M | re.S)
_ANSWER_HEADING = re.compile(r"^## Answer[^\n]*\n", re.M)
STALE_ANSWER_NOTE = ("> Note: this answer was written before your correction changed the conclusion; "
                     "see \"What changed after your correction\".\n")


class CorrectionError(Exception):
    """A correction that cannot be applied; .status is the HTTP status to return."""

    status = 400


class CorrectionNotFound(CorrectionError):
    """Unknown measurement id."""

    status = 404


class InvalidCorrection(CorrectionError):
    """Values out of range, nothing to change, or a study with no regression."""

    status = 422


class UnverifiableCorrection(CorrectionError):
    """The corrected brief would contain numbers that fail the citation check."""

    status = 500


@dataclass
class CorrectionResult:
    """The corrected study run plus what the UI needs to show the change."""

    run: StudyRun
    corrected_measurement: Measurement
    changes: list[dict[str, Any]]
    conclusion: str
    correction_entry: dict[str, Any]
    regression_entry: dict[str, Any]
    previous_regression_entry: dict[str, Any]
    problems: list[str] = field(default_factory=list)


def _num(value: Any) -> str:
    """Write a number so verify_brief can match it to its evidence (no scientific notation)."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1):
        return str(int(value))
    text = f"{value:.4g}"
    if "e" in text:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def _cite(value: Any, eid: str) -> str:
    """A value followed by its evidence tag, or 'unknown' (no number, no tag needed)."""
    return "unknown" if value is None else f"{_num(value)} [{eid}]"


def _cite_p(value: Any, eid: str) -> str:
    """A p-value with its tag; values below one in a million are written in words, not as 0."""
    if isinstance(value, (int, float)) and 0 <= value < 1e-6:
        return f"below one in a million [{eid}]"
    return _cite(value, eid)


def _original_regression(run: StudyRun) -> dict[str, Any]:
    """The run_regression entry behind study.stats (else the latest successful one)."""
    entries = [e for e in run.ledger.all() if e["tool"] == "run_regression" and "error" not in e["result"]]
    stats_id = run.study.stats.id if run.study.stats else None
    match = [e for e in entries if e["result"].get("id") == stats_id] or [e for e in entries if regression_ok(e["result"])]
    if not match:
        raise InvalidCorrection("This study has no regression to re-run, so it cannot be corrected.")
    return match[-1]


def _predictor_stats(result: dict[str, Any]) -> dict[str, Any]:
    """Coefficient, CI, p-value and sample size for the predictor (None when unknown)."""
    predictor = result.get("predictor")
    ci = (result.get("confidence_intervals") or {}).get(predictor) or (None, None)
    return {"coefficient": (result.get("coefficients") or {}).get(predictor), "ci_low": ci[0], "ci_high": ci[1],
            "p_value": (result.get("p_values") or {}).get(predictor), "sample_size": result.get("sample_size")}


def _direction(before: Any, after: Any) -> str:
    """'up', 'down', 'unchanged' or 'n/a' for the UI."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "unchanged" if before == after else "n/a"
    if math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-12):
        return "unchanged"
    return "up" if after > before else "down"


def conclusion_sentence(before: dict[str, Any], after: dict[str, Any], is_mock: bool) -> str:
    """One plain sentence, from the numbers, on whether the conclusion changed (no digits)."""
    if after["coefficient"] is None:
        return "The conclusion changed: after the correction the regression no longer produces an estimate."
    if before["coefficient"] is None:
        return "The conclusion changed: after the correction the regression now produces an estimate."
    sig = lambda s: s["p_value"] is not None and s["p_value"] < 0.05  # noqa: E731
    sign = lambda s: "positive" if s["coefficient"] > 0 else "negative" if s["coefficient"] < 0 else "zero"  # noqa: E731
    demo = (" Demo studies use fixed illustrative regression numbers, so a correction cannot move them."
            if is_mock and before == after else "")
    if sig(before) != sig(after):
        state = "now statistically significant" if sig(after) else "no longer statistically significant"
        return f"The conclusion changed: the association is {state} at the conventional threshold." + demo
    if sign(before) != sign(after):
        return (f"The conclusion changed: the direction of the association flipped from {sign(before)} to "
                f"{sign(after)}, though its statistical significance did not change.") + demo
    state = "statistically significant" if sig(after) else "not statistically significant"
    return f"The conclusion did not change: the association is still {sign(after)} and still {state}." + demo


def _patched_measurement(original: Measurement, patch: dict[str, Any], note: str | None,
                         fields: list[str], when: str) -> Measurement:
    """Apply the patch (explicit None clears to unknown), mark corrected, log the note."""
    data = original.model_dump(mode="json")
    data.update(patch)
    observations = dict(data.get("observations") or {})
    history = list(observations.get("corrections") or [])
    history.append({"fields": fields, "note": note, "at": when})
    observations["corrections"] = history
    if note:
        observations["user_note"] = note
    data.update({"observations": observations, "is_corrected": True})
    try:
        return Measurement.model_validate(data)
    except ValidationError:
        raise InvalidCorrection("Counts must be 0 or more and percentages must be between 0 and 100.") from None


def _rerun(question: str, buildings: list, measurements: list, source: str) -> StatsResult:
    """Same implementation as the original regression: mock for demo studies, else stats."""
    if source == "mock":
        from backend.tools import mock as implementation
    else:
        from backend.tools import stats as implementation
    return StatsResult.model_validate(implementation.run_study(question, buildings, measurements))


def _p_phrase(value: Any, eid: str) -> str:
    """'= 0.031 [E9]' or 'below one in a million [E9]' for the Result sentence."""
    cited = _cite_p(value, eid)
    return cited if cited.startswith("below") else f"= {cited}"


def _result_section(after: dict[str, Any], result: dict[str, Any], eid: str) -> str:
    """The new Result section body, citing the new regression entry."""
    predictor = result.get("predictor") or "the predictor"
    if after["coefficient"] is None:
        reason = (result.get("findings") or {}).get("error") or "no estimate"
        return f"After your correction the regression did not produce a result: {reason} [{eid}].\n\n"
    controls = ", ".join(result.get("controls") or []) or "none"
    return (f"After your correction, across n = {_cite(after['sample_size'], eid)} buildings, the {predictor} "
            f"coefficient is {_cite(after['coefficient'], eid)} (95% CI {_cite(after['ci_low'], eid)} to "
            f"{_cite(after['ci_high'], eid)}), with p {_p_phrase(after['p_value'], eid)}. Controls used: {controls}.\n\n")


def apply_correction(run: StudyRun, measurement_id: str, corrections: MeasurementCorrection) -> CorrectionResult:
    """Patch one measurement, re-run the regression and return a new, cited study run.

    The input run is never modified; the result has a new id "<original>-c<N>".
    """
    study = run.study
    original = next((m for m in study.measurements if m.id == measurement_id), None)
    if original is None:
        raise CorrectionNotFound("Measurement not found in this study.")
    patch = corrections.model_dump(mode="json", exclude_unset=True)
    note = patch.pop("note", None)
    if not patch:
        raise InvalidCorrection("Give at least one measurement field to correct.")
    fields = sorted(patch)
    when = datetime.now(timezone.utc).isoformat()
    corrected = _patched_measurement(original, patch, note, fields, when)

    if study.solar_results is not None:  # retrofit / property studies: re-estimate solar, re-rank
        return _apply_solar_correction(run, original, corrected, measurement_id, fields, note, when)
    ledger = Ledger.from_entries(run.ledger.to_json())  # never touch the original ledger
    previous = _original_regression(run)
    source = previous["source"]
    building = next((b for b in study.buildings if b.id == original.property_id), None)
    place = (building.name if building and building.name else f"property {original.property_id}")

    summarize = lambda m, f: len(m.detections or []) if f == "detections" else getattr(m, f)  # noqa: E731
    before_vals = {f: summarize(original, f) for f in fields}
    after_vals = {f: summarize(corrected, f) for f in fields}
    correction_entry = ledger.add("correct_measurement", {
        # The name is recorded so digits in it (e.g. "Park 45") are cited evidence too.
        "measurement_id": measurement_id, "property_id": original.property_id, "property_name": place,
        "before": before_vals, "after": after_vals, "note": note,
    }, corrected.model_dump(mode="json"), "mock" if study.is_mock else "real")

    ids = previous["args"].get("property_ids") or [b.id for b in study.buildings]
    by_id = {b.id: b for b in study.buildings}
    measurements = {m.property_id: m for m in study.measurements}
    measurements[corrected.property_id] = corrected
    buildings = [by_id[i] for i in ids if i in by_id]
    rows = [measurements[b.id] for b in buildings if b.id in measurements]
    question = previous["args"].get("question") or study.question
    new_stats = _rerun(question, buildings, rows, source)
    if (new_stats.outcome, new_stats.predictor) != (previous["result"].get("outcome"), previous["result"].get("predictor")):
        raise CorrectionError("The re-run did not reproduce the original outcome and predictor.")
    result = new_stats.model_dump(mode="json")
    regression_entry = ledger.add("run_regression", {
        "question": question, "property_ids": [b.id for b in buildings], "joined_rows": len(rows),
        "measurement_count": len(rows), "note": f"after correction of {measurement_id}",
    }, result, source)

    old_e, new_e, cor_e = previous["id"], regression_entry["id"], correction_entry["id"]
    before, after = _predictor_stats(previous["result"]), _predictor_stats(result)
    conclusion = conclusion_sentence(before, after, study.is_mock)
    predictor = result.get("predictor")

    changes = [{"label": f"{f} ({place})", "before": before_vals[f], "after": after_vals[f],
                "direction": _direction(before_vals[f], after_vals[f])} for f in fields]
    labels = {"coefficient": f"{predictor} coefficient", "ci_low": "95% CI low", "ci_high": "95% CI high",
              "p_value": "p-value", "sample_size": "sample size"}
    changes += [{"label": labels[k], "before": before[k], "after": after[k], "direction": _direction(before[k], after[k])}
                for k in labels]

    field_lines = "\n".join(
        f"- {f} for {place} (property {original.property_id} [{cor_e}]): "
        f"{_cite(before_vals[f], cor_e)} before, {_cite(after_vals[f], cor_e)} after." for f in fields)
    cite = lambda k, v, e: _cite_p(v, e) if k == "p_value" else _cite(v, e)  # noqa: E731
    stat_lines = "\n".join(
        f"- {labels[k]}: {cite(k, before[k], old_e)} before, {cite(k, after[k], new_e)} after." for k in labels)
    # The note is free text (it may contain numbers), so it is kept with the correction
    # entry, the measurement and the caveats rather than in the cited brief.
    block = (f"{CORRECTION_HEADING}\n\n{field_lines}\n{stat_lines}\n\n{conclusion}\n"
             + (f"\nYour note is saved with the correction [{cor_e}].\n" if note else ""))

    brief = _CORRECTION_SECTION.sub("\n\n", study.brief_markdown).rstrip()
    if _RESULT_SECTION.search(brief):
        brief = _RESULT_SECTION.sub(lambda m: m.group(1) + _result_section(after, result, new_e), brief, count=1)
    else:
        brief += "\n\n## Result\n" + _result_section(after, result, new_e)
    if conclusion.startswith("The conclusion changed") and STALE_ANSWER_NOTE not in brief:
        brief = _ANSWER_HEADING.sub(lambda m: m.group(0) + STALE_ANSWER_NOTE, brief, count=1)
    brief = brief.rstrip() + "\n\n" + block

    problems = verify_brief(brief, ledger)
    baseline = set(verify_brief(study.brief_markdown, run.ledger))
    new_problems = [p for p in problems if p not in baseline]
    if new_problems:
        raise UnverifiableCorrection("The corrected brief failed the citation check; nothing was saved.")

    new_id, n = next_corrected_id(run)
    caveat = (f"User correction {n}: {', '.join(fields)} of {place} (measurement {measurement_id}) was "
              "changed by hand and the regression was re-run.")
    caveats = list(study.caveats)
    if note:
        caveat += f" Note: {note}"
    new_study = study.model_copy(deep=True, update={
        "id": new_id, "brief_markdown": brief, "stats": new_stats, "caveats": [*caveats, caveat],
        "measurements": [corrected if m.id == measurement_id else m for m in study.measurements],
    })
    history = [*(run.meta.get("corrections") or []),
               {"n": n, "measurement_id": measurement_id, "fields": fields, "at": when,
                "correction_entry": cor_e, "regression_entry": new_e}]
    meta = {"original_id": run.meta.get("original_id") or study.id, "corrected_from": study.id,
            "correction_count": n, "corrections": history}
    return CorrectionResult(run=StudyRun(study=new_study, ledger=ledger, meta=meta), corrected_measurement=corrected,
                            changes=changes, conclusion=conclusion, correction_entry=correction_entry,
                            regression_entry=regression_entry, previous_regression_entry=previous, problems=problems)


# ---------------------------------------------------------------------------
# Retrofit / property studies: re-estimate solar, re-rank, re-screen, re-draft
# ---------------------------------------------------------------------------

_RANKING_SECTION = re.compile(r"(^## Ranking[^\n]*\n)(.*?)(?=^## |\Z)", re.M | re.S)
SOLAR_SCALING_NOTE = ("re-estimated after a correction by scaling the original estimate's kWh per kW "
                      "(no new PVWatts call)")


def _latest(ledger: Ledger, tool: str, property_id: int | None = None) -> dict[str, Any] | None:
    """The latest successful entry for a tool (optionally for one property)."""
    for entry in reversed(ledger.all()):
        r = entry["result"]
        if entry["tool"] == tool and "error" not in r and (property_id is None or r.get("property_id") == property_id):
            return entry
    return None


def rescale_solar(old: SolarResult, roof_sq_ft: float | None) -> SolarResult:
    """Deterministic rooftop re-estimate for a corrected roof area, from the old estimate's assumptions.

    usable = roof x usable fraction; kW = usable x W per sq ft / 1000; kWh = kW x (old kWh / old kW);
    cost = kW x 1000 x $/W; savings = kWh x $/kWh; payback = cost / savings. PVWatts output is linear
    in system size for one site and orientation, so scaling kWh per kW is valid without a new call.
    """
    from backend.tools import solar as solar_tool

    a = dict(old.assumptions or {})
    if not roof_sq_ft or roof_sq_ft <= 0:
        return old.model_copy(update={"usable_roof_area_sq_ft": None, "capacity_kw": None, "annual_energy_kwh": None,
                                      "payback_years": None, "score": None, "rank": None,
                                      "caveats": ["No measured roof area after the correction, so no solar estimate."]})
    fraction = a.get("usable_fraction_of_roof") or solar_tool.USABLE_FRACTION[None]
    wpsf = a.get("watts_per_usable_sq_ft") or solar_tool.WATTS_PER_USABLE_SQ_FT
    cost_w = a.get("installed_cost_usd_per_watt") or solar_tool.INSTALLED_COST_PER_WATT
    price = a.get("electricity_price_usd_per_kwh") or solar_tool.DEFAULT_PRICE_PER_KWH
    kwh_per_kw = (old.annual_energy_kwh / old.capacity_kw) if old.capacity_kw and old.annual_energy_kwh else solar_tool.DEFAULT_KWH_PER_KW
    usable = roof_sq_ft * fraction
    kw = usable * wpsf / 1000
    kwh = kw * kwh_per_kw
    cost = kw * 1000 * cost_w
    savings = kwh * price
    payback = round(cost / savings, 1) if savings > 0 else None
    a.update({"measured_roof_area_sq_ft": roof_sq_ft, "estimated_cost_usd": round(cost),
              "estimated_annual_savings_usd": round(savings), "estimated_payback_years": payback,
              "energy_source": f"{a.get('energy_source') or 'unknown'}; {SOLAR_SCALING_NOTE}"})
    return old.model_copy(update={"usable_roof_area_sq_ft": round(usable), "capacity_kw": round(kw, 1),
                                  "annual_energy_kwh": round(kwh), "payback_years": payback, "score": payback,
                                  "rank": None, "assumptions": a})


def _ranking_section(ranked: list[dict[str, Any]], eid: str, rows: int) -> str:
    """A regenerated, cited ranking table for the Ranking section."""
    head = ("| Rank | Property id | Usable roof area (sq ft) | Capacity (kW) | Annual energy (kWh) | Payback (years) |\n"
            "|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r.get('rank')} | {r.get('property_id')} [{eid}] | {_cite(r.get('usable_roof_area_sq_ft'), eid)} | "
        f"{_cite(r.get('capacity_kw'), eid)} | {_cite(r.get('annual_energy_kwh'), eid)} | {_cite(r.get('payback_years'), eid)} |"
        for r in ranked if r.get("rank") is not None and r["rank"] <= rows)
    return f"After your correction the ranking was re-run [{eid}].\n\n{head}{body}\n\n"


def _apply_solar_correction(run: StudyRun, original: Measurement, corrected: Measurement, measurement_id: str,
                            fields: list[str], note: str | None, when: str) -> CorrectionResult:
    """Correction for a study with solar_results: re-estimate, re-rank, re-screen and re-draft."""
    from backend.agent import upgrades
    from backend.agent.registry import ranking_ok

    study = run.study
    ledger = Ledger.from_entries(run.ledger.to_json())
    previous = next((e for e in reversed(ledger.all()) if e["tool"] == "rank_buildings" and ranking_ok(e["result"])), None)
    if previous is None:
        raise InvalidCorrection("This study has no solar ranking to re-run, so it cannot be corrected.")
    source = previous["source"]
    pid = original.property_id
    building = next((b for b in study.buildings if b.id == pid), None)
    place = building.name if building and building.name else f"property {pid}"
    summarize = lambda m, f: len(m.detections or []) if f == "detections" else getattr(m, f)  # noqa: E731
    before_vals = {f: summarize(original, f) for f in fields}
    after_vals = {f: summarize(corrected, f) for f in fields}
    correction_entry = ledger.add("correct_measurement", {
        "measurement_id": measurement_id, "property_id": pid, "property_name": place,
        "before": before_vals, "after": after_vals, "note": note,
    }, corrected.model_dump(mode="json"), "mock" if study.is_mock else "real")

    old_ranked = [SolarResult.model_validate(r) for r in previous["result"]["ranked"]]
    old_estimate = next((r for r in old_ranked if r.property_id == pid), None)
    if source == "mock":
        from backend.tools import mock as implementation
        new_estimate = implementation.estimate_solar(building, corrected) if building else old_estimate
    else:
        from backend.tools import solar as implementation
        new_estimate = rescale_solar(old_estimate, corrected.roof_area_sq_ft) if old_estimate else None
    estimates = [r.model_copy(update={"rank": None}) for r in old_ranked if r.property_id != pid]
    if new_estimate is not None:
        ledger.add("solar_payback", {"property_id": pid, "roof_area_sq_ft": corrected.roof_area_sq_ft,
                                                   "note": f"after correction of {measurement_id}"},
                                 new_estimate.model_dump(mode="json"), source)
        estimates.append(new_estimate)
    ranked = implementation.rank_buildings(estimates)
    rank_entry = ledger.add("rank_buildings", {"estimates": len(estimates), "note": f"after correction of {measurement_id}"},
                            {"ranked": [r.model_dump(mode="json") for r in ranked], "count": len(ranked),
                             "estimated": len(estimates), "skipped_no_roof": previous["result"].get("skipped_no_roof", 0)},
                            source)

    screened = _latest(ledger, "screen_upgrades", pid)
    upgrade_entry = None
    new_by_id = {r.property_id: r for r in ranked}
    if screened is not None and building is not None:
        interior = _latest(ledger, "interior_inspect", pid)
        upgrade_entry = ledger.add("screen_upgrades", {"property_id": pid, "note": f"after correction of {measurement_id}"},
                                   upgrades.screen(building.model_dump(mode="json"), corrected.model_dump(mode="json"),
                                                   new_by_id.get(pid).model_dump(mode="json") if new_by_id.get(pid) else None,
                                                   interior["result"] if interior else None),
                                   "mock" if study.is_mock else "real")

    drafts = list(study.outreach_drafts or [])
    for i, draft in enumerate(drafts):
        if draft.property_id == pid and building is not None and new_by_id.get(pid) is not None:
            estimate = new_by_id[pid]
            if source == "mock":
                from backend.tools import mock
                redrafted = mock.draft_outreach(building, estimate)
            else:
                from backend.tools import outreach
                subject, body = outreach.template_draft(building, estimate)
                redrafted = draft.model_copy(update={"subject": subject, "body_markdown": body})
            caveats = list(dict.fromkeys([*(draft.caveats or []),
                                          "Redrafted from the fixed template after a measurement correction; review before use."]))
            drafts[i] = redrafted.model_copy(update={"caveats": caveats})
            ledger.add("draft_email", {"property_id": pid, "note": f"after correction of {measurement_id}"},
                       drafts[i].model_dump(mode="json"), source)

    old_e, new_e, cor_e = previous["id"], rank_entry["id"], correction_entry["id"]
    old_row = next((r for r in previous["result"]["ranked"] if r.get("property_id") == pid), {})
    new_row = next((r.model_dump(mode="json") for r in ranked if r.property_id == pid), {})
    old_top = next((r for r in previous["result"]["ranked"] if r.get("rank") == 1), {}).get("property_id")
    new_top = next((r.property_id for r in ranked if r.rank == 1), None)
    keys = {"capacity_kw": "capacity (kW)", "annual_energy_kwh": "annual energy (kWh)",
            "payback_years": "simple payback (years)", "rank": "rank"}
    changes = [{"label": f"{f} ({place})", "before": before_vals[f], "after": after_vals[f],
                "direction": _direction(before_vals[f], after_vals[f])} for f in fields]
    changes += [{"label": f"{label} ({place})", "before": old_row.get(k), "after": new_row.get(k),
                 "direction": _direction(old_row.get(k), new_row.get(k))} for k, label in keys.items()]
    if old_top != new_top:
        conclusion = (f"The conclusion changed: the top candidate is now property {new_top} [{new_e}] "
                      f"instead of property {old_top} [{old_e}].")
    else:
        conclusion = "The conclusion did not change: the top-ranked candidate is the same."
    if source == "mock":
        conclusion += " Demo studies use fixed illustrative solar numbers, so a correction cannot move them."
    field_lines = "\n".join(
        f"- {f} for {place} (property {pid} [{cor_e}]): {_cite(before_vals[f], cor_e)} before, "
        f"{_cite(after_vals[f], cor_e)} after." for f in fields)
    solar_lines = "\n".join(f"- {label}: {_cite(old_row.get(k), old_e)} before, {_cite(new_row.get(k), new_e)} after."
                             for k, label in keys.items())
    block = (f"{CORRECTION_HEADING}\n\n{field_lines}\n{solar_lines}\n\n{conclusion}\n"
             + ("\nThe owner email for this building was redrafted from the fixed template.\n"
                if any(d.property_id == pid for d in drafts) else "")
             + (f"\nYour note is saved with the correction [{cor_e}].\n" if note else ""))

    brief = _CORRECTION_SECTION.sub("\n\n", study.brief_markdown).rstrip()
    table_rows = max(1, sum(1 for r in previous["result"]["ranked"] if (r.get("rank") or 99) <= 10))
    ranking_json = [r.model_dump(mode="json") for r in ranked]
    if _RANKING_SECTION.search(brief):
        brief = _RANKING_SECTION.sub(lambda m: m.group(1) + _ranking_section(ranking_json, new_e, table_rows), brief, count=1)
    if conclusion.startswith("The conclusion changed") and STALE_ANSWER_NOTE not in brief:
        brief = _ANSWER_HEADING.sub(lambda m: m.group(0) + STALE_ANSWER_NOTE, brief, count=1)
    brief = brief.rstrip() + "\n\n" + block

    problems = verify_brief(brief, ledger)
    baseline = set(verify_brief(study.brief_markdown, run.ledger))
    if [p for p in problems if p not in baseline]:
        raise UnverifiableCorrection("The corrected brief failed the citation check; nothing was saved.")

    new_id, n = next_corrected_id(run)
    caveat = (f"User correction {n}: {', '.join(fields)} of {place} (measurement {measurement_id}) was changed "
              "by hand; solar estimate, ranking, upgrade screening and email draft were recomputed.")
    if note:
        caveat += f" Note: {note}"
    new_study = study.model_copy(deep=True, update={
        "id": new_id, "brief_markdown": brief, "solar_results": ranked, "outreach_drafts": drafts or None,
        "caveats": [*study.caveats, caveat],
        "measurements": [corrected if m.id == measurement_id else m for m in study.measurements],
    })
    history = [*(run.meta.get("corrections") or []),
               {"n": n, "measurement_id": measurement_id, "fields": fields, "at": when,
                "correction_entry": cor_e, "regression_entry": new_e}]
    meta = {"original_id": run.meta.get("original_id") or study.id, "corrected_from": study.id,
            "correction_count": n, "corrections": history}
    result = CorrectionResult(run=StudyRun(study=new_study, ledger=ledger, meta=meta), corrected_measurement=corrected,
                              changes=changes, conclusion=conclusion, correction_entry=correction_entry,
                              regression_entry=rank_entry, previous_regression_entry=previous, problems=problems)
    result.upgrade_entry = upgrade_entry  # type: ignore[attr-defined]
    return result
