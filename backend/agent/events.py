"""Shared agent event shapes for live streaming and the final AskResponse.

Events carry ids, tool names, sources and short redacted text only: never tool
arguments, raw provider (Vexcel) responses or secrets.
"""

from __future__ import annotations

from typing import Any

from backend.agent.registry import redact

REGRESSION_INCOMPLETE = "Study incomplete: regression did not produce a result"


def tool_call_event(entry: dict[str, Any]) -> dict[str, Any]:
    """A tool_call event for one ledger entry (no arguments or results)."""
    return {"event_type": "tool_call", "message": f"{entry['tool']} [{entry['source']}]",
            "timestamp": entry.get("ts"),
            "payload": {"evidence_id": entry["id"], "tool": entry["tool"], "source": entry["source"],
                        "cached": bool(entry.get("cached")), "failed": "error" in entry["result"],
                        # from the result, never the arguments: lets the UI show per-building provenance
                        "property_id": entry["result"].get("property_id") if isinstance(entry["result"], dict) else None}}


def handoff_event(handoff: dict[str, str]) -> dict[str, Any]:
    """A handoff event for one subagent delegation (task text is redacted and short)."""
    subagent = str(redact(handoff.get("subagent", "")))
    return {"event_type": "handoff", "message": f"hand-off -> {subagent}", "timestamp": handoff.get("ts") or None,
            "payload": {"subagent": subagent, "task": str(redact(handoff.get("task", "")))[:100]}}


def citation_event(verified: bool, complete: bool, problems: list[str], timestamp: str | None = None) -> dict[str, Any]:
    """The final citation_check event: PASS, FAIL, or FAIL (study incomplete) with a reason."""
    payload: dict[str, Any] = {"verified": verified, "problems": [str(redact(p)) for p in problems]}
    if not complete:
        message = "FAIL (study incomplete)"
        payload["reason"] = REGRESSION_INCOMPLETE if REGRESSION_INCOMPLETE in problems else "study incomplete"
    else:
        message = "PASS" if verified else "FAIL"
    return {"event_type": "citation_check", "message": message, "timestamp": timestamp, "payload": payload}


def _num(value: Any) -> str:
    """Format a number compactly for a summary."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def evidence_summary(entry: dict[str, Any]) -> str:
    """A short human summary of one ledger entry, from its result only (never args/raw)."""
    tool, result = entry["tool"], entry["result"]
    if "error" in result:
        return str(redact(f"failed: {result['error']}"))[:120]
    pid = result.get("property_id")
    if tool == "sample_buildings":
        return f"{result.get('count', 0)} buildings sampled"
    if tool == "vexcel_detect":
        return f"Vexcel parcel attributes ({len(result.get('attributes') or {})} values, " \
               f"{len(result.get('structures') or [])} structures)"
    if tool == "get_aerial":
        return f"aerial image for property {pid}"
    if tool == "vision_inspect":
        fields = [(k, result.get(k)) for k in ("parking_space_count", "green_cover_percent",
                                               "paved_surface_percent", "roof_area_sq_ft")]
        known = ", ".join(f"{k} {_num(v)}" for k, v in fields if v is not None) or "no measurements"
        return f"{known} for property {pid}" + (" (cached)" if entry.get("cached") else "")
    if tool == "run_regression":
        predictor = result.get("predictor")
        coef = (result.get("coefficients") or {}).get(predictor)
        if coef is None:
            reason = (result.get("findings") or {}).get("error", "no estimate")
            return str(redact(f"no regression result: {reason}"))[:120]
        low, high = (result.get("confidence_intervals") or {}).get(predictor, (None, None))
        p = (result.get("p_values") or {}).get(predictor)
        return (f"{predictor} coefficient {_num(coef)} (95% CI {_num(low)} to {_num(high)}, p={_num(p)}), "
                f"n={result.get('sample_size')}")
    if tool == "solar_payback":
        if result.get("skipped"):
            return f"property {pid} skipped: {result.get('reason', 'no measured roof area')}"
        return f"payback {_num(result.get('payback_years'))} years, {_num(result.get('capacity_kw'))} kW " \
               f"for property {pid}"
    if tool == "rank_buildings":
        ranked = result.get("ranked") or []
        top = ranked[0].get("property_id") if ranked else None
        return f"{len(ranked)} buildings ranked; top property {top}"
    if tool == "draft_email":
        return str(redact(f"draft for property {pid}: {result.get('subject') or ''}"))[:120]
    if tool == "lookup_property":
        b = result.get("building") or {}
        owners = (result.get("owners") or {}).get("true_owner")
        return str(redact(f"property {b.get('id')} {b.get('name') or ''} matched by {result.get('matched_by')}"
                          + (f"; owner {owners[0]}" if owners else "")))[:120]
    if tool == "interior_inspect":
        if not result.get("available"):
            return f"no interior scan for property {pid}: {result.get('reason', '')}"[:120]
        s = result.get("summary") or {}
        return f"Matterport scan: {s.get('floor_count')} floors, {s.get('room_count')} rooms for property {pid}"
    if tool == "screen_upgrades":
        ups = result.get("upgrades") or []
        est = [u for u in ups if u.get("status") == "estimated"]
        best = min(est, key=lambda u: u.get("simple_payback_years") or 1e9, default=None)
        return (f"{len(est)} of {len(ups)} upgrades estimated for property {pid}"
                + (f"; best {best['name']} payback {_num(best.get('simple_payback_years'))} years" if best else ""))
    return tool


def evidence_list(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """[{id, tool, source, summary}] for every ledger entry."""
    return [{"id": e["id"], "tool": e["tool"], "source": e["source"], "summary": evidence_summary(e)}
            for e in entries]


# ---- planning and structured results (shared by the live stream and the saved response) ----

def plan_event(todos: list[dict[str, Any]], timestamp: str | None = None) -> dict[str, Any]:
    """The orchestrator's to-do list, from DeepAgents' write_todos tool (content + status)."""
    items = [{"content": str(redact(str(t.get("content", ""))))[:200], "status": str(t.get("status", "pending"))}
             for t in todos if isinstance(t, dict)]
    done = sum(t["status"] == "completed" for t in items)
    return {"event_type": "plan", "message": f"plan: {done}/{len(items)} steps done", "timestamp": timestamp,
            "payload": {"todos": items}}


def note_event(path: str, chars: int, timestamp: str | None = None) -> dict[str, Any]:
    """A note the orchestrator wrote to its DeepAgents scratch filesystem (path and size only)."""
    return {"event_type": "note", "message": f"note saved: {redact(path)}", "timestamp": timestamp,
            "payload": {"path": str(redact(path))[:120], "chars": int(chars)}}


DETAIL_EVENTS = {"lookup_property": "property_details", "interior_inspect": "interior", "screen_upgrades": "upgrade_screening"}


def detail_event(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Structured results the UI renders (property details, interior scan, upgrade screening).

    The StudyResult contract has no fields for these, so they travel as event payloads
    built from the ledger entry's result (never its arguments).
    """
    kind = DETAIL_EVENTS.get(entry["tool"])
    if kind is None or "error" in entry["result"]:
        return None
    result = dict(entry["result"])
    if kind == "property_details":
        result = {k: result.get(k) for k in ("matched_by", "owners", "tenants", "unavailable")}
        result["property_id"] = (entry["result"].get("building") or {}).get("id")
    return {"event_type": kind, "message": f"{kind.replace('_', ' ')} [{entry['source']}]",
            "timestamp": entry.get("ts"), "payload": {**redact(result), "evidence_id": entry["id"], "source": entry["source"]}}
