"""LangChain tools for the CityLab agent, each recording its result to a Ledger.

TOOLS_MODE selects implementations: "auto" (default) tries the real function in
backend/tools/ and falls back to backend/tools/mock.py only when the real one
raises NotImplementedError; "real" never falls back; "mock" is fully synthetic.
Agent tool names and argument names follow docs/CONTRACTS.md.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import TypeAdapter, ValidationError

from backend import config  # also loads .env so TOOLS_MODE is visible
from backend.agent.evidence import Ledger, Source
from backend.cache import CacheMiss
from backend.schemas import (
    AerialImage, Building, Measurement, OutreachDraft, SolarResult, StatsResult, VexcelPropertyInfo,
)

log = logging.getLogger(__name__)

MODES = ("auto", "real", "mock")
INTENTS = ("research", "retrofit", "property")
MULTIFAMILY = 11
# CoStar records these only for some property types (see docs/CONTRACTS.md mappings).
TYPE_LIMITED_FIELDS = {
    "total_num_units": "CoStar records unit counts only for multifamily properties",
    "max_rent": "asking rent comes from CoStar apartment listings, so it exists only for multifamily properties",
    "vacancy_percent": "CoStar records vacancy (propertyApartment.vacancyPercent) only for multifamily properties",
}
MAX_TOP_N = 10
DEFAULT_N = 40
MAX_N = 60

# agent tool name -> (backend.tools module, canonical function, return type)
BINDINGS: dict[str, tuple[str, str, Any]] = {
    "sample_buildings": ("costar_data", "find_buildings", list[Building]),
    "get_aerial": ("vexcel", "get_aerial", AerialImage),
    "vexcel_detect": ("vexcel", "get_property_info", VexcelPropertyInfo),
    "vision_inspect": ("vision", "measure_building", Measurement),
    "run_regression": ("stats", "run_study", StatsResult),
    "solar_payback": ("solar", "estimate_solar", SolarResult),
    "rank_buildings": ("solar", "rank_buildings", list[SolarResult]),
    "draft_email": ("outreach", "draft_outreach", OutreachDraft),
}

# Large provider payloads stay in the ledger but are not sent back to the model.
_HIDDEN_FROM_MODEL = ("raw_response",)
_SENSITIVE = re.compile(r"key|secret|token|password|authorization", re.I)


def resolve_mode(mode: str | None = None) -> str:
    """Return a valid tool mode from the argument or TOOLS_MODE (blank means auto)."""
    value = (mode if mode is not None else os.getenv("TOOLS_MODE", "")).strip().lower() or "auto"
    if value not in MODES:
        raise ValueError(f"TOOLS_MODE must be one of {', '.join(MODES)}.")
    return value


def redact(value: Any) -> Any:
    """Remove credential-like fields and environment secret values from printable data."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if _SENSITIVE.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        secrets = [v for k, v in os.environ.items() if v and len(v) >= 6 and _SENSITIVE.search(k)]
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
    return value


def _compact(args: dict[str, Any]) -> dict[str, Any]:
    """Replace list/dict argument values with short placeholders and redact the rest."""
    return redact({k: (f"<{len(v)} items>" if isinstance(v, (list, dict)) else v) for k, v in args.items()})


def short_args(args: dict[str, Any], width: int = 120) -> str:
    """Return a one-line, redacted summary of tool arguments for console output."""
    text = json.dumps(_compact(args), default=str, ensure_ascii=False)
    return text if len(text) <= width else text[: width - 3] + "..."


def _jsonable(value: Any) -> Any:
    """Dump pydantic models (and lists of them) to JSON-safe data."""
    return TypeAdapter(Any).dump_python(value, mode="json")


def vision_cache_path(property_id: int) -> Path:
    """Return data/cache/vision/m-<property_id>.json (the format study.py and
    scripts/accuracy_check.py use)."""
    return config.CACHE_DIR / "vision" / f"m-{property_id}.json"


def _cacheable(measurement: Measurement, property_id: int) -> bool:
    """A measurement is reusable only if it is for this property, not an error and not mock."""
    observations = measurement.observations or {}
    return (measurement.property_id == property_id and not observations.get("error")
            and not observations.get("is_mock"))


def load_cached_measurement(property_id: int) -> Measurement | None:
    """Return a valid cached real measurement for this property, or None."""
    try:
        measurement = Measurement.model_validate_json(vision_cache_path(property_id).read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None
    return measurement if _cacheable(measurement, property_id) else None


def save_cached_measurement(measurement: Measurement) -> bool:
    """Save a successful real measurement; errors and mocks are never cached."""
    if measurement.property_id is None or not _cacheable(measurement, measurement.property_id):
        return False
    path = vision_cache_path(measurement.property_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(measurement.model_dump_json(), encoding="utf-8")
    return True


# Complete-case sampling: stats.run_study drops rows missing any of these controls (and
# building classes outside A/B/C/F), so the sample keeps only buildings that can enter
# the regression. sample_buildings scans limit x SCAN_FACTOR candidates (by id).
COMPLETE_FIELDS = ("total_num_units", "year_built", "building_class")
_VALID_CLASSES = (1, 2, 3, 4)
SCAN_FACTOR, MIN_SCAN, MAX_SCAN = 5, 50, 500


def _is_complete(building: Building, required: list[str]) -> bool:
    """True when every required field is known (and the class is A, B, C or F)."""
    if any(getattr(building, f, None) is None for f in required):
        return False
    return "building_class" not in required or building.building_class in _VALID_CLASSES


def infer_study_fields(question: str | None) -> tuple[str | None, str | None]:
    """Return (outcome, predictor) as stats.run_study infers them from the question words."""
    if not question:
        return None, None
    from backend.tools import stats  # read-only use of the tool's own keyword tables

    return stats._infer(question, stats._OUTCOME_WORDS), stats._infer(question, stats._PREDICTOR_WORDS)


def regression_ok(result: dict[str, Any]) -> bool:
    """True when a run_regression result has its predictor's coefficient and a sample size."""
    predictor = result.get("predictor")
    coefficients = result.get("coefficients") or {}
    return ("error" not in result and bool(predictor) and coefficients.get(predictor) is not None
            and bool(result.get("sample_size")))


def ranking_ok(result: dict[str, Any]) -> bool:
    """True when a rank_buildings result ranked at least one building."""
    return "error" not in result and any(r.get("rank") is not None for r in result.get("ranked") or [])


def draft_ok(result: dict[str, Any]) -> bool:
    """True when a draft_email result contains an email body."""
    return "error" not in result and bool((result.get("body_markdown") or "").strip())


_RETROFIT_WORDS = re.compile(r"\b(solar|rooftop|retrofit)|best candidates", re.I)
_STREET_ADDRESS = re.compile(
    r"\b\d{1,6}\s+(?:[nsew]\.?\s+)?[a-z0-9]+(?:\s+[a-z0-9]+){0,3}\s+"
    r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|pkwy|parkway|hwy|highway|"
    r"ct|court|pl|place|fwy|freeway|cir|circle|ter|terrace)\b", re.I)


def is_property_query(question: str) -> bool:
    """True for a single-property request: a CoStar property id or a street address."""
    from backend.agent.property_lookup import parse_property_id

    return parse_property_id(question) is not None or bool(_STREET_ADDRESS.search(question or ""))


def detect_intent(question: str, intent: str | None = "auto") -> str:
    """Resolve intent: explicit research/retrofit/property, or auto from the question.

    auto: a property id or street address means one building ("property"); solar,
    rooftop, retrofit or "best candidates" means "retrofit"; anything else "research".
    """
    value = (intent or "auto").strip().lower()
    if value == "auto":
        if is_property_query(question):
            return "property"
        return "retrofit" if _RETROFIT_WORDS.search(question) else "research"
    if value not in INTENTS:
        raise ValueError("intent must be auto, research, retrofit or property.")
    return value


def parse_top_n(question: str, default: int = MAX_TOP_N) -> int:
    """Return the requested number of candidates (first non-year integer), capped at 10."""
    for match in re.finditer(r"(?<![\d.])(\d{1,4})(?![\d.])", question):
        value = int(match.group(1))
        if not 1900 <= value <= 2100 and value > 0:
            return min(value, MAX_TOP_N)
    return default


def _as_result(name: str, value: Any) -> dict[str, Any]:
    """Validate a tool return against its schema and convert it to a JSON dict."""
    adapter = TypeAdapter(BINDINGS[name][2])
    data = adapter.dump_python(adapter.validate_python(value), mode="json")
    if isinstance(data, list):
        key = {"sample_buildings": "buildings", "rank_buildings": "ranked"}.get(name, "items")
        return {key: data, "count": len(data)}
    return data


class _Runner:
    """Choose real/mock implementations for one run and record every call."""

    def __init__(self, mode: str, ledger: Ledger, on_call: Callable[[dict[str, Any]], None] | None) -> None:
        """Bind the tool mode, the run's ledger and an optional print hook."""
        self.mode = mode
        self.ledger = ledger
        self.on_call = on_call
        self._warned: set[str] = set()

    def _function(self, name: str, source: Source) -> Callable[..., Any]:
        """Look up the function at call time so tests can patch backend.tools modules."""
        module_name, function_name, _ = BINDINGS[name]
        module = importlib.import_module(f"backend.tools.{'mock' if source == 'mock' else module_name}")
        return getattr(module, function_name)

    def _call(self, name: str, kwargs: dict[str, Any]) -> tuple[Any, Source]:
        """Run the selected implementation, applying the auto-mode fallback rule."""
        if self.mode == "mock":
            return self._function(name, "mock")(**kwargs), "mock"
        try:
            return self._function(name, "real")(**kwargs), "real"
        except NotImplementedError:
            if self.mode == "real":
                raise
            if name not in self._warned:
                self._warned.add(name)
                log.warning("REAL TOOL NOT READY, USING MOCK: %s", name)
            return self._function(name, "mock")(**kwargs), "mock"

    def run(self, name: str, kwargs: dict[str, Any], record_args: dict[str, Any]) -> dict[str, Any]:
        """Call a tool, record it to the ledger, and return the result with its evidence id."""
        try:
            value, source = self._call(name, kwargs)
            result = _as_result(name, value)
        except NotImplementedError:
            raise
        except Exception as exc:
            self._fail(name, record_args, exc)
        return self._record(name, record_args, result, source)

    def _record(self, name: str, args: dict[str, Any], result: dict[str, Any], source: Source,
                cached: bool = False) -> dict[str, Any]:
        """Add a successful result to the ledger and return what the model sees."""
        entry = self.ledger.add(name, args, result, source, cached=cached)
        if self.on_call:
            self.on_call(entry)
        visible = {k: v for k, v in result.items() if k not in _HIDDEN_FROM_MODEL}
        return {**visible, "evidence_id": entry["id"], "source": source, **({"cached": True} if cached else {})}

    def _fail(self, name: str, args: dict[str, Any], exc: Exception) -> None:
        """Record a failed call and raise a ToolException the model can read."""
        message = str(redact(f"{type(exc).__name__}: {exc}"))[:300]
        entry = self.ledger.add(name, args, {"error": message}, "mock" if self.mode == "mock" else "real")
        if self.on_call:
            self.on_call(entry)
        raise ToolException(f"{name} failed ({entry['id']}): {message}") from None

    def sample(self, market_id: int, property_type: int, limit: int, required: list[str]) -> dict[str, Any]:
        """sample_buildings restricted to complete cases: the first *limit* buildings (by id)
        with every *required* field known (research: regression controls and outcome;
        retrofit: none beyond the geocoding find_buildings already enforces)."""
        scan = min(max(limit * SCAN_FACTOR, MIN_SCAN), MAX_SCAN)
        args = {"market_id": market_id, "property_type": property_type, "limit": limit,
                "scanned_limit": scan, "required_fields": required}
        try:
            value, source = self._call("sample_buildings", {"market_id": market_id,
                                                            "property_type": property_type, "limit": scan})
            candidates = TypeAdapter(list[Building]).validate_python(value)
        except NotImplementedError:
            raise
        except Exception as exc:
            self._fail("sample_buildings", args, exc)
        if property_type != MULTIFAMILY:
            required = [f for f in required if f != "total_num_units"]  # units exist only for multifamily
            args["required_fields"] = required
        complete = [b for b in candidates if _is_complete(b, required)]
        result = _as_result("sample_buildings", complete[:limit])
        missing = {f: sum(getattr(b, f, None) is None for b in candidates) for f in required}
        result.update({"candidates_scanned": len(candidates), "excluded_incomplete": len(candidates) - len(complete),
                       "missing_by_field": {f: n for f, n in missing.items() if n}})
        blocking = [f for f, n in missing.items() if candidates and n == len(candidates)]
        if not complete and blocking:
            why = "; ".join(TYPE_LIMITED_FIELDS.get(f, f"{f} is unknown for every candidate") for f in blocking)
            result["unavailable_reason"] = (f"No complete cases: {', '.join(blocking)} is missing for all "
                                            f"{len(candidates)} candidate buildings ({why}).")
        return self._record("sample_buildings", args, result, source)

    def vision(self, property_id: int, image_path: str) -> dict[str, Any]:
        """vision_inspect: resolve the image, then use the vision cache or the model.

        An empty image_path first calls get_aerial for the sampled property (its own
        ledger entry); if no image can be fetched, a ledger error is recorded and the
        call fails. "" is never passed to vision. A valid cached real measurement
        (data/cache/vision/m-<id>.json) is then returned (source "real", cached) without
        a model call; with USE_CACHE_ONLY a miss is an error. Only successful real
        measurements (no observations.error / is_mock) are saved.
        """
        path = image_path.strip() if image_path else ""
        if not path:
            path = self.image_path_for(property_id)
        args = {"image_path": path, "property_id": property_id}
        if self.mode != "mock":
            cached = load_cached_measurement(property_id)
            if cached is not None:
                return self._record("vision_inspect", args, _as_result("vision_inspect", cached), "real", cached=True)
            if config.USE_CACHE_ONLY:
                self._fail("vision_inspect", args,
                           CacheMiss(f"no cached vision measurement for {property_id} and USE_CACHE_ONLY=true"))
        out = self.run("vision_inspect", args, args)
        if out["source"] == "real":
            save_cached_measurement(Measurement.model_validate(self.ledger.get(out["evidence_id"])["result"]))
        return out

    def regression(self, question: str, model_rows: int) -> dict[str, Any]:
        """run_regression over exactly the inspected properties, joined by property_id.

        Idempotent per run: a successful regression already in the ledger for the same
        outcome and predictor is returned (same evidence id) instead of running again.
        """
        outcome, predictor = infer_study_fields(question)
        for entry in reversed(self.ledger.all()):
            result = entry["result"]
            if (entry["tool"] == "run_regression" and regression_ok(result)
                    and (outcome, predictor) == (result.get("outcome"), result.get("predictor"))):
                log.info("run_regression reused %s for outcome=%s predictor=%s", entry["id"], outcome, predictor)
                visible = {k: v for k, v in result.items() if k not in _HIDDEN_FROM_MODEL}
                return {**visible, "evidence_id": entry["id"], "source": entry["source"], "reused": True}

        measured = {m.property_id: m for m in self.measurements()}
        sampled = self.sampled_buildings()
        buildings = [b for b in sampled if b.id in measured]
        measurements = [measured[b.id] for b in buildings]
        record = {"question": question, "property_ids": [b.id for b in buildings],
                  "sampled_buildings": len(sampled), "inspected_properties": len(measured),
                  "joined_rows": len(buildings), "measurement_count": len(measurements),
                  "model_supplied_rows_ignored": model_rows}
        log.info("run_regression joined %d rows (%d sampled, %d inspected)", len(buildings), len(sampled), len(measured))
        if not buildings:
            self._fail("run_regression", record,
                       ValueError("no inspected properties match the sampled buildings; run sampler and inspector first"))
        return self.run("run_regression", {"question": question, "buildings": buildings,
                                           "measurements": measurements}, record)

    # ---- retrofit: solar estimate, ranking, outreach drafts ----------------------------

    def _building(self, property_id: int) -> Building | None:
        """Return the sampled Building with this id, if any."""
        return next((b for b in self.sampled_buildings() if b.id == property_id), None)

    def solar(self, property_id: int) -> dict[str, Any]:
        """solar_payback for one inspected building, from ledger records only.

        Math first, model last: estimate_solar uses a measured roof when the measurement
        carries one (a Vexcel parcel footprint, or the aerial image) and otherwise falls
        back to CoStar building-footprint fields, recording which in roof_area_source.
        Only a building with no roof area from any source is recorded as skipped.
        """
        building = self._building(property_id)
        measurement = next((m for m in self.measurements() if m.property_id == property_id), None)
        args: dict[str, Any] = {"property_id": property_id}
        if building is None or measurement is None:
            self._fail("solar_payback", args, ValueError(
                f"property {property_id} needs a sample_buildings record and a successful vision_inspect first"))
        args["roof_area_sq_ft"] = measurement.roof_area_sq_ft
        try:
            value, source = self._call("solar_payback", {"building": building, "vision_result": measurement})
            result = _as_result("solar_payback", value)
        except NotImplementedError:
            raise
        except Exception as exc:
            self._fail("solar_payback", args, exc)
        args["roof_area_source"] = (result.get("assumptions") or {}).get("roof_area_source")
        if result.get("usable_roof_area_sq_ft") is None and result.get("capacity_kw") is None:
            return self._record("solar_payback", args, {
                "property_id": property_id, "skipped": True,
                "reason": "no roof area from any source (aerial measurement, Vexcel footprint or CoStar fields)",
                "caveats": result.get("caveats") or [],
            }, source)
        return self._record("solar_payback", args, result, source)

    def solar_estimates(self) -> list[SolarResult]:
        """Latest successful (non-skipped) solar_payback result per property."""
        latest: dict[Any, SolarResult] = {}
        for entry in self.ledger.all():
            result = entry["result"]
            if entry["tool"] == "solar_payback" and "error" not in result and not result.get("skipped"):
                estimate = SolarResult.model_validate(result)
                latest[estimate.property_id] = estimate
        return list(latest.values())

    def rank(self) -> dict[str, Any]:
        """rank_buildings once over every solar estimate in the ledger (payback order)."""
        estimates = self.solar_estimates()
        skipped = {e["result"]["property_id"] for e in self.ledger.all()
                   if e["tool"] == "solar_payback" and e["result"].get("skipped")}
        args = {"estimates": len(estimates), "skipped_no_roof": len(skipped)}
        if not estimates:
            self._fail("rank_buildings", args, ValueError("no solar estimates to rank; run solar_payback first"))
        try:
            value, source = self._call("rank_buildings", {"estimates": estimates})
            result = _as_result("rank_buildings", value)
        except NotImplementedError:
            raise
        except Exception as exc:
            self._fail("rank_buildings", args, exc)
        result.update({"estimated": len(estimates), "skipped_no_roof": len(skipped)})
        return self._record("rank_buildings", args, result, source)

    def latest_ranking(self) -> list[SolarResult]:
        """SolarResults from the latest successful rank_buildings entry, in rank order."""
        for entry in reversed(self.ledger.all()):
            if entry["tool"] == "rank_buildings" and ranking_ok(entry["result"]):
                return [SolarResult.model_validate(r) for r in entry["result"]["ranked"]]
        return []

    def draft(self, property_id: int, top_n: int) -> dict[str, Any]:
        """draft_email for a top-N ranked building; never sends anything."""
        args = {"property_id": property_id, "top_n": top_n}
        ranking = self.latest_ranking()
        estimate = next((r for r in ranking if r.property_id == property_id), None)
        building = self._building(property_id)
        if estimate is None or building is None:
            self._fail("draft_email", args, ValueError(f"property {property_id} is not in the latest ranking"))
        if estimate.rank is None or estimate.rank > top_n:
            self._fail("draft_email", args, ValueError(
                f"property {property_id} has rank {estimate.rank}; drafts are only for the top {top_n}"))
        args["rank"] = estimate.rank
        return self.run("draft_email", {"building": building, "solar_estimate": estimate}, args)

    # ---- ledger lookups used to fill omitted arguments --------------------------------

    def sampled_buildings(self) -> list[Building]:
        """Buildings from the latest successful sample_buildings (or lookup_property) entry."""
        for entry in reversed(self.ledger.all()):
            if entry["tool"] == "sample_buildings" and "buildings" in entry["result"]:
                return [Building.model_validate(b) for b in entry["result"]["buildings"]]
            if entry["tool"] == "lookup_property" and "building" in entry["result"]:
                return [Building.model_validate(entry["result"]["building"])]
        return []

    # ---- single property, interior scan and upgrade screening -------------------------

    def lookup(self, query: str) -> dict[str, Any]:
        """lookup_property: one CoStar property by id or street address, with owners and tenants."""
        from backend.agent import property_lookup

        args = {"query": query}
        source: Source = "mock" if self.mode == "mock" else "real"
        try:
            result = property_lookup.lookup(query, self.mode)
        except Exception as exc:  # LookupError carries a user-facing message
            self._fail("lookup_property", args, exc)
        return self._record("lookup_property", args, result, source)

    def interior(self, property_id: int) -> dict[str, Any]:
        """interior_inspect: Matterport spatial metadata when a scan exists; skipped otherwise."""
        args: dict[str, Any] = {"property_id": property_id}
        if self.mode == "mock":
            return self._record("interior_inspect", args, {
                "property_id": property_id, "available": False,
                "reason": "the synthetic demo data has no Matterport scans"}, "mock")
        from backend.tools import matterport

        model_id = matterport.model_id_for(property_id)
        if not model_id:
            return self._record("interior_inspect", args, {
                "property_id": property_id, "available": False,
                "reason": "no Matterport scan is linked to this property in commercial-properties.csv"}, "real")
        args["model_id"] = model_id
        try:
            model = matterport.get_model(model_id)
        except Exception as exc:
            self._fail("interior_inspect", args, exc)
        return self._record("interior_inspect", args, {
            "property_id": property_id, "available": True, "model_id": model_id, "summary": model.metadata,
            "not_observed": ["lighting type", "window characteristics", "visible HVAC equipment"],
            "not_observed_reason": ("the Matterport model API used here returns measured floors and rooms but no "
                                    "interior images, so these were not assessed"),
        }, "real")

    def upgrades(self, property_id: int) -> dict[str, Any]:
        """screen_upgrades: rooftop solar, solar carport, LED, window film, HVAC for one building."""
        from backend.agent import upgrades

        args: dict[str, Any] = {"property_id": property_id}
        building = self._building(property_id)
        if building is None:
            self._fail("screen_upgrades", args, ValueError(f"property {property_id} was not sampled or looked up"))
        latest = lambda tool: next((e["result"] for e in reversed(self.ledger.all()) if e["tool"] == tool  # noqa: E731
                                    and e["result"].get("property_id") == property_id and "error" not in e["result"]), None)
        result = upgrades.screen(building.model_dump(mode="json"), latest("vision_inspect"),
                                 latest("solar_payback"), latest("interior_inspect"))
        return self._record("screen_upgrades", args, result, "mock" if self.mode == "mock" else "real")

    def measurements(self) -> list[Measurement]:
        """Return the latest measurement per property from successful vision_inspect calls."""
        latest: dict[Any, Measurement] = {}
        for entry in self.ledger.all():
            if entry["tool"] == "vision_inspect" and "error" not in entry["result"]:
                m = Measurement.model_validate(entry["result"])
                latest[m.property_id] = m
        return list(latest.values())

    def image_path_for(self, property_id: int) -> str:
        """Fetch (and record) the aerial image for a sampled building; fail loudly without one."""
        building = next((b for b in self.sampled_buildings() if b.id == property_id), None)
        if building is None or building.latitude is None or building.longitude is None:
            self._fail("get_aerial", {"property_id": property_id},
                       ValueError(f"property {property_id} is not a geocoded building from sample_buildings"))
        args = {"lat": building.latitude, "lon": building.longitude, "property_id": property_id}
        image = self.run("get_aerial", args, args)
        path = str(image.get("image_path") or "").strip()
        if not path:
            self._fail("vision_inspect", {"image_path": "", "property_id": property_id},
                       ValueError(f"get_aerial ({image['evidence_id']}) returned no image for {property_id}"))
        return path


def get_tools(
    mode: str | None, ledger: Ledger, n: int = DEFAULT_N,
    on_call: Callable[[dict[str, Any]], None] | None = None, question: str | None = None,
    intent: str = "research", top_n: int = MAX_TOP_N,
) -> list[BaseTool]:
    """Return the CityLab LangChain tools bound to one run's ledger.

    Tools: sample_buildings, vexcel_detect, vision_inspect, run_regression,
    solar_payback, rank_buildings, draft_email, read_evidence and list_evidence. *n* is the
    default sample size (capped at MAX_N). *on_call* receives each new entry.
    *question* (the run's research question) tells sampling which outcome must be known.
    *intent* "research" samples complete regression cases; "retrofit" needs none. *top_n*
    (capped at 10) limits which ranked buildings draft_email accepts.
    """
    runner = _Runner(resolve_mode(mode), ledger, on_call)
    default_n = max(1, min(int(n), MAX_N))
    outcome, _ = infer_study_fields(question)
    required = [*COMPLETE_FIELDS, *([outcome] if outcome else [])] if intent == "research" else []
    top_n = max(1, min(int(top_n), MAX_TOP_N))

    def sample_buildings(market_id: int, property_type: int, limit: int = default_n) -> dict[str, Any]:
        """Sample buildings from CoStar for a market (Houston 14, Los Angeles 20) and
        property type (multifamily 11, office 5, industrial 2, retail 6). limit is capped at
        60. For research, only complete cases are returned (year built, class, the outcome,
        and units for multifamily); unavailable_reason explains an empty sample."""
        return runner.sample(market_id, property_type, max(1, min(limit, MAX_N)), required)

    def vexcel_detect(lat: float, lon: float) -> dict[str, Any]:
        """Get Vexcel's precomputed parcel and structure evidence for a building's location."""
        args = {"lat": lat, "lon": lon}
        return runner.run("vexcel_detect", args, args)

    def vision_inspect(property_id: int, image_path: str = "") -> dict[str, Any]:
        """Measure one sampled building from its aerial image (parking spaces, green
        cover, paved surface, roof area). Leave image_path empty to fetch the image."""
        return runner.vision(property_id, image_path)

    def run_regression(
        question: str, buildings: list[Building] | None = None, measurements: list[Measurement] | None = None,
    ) -> dict[str, Any]:
        """Fit the study regression. Name the outcome and predictor in question. Do not
        pass buildings or measurements: the tool always uses this run's sampled buildings
        and inspected measurements from the ledger, joined by property_id (any values
        passed are ignored). Calling it again for the same outcome and predictor returns
        the first successful result."""
        return runner.regression(question, len(buildings or []) + len(measurements or []))

    def solar_payback(property_id: int) -> dict[str, Any]:
        """Estimate rooftop solar (usable roof, capacity kW, annual kWh, simple payback years)
        for ONE inspected building, using its image-measured roof area from the ledger
        (never floor area). Buildings with no measured roof are recorded as skipped."""
        return runner.solar(property_id)

    def rank_buildings() -> dict[str, Any]:
        """Rank every solar_payback estimate in the ledger by payback (missing last). Call
        once, after all estimates. Reports how many buildings were skipped for no roof."""
        return runner.rank()

    def draft_email(property_id: int) -> dict[str, Any]:
        """Draft (never send) an owner email for one building in the latest ranking's top N."""
        return runner.draft(property_id, top_n)

    def lookup_property(query: str) -> dict[str, Any]:
        """Look up ONE CoStar property by its id (e.g. "23453") or street address (e.g.
        "4620 N Braeswood Blvd, Houston"). Returns the building, its owners, property
        manager and current tenants where CoStar records them, and lists unavailable fields."""
        return runner.lookup(query)

    def interior_inspect(property_id: int) -> dict[str, Any]:
        """Optional interior check: the property's Matterport scan (measured floors, rooms and
        ceiling heights) when one exists; recorded as skipped when there is no scan."""
        return runner.interior(property_id)

    def screen_upgrades(property_id: int) -> dict[str, Any]:
        """Screen rooftop solar, a solar carport, LED lighting, window film and HVAC for one
        building. Only upgrades with enough measured inputs are estimated; the others are
        listed with the missing input. Call after solar_payback for that building."""
        return runner.upgrades(property_id)

    def read_evidence(evidence_id: str) -> dict[str, Any]:
        """Return one ledger entry (id, tool, args, result, source) by id, e.g. "E3"."""
        entry = ledger.get(evidence_id)
        if entry is None:
            return {"error": f"{evidence_id} is not in the ledger"}
        result = {k: v for k, v in entry["result"].items() if k not in _HIDDEN_FROM_MODEL}
        return {**entry, "result": result}

    def list_evidence() -> list[dict[str, Any]]:
        """List every ledger entry with its id, tool, source, arguments and top-level values."""
        rows = []
        for entry in ledger.all():
            preview = {k: v for k, v in entry["result"].items()
                       if not isinstance(v, (dict, list)) or k == "error"}
            rows.append({"id": entry["id"], "tool": entry["tool"], "source": entry["source"],
                         "args": _compact(entry["args"]), "result_preview": preview})
        return rows

    functions = [sample_buildings, vexcel_detect, vision_inspect, run_regression,
                 solar_payback, rank_buildings, draft_email, lookup_property, interior_inspect,
                 screen_upgrades, read_evidence, list_evidence]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=(f.__doc__ or "").strip(),
                                         handle_tool_error=True) for f in functions]
