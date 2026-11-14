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


def _as_result(name: str, value: Any) -> dict[str, Any]:
    """Validate a tool return against its schema and convert it to a JSON dict."""
    adapter = TypeAdapter(BINDINGS[name][2])
    data = adapter.dump_python(adapter.validate_python(value), mode="json")
    if isinstance(data, list):
        key = "buildings" if name == "sample_buildings" else "items"
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

    def vision(self, property_id: int, image_path: str) -> dict[str, Any]:
        """vision_inspect with the data/cache/vision/m-<id>.json cache for real measurements.

        A valid cached measurement is returned (source "real", cached) without fetching
        imagery or calling the model. With USE_CACHE_ONLY a miss is an error. Only
        successful real measurements (no observations.error / is_mock) are saved.
        """
        if self.mode != "mock":
            cached = load_cached_measurement(property_id)
            if cached is not None:
                args = {"image_path": image_path, "property_id": property_id}
                return self._record("vision_inspect", args, _as_result("vision_inspect", cached), "real", cached=True)
            if config.USE_CACHE_ONLY:
                self._fail("vision_inspect", {"image_path": image_path, "property_id": property_id},
                           CacheMiss(f"no cached vision measurement for {property_id} and USE_CACHE_ONLY=true"))
        path = image_path or self.image_path_for(property_id)
        args = {"image_path": path, "property_id": property_id}
        out = self.run("vision_inspect", args, args)
        if out["source"] == "real":
            save_cached_measurement(Measurement.model_validate(self.ledger.get(out["evidence_id"])["result"]))
        return out

    # ---- ledger lookups used to fill omitted arguments --------------------------------

    def sampled_buildings(self) -> list[Building]:
        """Return the buildings from the latest successful sample_buildings entry."""
        for entry in reversed(self.ledger.all()):
            if entry["tool"] == "sample_buildings" and "buildings" in entry["result"]:
                return [Building.model_validate(b) for b in entry["result"]["buildings"]]
        return []

    def measurements(self) -> list[Measurement]:
        """Return the latest measurement per property from successful vision_inspect calls."""
        latest: dict[Any, Measurement] = {}
        for entry in self.ledger.all():
            if entry["tool"] == "vision_inspect" and "error" not in entry["result"]:
                m = Measurement.model_validate(entry["result"])
                latest[m.property_id] = m
        return list(latest.values())

    def image_path_for(self, property_id: int) -> str:
        """Fetch (and record) the aerial image for a sampled building."""
        building = next((b for b in self.sampled_buildings() if b.id == property_id), None)
        if building is None or building.latitude is None or building.longitude is None:
            raise ToolException(f"Property {property_id} is not a geocoded building from sample_buildings.")
        args = {"lat": building.latitude, "lon": building.longitude, "property_id": property_id}
        image = self.run("get_aerial", args, args)
        return str(image["image_path"])


def get_tools(
    mode: str | None, ledger: Ledger, n: int = DEFAULT_N,
    on_call: Callable[[dict[str, Any]], None] | None = None,
) -> list[BaseTool]:
    """Return the CityLab LangChain tools bound to one run's ledger.

    Tools: sample_buildings, vexcel_detect, vision_inspect, run_regression,
    solar_payback, draft_email, read_evidence and list_evidence. *n* is the
    default sample size (capped at MAX_N). *on_call* receives each new entry.
    """
    runner = _Runner(resolve_mode(mode), ledger, on_call)
    default_n = max(1, min(int(n), MAX_N))

    def sample_buildings(market_id: int, property_type: int, limit: int = default_n) -> dict[str, Any]:
        """Sample buildings from CoStar for a market (Houston 14, Los Angeles 20) and
        property type (apartments/multifamily 11). limit is capped at 60."""
        args = {"market_id": market_id, "property_type": property_type, "limit": max(1, min(limit, MAX_N))}
        return runner.run("sample_buildings", args, args)

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
        """Fit the study regression. Name the outcome and predictor in question. Omit
        buildings and measurements to use this run's sampled and inspected records."""
        b = buildings if buildings else runner.sampled_buildings()
        m = measurements if measurements is not None else runner.measurements()
        record = {"question": question, "property_ids": [x.id for x in b],
                  "measurement_count": len(m)}
        return runner.run("run_regression", {"question": question, "buildings": b, "measurements": m}, record)

    def solar_payback(building: Building, vision_result: Measurement) -> dict[str, Any]:
        """Estimate rooftop solar capacity, energy and simple payback for one building."""
        kwargs = {"building": building, "vision_result": vision_result}
        return runner.run("solar_payback", kwargs, _jsonable(kwargs))

    def draft_email(building: Building, solar_estimate: SolarResult) -> dict[str, Any]:
        """Draft (never send) an owner email about a rooftop solar estimate."""
        kwargs = {"building": building, "solar_estimate": solar_estimate}
        return runner.run("draft_email", kwargs, _jsonable(kwargs))

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
                 solar_payback, draft_email, read_evidence, list_evidence]
    return [StructuredTool.from_function(func=f, name=f.__name__, description=(f.__doc__ or "").strip(),
                                         handle_tool_error=True) for f in functions]
