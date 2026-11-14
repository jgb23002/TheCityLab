"""Check the synthetic study and the documented signatures of every tool stub."""

import ast
import importlib
import inspect
import json
import re
import textwrap
import types
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

from backend import schemas
from backend.schemas import (
    AskRequest,
    AskResponse,
    CorrectRequest,
    CorrectResponse,
    MeasurementCorrection,
    StudyResult,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "docs" / "CONTRACTS.md"
MOCK_STUDY = ROOT / "data" / "showcase" / "mock_study.json"


def documented_tools() -> dict[str, ast.FunctionDef]:
    """Read authoritative Python signature fences without executing their bodies."""
    sections = re.findall(
        r"^### (backend\.tools\.[a-z_]+)\n\n```python\n(.*?)\n```",
        CONTRACTS.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL,
    )
    functions = {}
    for module, declarations in sections:
        for node in ast.parse(declarations).body:
            assert isinstance(node, ast.FunctionDef), module
            key = f"{module}.{node.name}"
            assert key not in functions, f"Duplicate documented tool: {key}"
            functions[key] = node
    return functions


def annotation_type(node: ast.expr) -> object:
    """Resolve only type expressions against schema classes and builtin types."""
    namespace = {"__builtins__": {}, "list": list, "str": str, "int": int, "float": float}
    namespace.update({
        name: value for name, value in vars(schemas).items()
        if inspect.isclass(value) and issubclass(value, BaseModel)
    })
    assert all(not isinstance(child, ast.Call) for child in ast.walk(node))
    return eval(compile(ast.Expression(body=node), "<contract annotation>", "eval"), namespace)


def test_mock_study_is_complete_and_explicitly_synthetic() -> None:
    """Validate all 12 linked buildings, Houston locations and the three-paragraph brief."""
    raw = json.loads(MOCK_STUDY.read_text(encoding="utf-8"))
    study = StudyResult.model_validate(raw)
    assert study.is_mock is True
    assert len(study.buildings) == len(study.measurements) == 12
    ids = {building.id for building in study.buildings}
    assert len(ids) == 12
    assert {measurement.property_id for measurement in study.measurements} == ids
    for building in study.buildings:
        assert building.market_id == 14 and building.property_type == 11
        assert 29.5 <= building.latitude <= 30.1
        assert -95.9 <= building.longitude <= -95.0
        assert building.building_size_sq_ft > 0 and building.total_num_units > 0
        assert 1900 <= building.year_built <= 2026
        assert building.max_rent > 0 and 0 <= building.vacancy_percent <= 100
    for measurement in study.measurements:
        assert measurement.roof_area_sq_ft > 0
        assert 0 <= measurement.green_cover_percent <= 100
        assert 0 <= measurement.paved_surface_percent <= 100
        assert measurement.parking_space_count >= 0
        assert all(detection.property_id == measurement.property_id for detection in measurement.detections)
    assert study.stats.sample_size == 12
    assert set(study.stats.property_ids) == ids
    assert set(study.stats.coefficients) == set(study.stats.p_values) == set(study.stats.confidence_intervals)
    assert len(study.brief_markdown.strip().split("\n\n")) == 3
    assert "synthetic" in study.brief_markdown.lower()
    assert any("synthetic" in caveat.lower() for caveat in study.caveats)
    assert StudyResult.model_validate_json(study.model_dump_json()) == study


def test_every_tool_signature_matches_contracts() -> None:
    """Check complete coverage, exact parameter names/types/defaults and return models."""
    documented = documented_tools()
    actual = {}
    for path in sorted((ROOT / "backend" / "tools").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = importlib.import_module(f"backend.tools.{path.stem}")
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ == module.__name__:
                actual[f"{module.__name__}.{name}"] = function
    assert documented.keys() == actual.keys(), "Every tool must appear exactly once in CONTRACTS.md"
    for name, function in actual.items():
        declaration = documented[name]
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        args = declaration.args
        assert not args.posonlyargs and not args.kwonlyargs and args.vararg is None and args.kwarg is None
        assert list(signature.parameters) == [arg.arg for arg in args.args], name
        defaults = [inspect.Parameter.empty] * (len(args.args) - len(args.defaults)) + [
            ast.literal_eval(default) for default in args.defaults
        ]
        for expected, default in zip(args.args, defaults):
            parameter = signature.parameters[expected.arg]
            assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, name
            assert expected.annotation is not None, name
            assert hints[expected.arg] == annotation_type(expected.annotation), name
            assert parameter.default == default, name
        assert declaration.returns is not None and hints["return"] == annotation_type(declaration.returns), name
        assert inspect.getdoc(function), name

        # Return annotations may wrap schema classes in list or an optional union.
        pending = [hints["return"]]
        while pending:
            annotation = pending.pop()
            if annotation is type(None):
                continue
            if get_origin(annotation) in (list, Union, types.UnionType):
                pending.extend(get_args(annotation))
            else:
                assert inspect.isclass(annotation) and issubclass(annotation, BaseModel), name
                assert annotation.__module__ == "backend.schemas", name

        source = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        body = source.body[1:] if ast.get_docstring(source) else source.body
        assert len(body) == 1 and isinstance(body[0], ast.Raise), name
        error = body[0].exc
        assert isinstance(error, ast.Name) and error.id == "NotImplementedError", name


def test_api_models_round_trip_mock_study_and_correction() -> None:
    """Ask and correct payloads share a typed study while preserving patch presence."""
    study = StudyResult.model_validate_json(MOCK_STUDY.read_text(encoding="utf-8"))
    request = AskRequest(question=study.question, market_id=14, property_type=11, limit=12)
    assert AskRequest.model_validate_json(request.model_dump_json()) == request
    response = AskResponse(study=study)
    assert AskResponse.model_validate_json(response.model_dump_json()).study == study
    correction = CorrectRequest(
        study_id=study.id, measurement_id=study.measurements[0].id,
        corrections=MeasurementCorrection(green_cover_percent=0, parking_space_count=None),
    )
    assert correction.corrections.model_dump(exclude_unset=True) == {
        "green_cover_percent": 0, "parking_space_count": None,
    }
    decoded = CorrectRequest.model_validate_json(correction.model_dump_json(exclude_unset=True))
    assert decoded == correction
    assert decoded.corrections.model_dump(exclude_unset=True) == {
        "green_cover_percent": 0, "parking_space_count": None,
    }
    result = CorrectResponse(study=study, corrected_measurement=study.measurements[0])
    assert CorrectResponse.model_validate_json(result.model_dump_json()) == result


def test_invalid_correction_and_incomplete_study_are_rejected() -> None:
    """Unknown patch fields, invalid percentages and missing required outputs fail validation."""
    with pytest.raises(ValidationError):
        MeasurementCorrection(green_cover_percent=101)
    with pytest.raises(ValidationError):
        MeasurementCorrection(parking_space_count=-1)
    with pytest.raises(ValidationError):
        MeasurementCorrection(unrecognized_measurement=2)
    study = json.loads(MOCK_STUDY.read_text(encoding="utf-8"))
    del study["brief_markdown"]
    with pytest.raises(ValidationError):
        StudyResult.model_validate(study)
