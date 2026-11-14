"""Test question-specific web studies and saved previews without network access."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.schemas import AskResponse, StudyResult


@pytest.fixture
def client() -> TestClient:
    """Use FastAPI's in-process client without starting a network server."""
    return TestClient(main.app)


def test_showcase_questions_preserve_existing_metadata(client: TestClient) -> None:
    """All three questions and their market/type/controls stay available to the UI."""
    response = client.get("/api/questions")
    assert response.status_code == 200
    expected = json.loads((main.SHOWCASE_DIR / "questions.json").read_text())
    assert response.json() == expected
    questions = response.json()["questions"]
    assert [question["id"] for question in questions] == ["Q1", "Q2", "Q3"]
    assert [question["market_id"] for question in questions] == [14, 14, 20]


def test_mock_preview_validates_as_existing_study_schema(client: TestClient) -> None:
    """The preview serves the existing linked, synthetic twelve-building study."""
    response = client.get("/api/mock-study")
    assert response.status_code == 200
    study = StudyResult.model_validate(response.json())
    assert study.is_mock
    assert len(study.buildings) == len(study.measurements) == 12
    assert {building.market_id for building in study.buildings} == {14}


@pytest.mark.parametrize("question", [
    "Do greener Houston apartments have higher rents?",
    "Which Los Angeles apartments should install solar?",
])
def test_mock_ask_runs_the_requested_question_without_live_calls(
    question: str, client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit mock mode follows the requested research or solar question offline."""
    import socket

    def no_network(*args: object, **kwargs: object) -> None:
        """Fail immediately if a demo route tries to contact an external service."""
        raise AssertionError("The mock web demo must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    for name in ("VEXCEL_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    response = client.post("/api/ask", json={"question": question, "use_mock_tools": True})
    assert response.status_code == 200
    result = AskResponse.model_validate(response.json())
    assert result.study.question == question
    assert result.study.is_mock
    assert len(result.study.buildings) == 12
    if "solar" in question:
        assert {building.market_id for building in result.study.buildings} == {20}
        assert result.study.stats is None
        assert result.study.solar_results and result.study.outreach_drafts
    else:
        assert result.study.stats.outcome == "max_rent"
        assert result.study.solar_results is None
    assert {event.event_type for event in result.events} >= {"plan", "sample", "measure", "inspect", "write"}
    assert len({event.id for event in result.events}) == len(result.events)
    assert all(event.run_id == result.study.id and event.timestamp for event in result.events)
    assert all(event.payload["is_mock"] is True for event in result.events)


@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": " "}, {"question": "test", "limit": 0},
    {"question": "test", "mode": "unknown"}, {"question": "test", "unexpected": True},
    {"question": "test", "market_id": 999},
])
def test_ask_checks_the_existing_request_contract(body: dict, client: TestClient) -> None:
    """Invalid questions/options receive validation errors before returning a study."""
    assert client.post("/api/ask", json=body).status_code == 422


@pytest.mark.parametrize("route", ["/api/questions", "/api/mock-study"])
def test_missing_fixture_is_a_clear_404(
    route: str, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing demo assets return an API error rather than the static HTML page."""
    monkeypatch.setattr(main, "SHOWCASE_DIR", tmp_path)
    response = client.get(route)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize("contents", ["{broken", "[]", '{}'])
def test_invalid_fixture_returns_a_sanitized_error(
    contents: str, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad JSON and incompatible schema data produce controlled server errors."""
    monkeypatch.setattr(main, "SHOWCASE_DIR", tmp_path)
    (tmp_path / "mock_study.json").write_text(contents)
    response = client.get("/api/mock-study")
    assert response.status_code == 500
    assert set(response.json()) == {"detail"}
    assert str(tmp_path) not in response.text


def test_mock_route_rejects_an_unlabeled_fixture(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demo must never present saved synthetic data as a live study."""
    study = json.loads((main.SHOWCASE_DIR / "mock_study.json").read_text())
    study["is_mock"] = False
    (tmp_path / "mock_study.json").write_text(json.dumps(study))
    monkeypatch.setattr(main, "SHOWCASE_DIR", tmp_path)
    response = client.get("/api/mock-study")
    assert response.status_code == 500
    assert response.json()["detail"] == "Showcase study must be marked as mock"
