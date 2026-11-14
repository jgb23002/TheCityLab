"""Outreach draft tests; the model is always mocked."""

import json
from unittest.mock import Mock

import pytest

INFO = {"building_name": "Sky Plaza", "address": "1 Main St, Houston, TX", "owner": "Acme Holdings",
        "owner_source": "CoStar owner record"}


@pytest.fixture(autouse=True)
def fake_owner_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.owners, "lookup", lambda pid: dict(INFO))

from backend.schemas import Building, SolarResult
from backend.tools import outreach

B = Building(id=5, name="Sky Plaza")
S = SolarResult(id="solar-5", property_id=5, usable_roof_area_sq_ft=6000, capacity_kw=72.0, annual_energy_kwh=93600,
                assumptions={"estimated_annual_savings_usd": 8424, "estimated_cost_usd": 144000,
                             "estimated_payback_years": 17.1, "measured_roof_area_sq_ft": 10000})


def test_uses_model_draft_when_it_follows_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    ask = Mock(return_value=json.dumps({"subject": "Solar at Sky Plaza", "body_markdown": "Dear owner, estimated 72 kW."}))
    monkeypatch.setattr(outreach.llm, "ask_text", ask)
    d = outreach.draft_outreach(B, S)
    assert d.subject == "Solar at Sky Plaza" and "72 kW" in d.body_markdown
    prompt = ask.call_args.args[0]
    assert '"system_size_kw": 72.0' in prompt and '"owner_company": "Acme Holdings"' in prompt
    assert '"Acme Holdings team"' in prompt


@pytest.mark.parametrize("body", ["Call me at (713) 555-1234.", "Email bob@example.com", "word " * 200])
def test_falls_back_to_template_when_rules_broken(body: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(return_value=json.dumps({"subject": "Hi", "body_markdown": body})))
    d = outreach.draft_outreach(B, S)
    assert "estimated 72 kW" in d.body_markdown and "17.1 years" in d.body_markdown
    assert any("template" in c for c in d.caveats)


def test_model_error_falls_back_with_owner_and_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(side_effect=RuntimeError("x")))
    d = outreach.draft_outreach(B, S)
    assert d.body_markdown.startswith("Dear Acme Holdings team,")
    assert "Sky Plaza (1 Main St, Houston, TX)" in d.body_markdown
    assert any("Acme Holdings" in c and "CoStar owner record" in c for c in d.caveats)


def test_no_owner_uses_building_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.owners, "lookup", lambda pid: {**INFO, "owner": None, "owner_source": None})
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(side_effect=RuntimeError("x")))
    d = outreach.draft_outreach(Building(id=9), S)
    assert d.body_markdown.startswith("Dear Owner of Sky Plaza,")
    assert any("No owner company" in c for c in d.caveats)


def test_no_estimate_no_email() -> None:
    d = outreach.draft_outreach(B, SolarResult(id="s"))
    assert d.body_markdown is None


def test_draft_uses_shared_payback_field_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """An estimate passed through the planner's schema retains its actual payback."""
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(side_effect=RuntimeError("offline")))
    result = outreach.draft_outreach(B, S.model_copy(update={"payback_years": 6.5}))
    assert "6.5 years" in result.body_markdown
    assert "17.1 years" not in result.body_markdown
