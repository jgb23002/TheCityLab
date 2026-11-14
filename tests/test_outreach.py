"""Outreach draft tests; the model is always mocked."""

import json
from unittest.mock import Mock

import pytest

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
    assert '"system_size_kw": 72.0' in ask.call_args.args[0]


@pytest.mark.parametrize("body", ["Call me at (713) 555-1234.", "Email bob@example.com", "word " * 200])
def test_falls_back_to_template_when_rules_broken(body: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(return_value=json.dumps({"subject": "Hi", "body_markdown": body})))
    d = outreach.draft_outreach(B, S)
    assert "estimated 72 kW" in d.body_markdown and "17.1 years" in d.body_markdown
    assert any("template" in c for c in d.caveats)


def test_model_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outreach.llm, "ask_text", Mock(side_effect=RuntimeError("x")))
    assert "Owner of Sky Plaza" in outreach.draft_outreach(B, S).body_markdown


def test_no_estimate_no_email() -> None:
    d = outreach.draft_outreach(B, SolarResult(id="s"))
    assert d.body_markdown is None
