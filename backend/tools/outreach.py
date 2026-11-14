"""Draft a rooftop-solar email to a building owner, for a human to review (never sent)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend import llm
from backend.schemas import Building, OutreachDraft, SolarResult

log = logging.getLogger(__name__)

MAX_WORDS = 180
CONTACT_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|https?://|www\.")

OUTREACH_PROMPT = """Write a short, friendly email from a rooftop-solar consultant to the
owner of a commercial building, proposing a free rooftop solar assessment.

Facts (use only these numbers, word each one as "estimated"):
{facts}

Rules:
- Address it to "{greeting}". Sign it "The CityLab team".
- Under {max_words} words. Plain markdown, no headings.
- Do not invent names, phone numbers, email addresses, links, prices or incentives.
- Say the numbers come from aerial imagery and public solar data and need an on-site check.
Return JSON only: {{"subject": "...", "body_markdown": "..."}}"""


def _facts(building: Building, solar: SolarResult) -> dict[str, Any]:
    """Collect the numbers the email may mention."""
    a = solar.assumptions or {}
    return {
        "building": building.name or f"CoStar property {building.id}",
        "measured_roof_area_sq_ft": a.get("measured_roof_area_sq_ft"),
        "usable_roof_area_sq_ft": solar.usable_roof_area_sq_ft,
        "system_size_kw": solar.capacity_kw,
        "yearly_output_kwh": solar.annual_energy_kwh,
        "yearly_savings_usd": a.get("estimated_annual_savings_usd"),
        "installed_cost_usd": a.get("estimated_cost_usd"),
        "simple_payback_years": a.get("estimated_payback_years"),
    }


def _fmt(value: Any, unit: str = "") -> str:
    """Format a number with thousands separators, or 'unknown'."""
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        text = f"{value:,.1f}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value):,}"
        return f"{text}{unit}"
    return str(value)


def template_draft(building: Building, solar: SolarResult) -> tuple[str, str]:
    """A fixed-wording email used when the model is unavailable or breaks the rules."""
    f = _facts(building, solar)
    greeting = building_greeting(building)
    subject = f"Estimated rooftop solar potential at {f['building']}"
    body = (
        f"Dear {greeting},\n\n"
        f"We studied the roof of {f['building']} from aerial imagery. Our estimate is that about "
        f"{_fmt(f['usable_roof_area_sq_ft'], ' sq ft')} could hold an estimated {_fmt(f['system_size_kw'], ' kW')} system, "
        f"producing an estimated {_fmt(f['yearly_output_kwh'], ' kWh')} a year and saving an estimated "
        f"${_fmt(f['yearly_savings_usd'])} a year, for an estimated simple payback of "
        f"{_fmt(f['simple_payback_years'], ' years')}.\n\n"
        "These are screening estimates from aerial imagery and public solar data; an on-site check "
        "would confirm them. Would you be open to a free assessment?\n\n"
        "Best regards,\nThe CityLab team"
    )
    return subject, body


def building_greeting(building: Building) -> str:
    """Generic greeting; CoStar owner names are not in the Building contract."""
    return f"Owner of {building.name}" if building.name else "Building owner"


def _valid(subject: str, body: str) -> bool:
    """Reject drafts that are too long or contain contact details or links."""
    return bool(subject.strip() and body.strip()) and len(body.split()) <= MAX_WORDS and not CONTACT_PATTERN.search(subject + body)


def draft_outreach(building: Building, solar_estimate: SolarResult) -> OutreachDraft:
    """Draft an outreach message for review without sending it."""
    caveats = ["Draft for human review; CityLab never sends email.",
               "All figures are screening estimates from aerial imagery."]
    if solar_estimate.capacity_kw is None:
        return OutreachDraft(id=f"outreach-{building.id}", property_id=building.id,
                             caveats=caveats + ["No solar estimate, so no email was drafted."])
    prompt = OUTREACH_PROMPT.format(facts=json.dumps(_facts(building, solar_estimate), indent=2),
                                    greeting=building_greeting(building), max_words=MAX_WORDS)
    subject = body = ""
    try:
        reply = llm.ask_text(prompt)
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        subject, body = str(data.get("subject") or ""), str(data.get("body_markdown") or "")
    except Exception as exc:
        log.warning("Outreach model call failed: %s", type(exc).__name__)
    if not _valid(subject, body):
        subject, body = template_draft(building, solar_estimate)
        caveats.append("Used the fixed template wording.")
    return OutreachDraft(id=f"outreach-{building.id}", property_id=building.id,
                         subject=subject, body_markdown=body, caveats=caveats)
