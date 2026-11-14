"""Draft a rooftop-solar email to a building owner, for a human to review (never sent)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend import llm
from backend.schemas import Building, OutreachDraft, SolarResult
from backend.tools import owners

log = logging.getLogger(__name__)

MAX_WORDS = 180
CONTACT_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|https?://|www\.")

OUTREACH_PROMPT = """Write a short, friendly email from a rooftop-solar consultant to the
owner of a commercial building, proposing a free rooftop solar assessment.

Facts (use only these numbers, word each one as "estimated"):
{facts}

Rules:
- Address it to "{greeting}". Name the building and its street address in the first sentence.
- Put the building name in the subject. Sign it "The CityLab team".
- Under {max_words} words. Plain markdown, no headings.
- Do not invent names, phone numbers, email addresses, links, prices or incentives.
- Say the numbers come from aerial imagery (from aircraft, never "satellite") and public solar data and need an on-site check.
Return JSON only: {{"subject": "...", "body_markdown": "..."}}"""


def _who(building: Building) -> dict[str, Any]:
    """Building name, street address and owner from the Building plus the CoStar kit."""
    info = owners.lookup(building.id)
    return {
        "building": building.name or info.get("building_name") or "your building",
        "address": info.get("address"),
        "owner": info.get("owner"),
        "owner_source": info.get("owner_source"),
    }


def _facts(building: Building, solar: SolarResult) -> dict[str, Any]:
    """Collect the facts and numbers the email may mention."""
    a = solar.assumptions or {}
    who = _who(building)
    return {
        "building": who["building"],
        "address": who["address"],
        "owner_company": who["owner"],
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
    where = f"{f['building']} ({f['address']})" if f["address"] else f["building"]
    body = (
        f"Dear {greeting},\n\n"
        f"We studied the roof of {where} from aerial imagery. Our estimate is that about "
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
    """'<Owner company> team' when CoStar names an owner, else 'Owner of <building>'."""
    who = _who(building)
    if who["owner"]:
        return f"{who['owner']} team"
    return f"Owner of {who['building']}" if who["building"] != "your building" else "Building owner"


def _valid(subject: str, body: str) -> bool:
    """Reject drafts that are too long or contain contact details or links."""
    return bool(subject.strip() and body.strip()) and len(body.split()) <= MAX_WORDS and not CONTACT_PATTERN.search(subject + body)


def draft_outreach(building: Building, solar_estimate: SolarResult) -> OutreachDraft:
    """Draft an outreach message for review without sending it."""
    caveats = ["Draft for human review; CityLab never sends email.",
               "All figures are screening estimates from aerial imagery."]
    who = _who(building)
    if who["owner"]:
        caveats.append(f"Owner '{who['owner']}' comes from the {who['owner_source']}; confirm before sending.")
    else:
        caveats.append("No owner company found in the CoStar kit; addressed generically.")
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
