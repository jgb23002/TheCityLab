"""Look up a building's name, street address and owner company in the local CoStar kit.

Owner sources, in order:
1. The property's current owner contact (contactRole 7 TrueOwner, then 3 RecordedOwner)
   joined on ``locationId`` to company-detail ``id``.
2. The buyer on the property's most recent sale comp (contactRole 43 TrueBuyer, then
   42 RecordedBuyer) by ``soldDate``; the last buyer may no longer own it.

The index is built once from the JSON Lines files and cached in data/cache/owner_index.json.
Kit contact details are fake by design, and none are returned here.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from backend import config

log = logging.getLogger(__name__)

OWNER_ROLES = (7, 3)  # TrueOwner, RecordedOwner
BUYER_ROLES = (43, 42)  # TrueBuyer, RecordedBuyer
INDEX_PATH = config.CACHE_DIR / "owner_index.json"


def _list(value: Any) -> list:
    """Return value if it is a list, else an empty list (JSON Lines rows can hold NaN)."""
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    """Return a stripped non-empty string, else None (JSON Lines rows can hold NaN)."""
    return value.strip().rstrip(",").strip() or None if isinstance(value, str) else None


def _street(addr: Any) -> str | None:
    """Format a CoStar primaryAddress as 'number street type, city, state'."""
    if not isinstance(addr, dict):
        return None
    street = " ".join(str(addr[k]) for k in ("streetNumber", "streetName", "streetType") if addr.get(k))
    parts = [p for p in (street, addr.get("city"), addr.get("state")) if p]
    return ", ".join(parts) or None


def build_index() -> dict[str, dict[str, Any]]:
    """Read the CoStar kit and map property id -> name, address, owner and owner source."""
    import pandas as pd

    data = config.COSTAR_DATA_DIR
    companies = {
        int(r["id"]): _text((r.get("profile") if isinstance(r.get("profile"), dict) else {}).get("companyName"))
        or _text(r.get("name"))
        for r in pd.read_json(data / "company-detail.jsonl", lines=True)[["id", "name", "profile"]].to_dict("records")
    }
    index: dict[str, dict[str, Any]] = {}
    props = pd.read_json(data / "commercial-property.jsonl", lines=True)
    for r in props[["id", "buildingName", "primaryAddress", "contacts"]].to_dict("records"):
        owner, source = None, None
        for role in OWNER_ROLES:
            names = [companies.get(c.get("locationId")) for c in _list(r.get("contacts")) if c.get("contactRole") == role]
            names = [n for n in names if n]
            if names:
                owner, source = names[0], "CoStar owner record" if role == 7 else "CoStar recorded owner"
                break
        name = _text(r.get("buildingName"))
        index[str(int(r["id"]))] = {"building_name": name, "address": _street(r.get("primaryAddress")),
                                    "owner": owner, "owner_source": source}

    sales = pd.read_json(data / "commercial-sale-comp.jsonl", lines=True)
    latest: dict[str, tuple[str, str, str]] = {}
    for r in sales[["soldDate", "entities", "contacts"]].to_dict("records"):
        sold = str(r.get("soldDate") or "")
        buyer = None
        for role in BUYER_ROLES:
            names = [_text(c.get("locationName")) for c in _list(r.get("contacts")) if c.get("contactRole") == role and c.get("locationName")]
            if names:
                buyer = names[0]
                break
        if not buyer:
            continue
        for e in _list(r.get("entities")):
            pid = e.get("propertyId")
            if pid is not None and str(int(pid)) in index and sold >= latest.get(str(int(pid)), ("",))[0]:
                latest[str(int(pid))] = (sold, buyer, sold[:10])
    for pid, (_, buyer, date) in latest.items():
        if not index[pid]["owner"]:
            index[pid].update(owner=buyer, owner_source=f"buyer in CoStar sale recorded {date}")
    return index


@lru_cache(maxsize=1)
def load_index() -> dict[str, dict[str, Any]]:
    """Load the cached index, building it from the kit on first use; {} if the kit is missing."""
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    try:
        index = build_index()
    except (FileNotFoundError, ValueError) as exc:
        log.warning("CoStar kit unavailable for owner lookup: %s", type(exc).__name__)
        return {}
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index))
    return index


def lookup(property_id: int) -> dict[str, Any]:
    """Return {building_name, address, owner, owner_source} for a property; values may be None."""
    return load_index().get(str(property_id), {"building_name": None, "address": None, "owner": None, "owner_source": None})
