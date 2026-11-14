"""Single-property lookup for the CityLab agent: CoStar id or street address, plus owners and tenants.

Reads the local CoStar kit through backend.tools.costar_data's loader (read only). Field
names are the documented ones: commercial-property ``contacts[].contactRole`` /
``contacts[].locationId`` (roles 7 TrueOwner, 3 RecordedOwner, 4 PropertyManager),
company-detail ``id`` / ``name``, and location-occupancy ``propertyId`` /
``tenantLocationId`` / ``toDate`` (null while the tenant is still in place). Anything
the kit does not record is returned as unavailable, never guessed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from backend.schemas import Building

OWNER_ROLES = {7: "true_owner", 3: "recorded_owner", 4: "property_manager"}
MAX_TENANTS = 10
_STREET_FIELDS = ("streetNumber", "streetDirection", "streetName", "streetType", "streetSuffix")


def normalize(text: str) -> str:
    """Lowercase words and digits only, single spaces."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", (text or "").lower()).split())


def parse_property_id(query: str) -> int | None:
    """A CoStar property id when the query is (or names) one, e.g. "23453" or "property 23453"."""
    q = (query or "").strip()
    # A bare number or "property <n>" is unambiguously an id (CoStar ids can be short).
    match = re.fullmatch(r"(?:costar\s+)?(?:property(?:\s+id)?\s*#?\s*)?(\d{1,10})", q, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\bproperty(?:\s+id)?\s*#?\s*(\d{1,10})\b", q, re.I)
    return int(match.group(1)) if match else None


def _street(address: dict[str, Any] | None) -> str:
    """'1200 Main St' from a CoStar primaryAddress."""
    a = address or {}
    return " ".join(str(a[k]) for k in _STREET_FIELDS if a.get(k))


def address_label(building: Building) -> str:
    """Street, city and state, or '' when the record has no address."""
    a = building.primary_address or {}
    place = ", ".join(str(a[k]) for k in ("city", "state") if a.get(k))
    return ", ".join(p for p in (_street(a), place) if p)


@lru_cache(maxsize=1)
def _address_index() -> list[tuple[int, str]]:
    """(property id, normalised 'street city') for every kit property with a street address."""
    from backend.tools import costar_data

    df = costar_data._load_jsonl("commercial-property.jsonl")
    out = []
    for pid, addr in zip(df["id"], df["primaryAddress"]):
        if isinstance(addr, dict):
            street = _street(addr)
            if street:
                out.append((int(pid), normalize(f"{street} {addr.get('city') or ''}")))
    return out


def search_address(query: str, limit: int = 5) -> list[int]:
    """Property ids whose street address contains every word of the query (house number first)."""
    words = normalize(query).split()
    if not words:
        return []
    number = next((w for w in words if w.isdigit()), None)
    hits = [pid for pid, text in _address_index()
            if all(w in text.split() for w in words) and (number is None or text.split()[0] == number)]
    return hits[:limit]


@lru_cache(maxsize=1)
def _companies() -> dict[int, str]:
    """company-detail id -> name."""
    from backend.tools import costar_data

    df = costar_data._load_jsonl("company-detail.jsonl")
    return {int(i): str(n) for i, n in zip(df["id"], df["name"]) if isinstance(n, str) and n.strip()}


def owners_and_manager(property_id: int) -> dict[str, Any]:
    """True owner, recorded owner and property manager company names from current contacts."""
    from backend.tools import costar_data

    df = costar_data._load_jsonl("commercial-property.jsonl")
    rows = df[df["id"] == property_id]
    found: dict[str, list[str]] = {name: [] for name in OWNER_ROLES.values()}
    if not rows.empty:
        for contact in rows.iloc[0].get("contacts") or []:
            role = OWNER_ROLES.get((contact or {}).get("contactRole"))
            location = (contact or {}).get("locationId")
            name = _companies().get(int(location)) if isinstance(location, (int, float)) else None
            if role and name and name not in found[role]:
                found[role].append(name)
    return {role: names or None for role, names in found.items()}


def current_tenants(property_id: int, limit: int = MAX_TENANTS) -> dict[str, Any]:
    """Current tenants (occupancies with no end date) and how many are unnamed."""
    from backend.tools import costar_data

    df = costar_data._load_jsonl("location-occupancy.jsonl")
    rows = df[(df["propertyId"] == property_id) & (df["toDate"].isna())]
    names, unknown = [], 0
    for location in rows["tenantLocationId"]:
        name = _companies().get(int(location)) if location == location and location is not None else None
        if name:
            if name not in names:
                names.append(name)
        else:
            unknown += 1
    return {"current_tenant_count": len(rows), "tenants": names[:limit] or None, "unnamed_tenants": unknown}


def lookup(query: str, mode: str) -> dict[str, Any]:
    """Resolve a property id or address to one Building plus owner/tenant context.

    Raises LookupError with a user-facing message when nothing (or nothing unique) matches.
    """
    property_id = parse_property_id(query)
    if mode == "mock":
        from backend.tools import mock

        if property_id is None:
            words = normalize(query).split()
            matches = [b.id for b in mock.load_properties() if words and all(
                w in normalize(f"{b.name} {address_label(b)}").split() for w in words)]
            property_id = matches[0] if len(matches) == 1 else None
        building = mock.get_property(property_id) if property_id is not None else None
        if building is None:
            raise LookupError("No demo property matches that id or address (demo ids are 990001-990012 and 990101-990112).")
        return {"building": building.model_dump(mode="json"), "matched_by": "id" if parse_property_id(query) else "address",
                "owners": None, "tenants": None,
                "unavailable": ["owners and tenants (not part of the synthetic demo data)"]}

    from backend.tools import costar_data

    matched_by = "id"
    if property_id is None:
        candidates = search_address(query)
        if not candidates:
            raise LookupError("No CoStar property in the local kit matches that address.")
        if len(candidates) > 1:
            raise LookupError(f"That address matches {len(candidates)} properties; use the CoStar property id "
                              f"(candidates: {', '.join(map(str, candidates))}).")
        property_id, matched_by = candidates[0], "address"
    building = costar_data.get_property(property_id)
    if building is None:
        raise LookupError(f"Property {property_id} is not in the local CoStar kit.")
    unavailable = [f for f in ("latitude", "longitude", "building_size_sq_ft", "year_built", "max_rent",
                               "vacancy_percent") if getattr(building, f) is None]
    return {"building": building.model_dump(mode="json"), "matched_by": matched_by,
            "owners": owners_and_manager(property_id), "tenants": current_tenants(property_id),
            "unavailable": unavailable}
