"""Pull measured roof, tree, paving and vehicle facts out of Vexcel's precomputed attributes.

``vexcel.get_property_info`` keeps Vexcel's dotted keys unchanged; this module reads the
few the AI tools need. Units are imperial because the request uses unit-system=imperial.
Every value is None when Vexcel did not return it.
"""

from __future__ import annotations

from typing import Any

from backend.schemas import VexcelPropertyInfo


def _num(value: Any) -> float | None:
    """Return a float for numbers, else None."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def roof_facts(info: VexcelPropertyInfo | None) -> dict[str, Any]:
    """Summarise one parcel: footprint, roof material/shape, roof objects, trees, paving, cars, date."""
    attrs = (info.attributes if info else None) or {}
    structures = [s for s in ((info.structures if info else None) or []) if isinstance(s, dict)]
    primary = next((s for s in structures if s.get("primary-structure") is True), structures[0] if structures else {})
    footprints = [_num(s.get("footprint.area")) for s in structures]
    footprints = [f for f in footprints if f]

    def total(key: str) -> int | None:
        vals = [_num(s.get(key)) for s in structures]
        vals = [v for v in vals if v is not None]
        return int(sum(vals)) if vals else None

    solar_flags = [s.get("roof-objects.solar.value") for s in structures if isinstance(s.get("roof-objects.solar.value"), bool)]
    date = attrs.get("metadata.ortho-date-estimate")
    return {
        "footprint_sq_ft": round(sum(footprints)) if footprints else None,
        "primary_footprint_sq_ft": round(_num(primary.get("footprint.area"))) if _num(primary.get("footprint.area")) else None,
        "structures": len(structures),
        "roof_material": primary.get("roof-characteristics.material.class"),
        "roof_shape": primary.get("roof-characteristics.shape.value"),
        "roof_condition": _num(primary.get("roof-characteristics.condition")),
        "roof_tree_cover_percent": _num(primary.get("roof-characteristics.tree-cover.percent")),
        "ac_units": total("roof-objects.ac-unit.count"),
        "roof_vents": total("roof-objects.roof-vent.count"),
        "skylights": total("roof-objects.skylight.count"),
        "existing_solar": any(solar_flags) if solar_flags else None,
        "parcel_tree_cover_percent": _num(attrs.get("parcel-tree-cover.percent")),
        "hardscape_percent": _num(attrs.get("hardscapes.percent")),
        "hardscape_sq_ft": round(_num(attrs.get("hardscapes.area"))) if _num(attrs.get("hardscapes.area")) else None,
        "vehicles_count": int(_num(attrs.get("vehicles.count"))) if _num(attrs.get("vehicles.count")) is not None else None,
        "ortho_date": str(date)[:10] if date else None,
    }


def obstruction_level(facts: dict[str, Any]) -> str | None:
    """'none'/'few'/'many' from Vexcel roof-object counts per 10,000 sq ft of roof."""
    roof = facts.get("footprint_sq_ft")
    objects = [facts.get(k) for k in ("ac_units", "roof_vents", "skylights")]
    if not roof or all(o is None for o in objects):
        return None
    density = sum(o or 0 for o in objects) / roof * 10_000
    return "none" if density < 1 else "few" if density < 8 else "many"
