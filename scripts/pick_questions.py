"""Rank outcome coverage for each city's three most common property types.

Run ``python scripts/pick_questions.py`` to replace notes/question_options.md.
Only local CoStar JSON Lines and enum definitions are read; no credentials or
external APIs are used. Occupancy rates are excluded under the model-doc rules.
"""

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "costar" / "data"
OUTPUT_PATH = PROJECT_ROOT / "notes" / "question_options.md"
MARKETS = {14: "Houston", 20: "Los Angeles"}


def finite_number(value: object) -> float | None:
    """Read a finite numeric scalar without interpreting missing values as zero."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def identifier(value: object) -> int | None:
    """Read a positive integer source id without rounding fractional values."""
    number = finite_number(value)
    return int(number) if number is not None and number > 0 and number.is_integer() else None


def nonzero_mean(*values: object) -> float | None:
    """Average the available nonzero bounds; one reported bound is sufficient."""
    numbers = [number for value in values if (number := finite_number(value)) is not None and number != 0]
    return sum(numbers) / len(numbers) if numbers else None


def completion_year(events: object) -> int | None:
    """Choose the earliest ConstructionComplete year, excluding renovation years."""
    if not isinstance(events, list):
        return None
    years = [
        year for event in events
        if isinstance(event, dict) and event.get("eventType") == 2
        and (year := identifier(event.get("year"))) is not None
    ]
    return min(years) if years else None


def annual_rent_per_sq_ft(low: object, high: object, basis: object, measurement: object) -> float | None:
    """Normalize an explicitly per-SF quote to yearly rent; reject unknown units."""
    rate = nonzero_mean(low, high)
    if rate is None or measurement != 1 or basis not in (1, 2):
        return None
    return rate * (12 if basis == 1 else 1)


def apartment_rents(listings: pd.DataFrame) -> dict[int, float]:
    """Join community asking-rent midpoints to propertyId, ignoring zero bounds."""
    linked = listings.dropna(subset=["propertyId"])
    if linked["propertyId"].duplicated().any():
        raise ValueError("Apartment listings must have unique propertyId values.")
    rents = {}
    for row in linked.to_dict(orient="records"):
        property_id = identifier(row.get("propertyId"))
        rent = nonzero_mean(row.get("minRent"), row.get("maxRent"))
        if property_id is not None and rent is not None:
            rents[property_id] = rent
    return rents


def commercial_rent_properties(listings: pd.DataFrame) -> set[int]:
    """Find properties with their own per-SF quote, avoiding group-rollup leakage."""
    covered = set()
    for row in listings.to_dict(orient="records"):
        lease = row.get("forLeaseSummary")
        if not isinstance(lease, dict):
            continue
        property_id = identifier(row.get("propertyId"))
        spaces = lease.get("spaces")
        spaces = [s for s in spaces if isinstance(s, dict)] if isinstance(spaces, list) else []
        properties = lease.get("properties")
        properties = [p for p in properties if isinstance(p, dict)] if isinstance(properties, list) else []
        group_ids = {
            value for obj, key in [(s, "propertyId") for s in spaces] + [(p, "id") for p in properties]
            if (value := identifier(obj.get(key))) is not None
        }
        if property_id is not None:
            group_ids.add(property_id)
        asset_count = finite_number(lease.get("assetCount"))
        is_group = row.get("listingType") in (16, 17) or len(group_ids) > 1 or (asset_count is not None and asset_count > 1)
        measurement = lease.get("askingRateMeasurementType")
        rate = annual_rent_per_sq_ft(
            lease.get("askingRateLow"), lease.get("askingRateHigh"),
            lease.get("askingRateBasis"), measurement,
        )
        if not is_group and property_id is not None and rate is not None:
            covered.add(property_id)
        for space in spaces:
            space_property = identifier(space.get("propertyId"))
            if space_property is None and not is_group:
                space_property = property_id
            rate = annual_rent_per_sq_ft(
                space.get("rateLow"), space.get("rateHigh"), space.get("rateBasis"), measurement,
            )
            if space_property is not None and rate is not None:
                covered.add(space_property)
    return covered


def rank_options(
    properties: pd.DataFrame, apartments: pd.DataFrame, commercial: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Count unique eligible properties and sort all city/type/outcome options."""
    if properties["id"].isna().any() or properties["id"].duplicated().any():
        raise ValueError("Commercial property ids must be present and unique.")
    monthly_rents = apartment_rents(apartments)
    commercial_ids = commercial_rent_properties(commercial)
    records = []
    for row in properties.to_dict(orient="records"):
        address = row.get("primaryAddress")
        address = address if isinstance(address, dict) else {}
        market = address.get("researchMarketId")
        property_type = identifier(row.get("propertyType"))
        if market not in MARKETS or property_type is None:
            continue
        size = finite_number(row.get("grossBuildingAreaSquareFeet"))
        year = completion_year(row.get("constructionEvents"))
        complete = (
            finite_number(address.get("latitude")) is not None
            and finite_number(address.get("longitude")) is not None
            and size is not None and size > 0 and year is not None
        )
        records.append({"id": row["id"], "market": market, "type": property_type, "complete": complete})
    options = []
    selected = []
    for market, city in MARKETS.items():
        city_records = [row for row in records if row["market"] == market]
        frequencies = Counter(row["type"] for row in city_records)
        top_types = sorted(frequencies, key=lambda code: (-frequencies[code], code))[:3]
        for code in top_types:
            pool = [row for row in city_records if row["type"] == code and row["complete"]]
            ids = {row["id"] for row in pool}
            selected.append({"city": city, "market": market, "type": code, "population": frequencies[code], "frequencies": dict(frequencies)})
            for outcome, count in (
                ("Controls only: building size + year built", len(ids)),
                ("Apartment asking rent (monthly midpoint)", len(ids & monthly_rents.keys())),
                ("Commercial asking rent per SF (annualized)", len(ids & commercial_ids)),
            ):
                options.append({"city": city, "type": code, "outcome": outcome, "count": count})
    options.sort(key=lambda row: (-row["count"], row["city"], row["type"], row["outcome"]))
    return options, selected


def render_report(
    options: list[dict[str, object]], selected: list[dict[str, object]], labels: dict[str, str],
) -> str:
    """Document source mappings, supported outcomes and the descending count table."""
    lines = [
        "# Question options by city and property type", "",
        "Generated with `python scripts/pick_questions.py` using local CoStar JSON Lines "
        "loaded by `pandas.read_json(path, lines=True)`. Counts are unique property ids, not listing or space rows.",
        "Read alongside `notes/data_summary.md`, `data/costar/models/commercial-property.md`, "
        "`apartment-listing.md`, `commercial-listing-summary.md` and `location-occupancy.md`.",
        "", "## Selected property types", "",
        "Top three types are chosen independently in each research market before outcome/control filtering. "
        "Ties use the smaller numeric propertyType code, giving exactly three types per city.", "",
        "| City | Market | Property type | All properties |", "|---|---:|---|---:|",
    ]
    for row in selected:
        name = labels.get(str(row["type"]), "Unknown")
        lines.append(f"| {row['city']} | {row['market']} | {name} ({row['type']}) | {row['population']} |")
    for market, city in MARKETS.items():
        city_selection = [row for row in selected if row["market"] == market]
        if not city_selection:
            continue
        boundary = city_selection[-1]
        excluded_ties = [
            code for code, count in boundary["frequencies"].items()
            if count == boundary["population"] and code not in {row["type"] for row in city_selection}
        ]
        if excluded_ties:
            names = ", ".join(f"{labels.get(str(code), 'Unknown')} ({code})" for code in sorted(excluded_ties))
            lines.extend(["", f"{city}: {names} ties at {boundary['population']} properties but falls after the third type under the stated tie rule."])
    lines.extend([
        "", "## Eligibility and outcome definitions", "",
        "- All rows require finite `primaryAddress.latitude` and `primaryAddress.longitude`, "
        "positive `grossBuildingAreaSquareFeet` (GBA, square feet), and a recorded "
        "`constructionEvents[].year` with `eventType == 2` (ConstructionComplete). "
        "Use the earliest completion year; renovation years never substitute. No unit-count or building-class filter is applied.",
        "- Building size uses GBA consistently, rather than mixing gross/rentable/leasable areas. "
        "The controls-only rows show the maximum cohort available before choosing a rent outcome.",
        "- Apartment rent is the mean of the nonzero, nonmissing `minRent` and `maxRent` from "
        "`apartment-listing.jsonl`, joined by `propertyId` to property `id`. One reported bound suffices; "
        "both zero/missing means unavailable. This is a monthly asking-range midpoint, not an achieved or unit-weighted rent. "
        "The model doc treats Unknown currency on these US communities as USD; unit/bed pricing bases still need care in research.",
        "- Commercial rent requires an explicitly per-square-foot lease quote: "
        "`forLeaseSummary.askingRateMeasurementType == 1`, with `askingRateLow`/`askingRateHigh` "
        "and `askingRateBasis` (1 Monthly, 2 Yearly), or a space's `rateLow`/`rateHigh` and `rateBasis`. "
        "Nonzero bounds are averaged and monthly rates multiplied by 12. Unknown time/area units and per-acre/per-meter quotes are excluded.",
        "- A single-property rollup joins on listing `propertyId`. Group/portfolio rollups are not assigned "
        "to each member: only space quotes with their own `forLeaseSummary.spaces[].propertyId` qualify. "
        "A missing space propertyId falls back to the listing propertyId only for an unambiguous single-property listing.",
        "- Commercial counts mean at least one attributable quote, counted once per building. They include "
        "Active and OffMarket records and off-market/pending space detail in this kit, so describe recorded quote coverage, "
        "not a current-market rent estimate. No rent averages are pooled across listings, currencies, serviceType or direct/sublet flavors; "
        "future analysis must separate/normalize those and area-weight any aggregation across spaces.",
        "", "## Ranked options", "",
        "Sorted globally by the final count, largest first. Zero means no qualifying rent outcome "
        "for that city/type; it does not mean observed zero rent.", "",
        "| City | Property type | Outcome | Buildings with lat/lon + outcome + size + year built |",
        "|---|---|---|---:|",
    ])
    for row in options:
        name = labels.get(str(row["type"]), "Unknown")
        lines.append(f"| {row['city']} | {name} ({row['type']}) | {row['outcome']} | {row['count']} |")
    lines.extend([
        "", "## Occupancy/vacancy outcome: unsupported from location-occupancy alone", "",
        "The model describes tenancy windows (`fromDate`/`toDate`) and occupied space "
        "(`spaces[].squareFeetOccupied`), not a ready property occupancy/vacancy rate. "
        "A current occupied-area sum does not establish complete, non-overlapping tenant coverage or a compatible total inventory denominator. "
        "Restricted/suppressed-space rules also affect aggregate SF. The doc explicitly says occupied area is not a vacancy numerator.",
        "No occupancy/vacancy rate is fabricated, and no numeric coverage row is generated for that unsupported outcome. "
        "`propertyApartment.vacancyPercent` and marketed available area are different sources and are not substituted "
        "for the requested location-occupancy outcome. The tenancy file is not aggregated for this report.",
    ])
    return "\n".join(lines) + "\n"


def generate_report(data_dir: Path = DATA_DIR, output_path: Path = OUTPUT_PATH) -> Path:
    """Load the three supported data sources and write the ranked options report."""
    properties = pd.read_json(data_dir / "commercial-property.jsonl", lines=True)
    apartments = pd.read_json(data_dir / "apartment-listing.jsonl", lines=True)
    commercial = pd.read_json(data_dir / "commercial-listing-summary.jsonl", lines=True)
    labels = json.loads((data_dir.parent / "schemas" / "enums.json").read_text(encoding="utf-8"))["PropertyType"]
    options, selected = rank_options(properties, apartments, commercial)
    report = render_report(options, selected, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(f"Wrote {generate_report()}")
