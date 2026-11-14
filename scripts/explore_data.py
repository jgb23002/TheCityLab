"""Summarize the local CoStar kit without loading credentials or calling APIs.

Run ``python scripts/explore_data.py`` from any directory. Field meanings follow
the kit's model docs; counts and the two small record projections come from data.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "costar" / "data"
OUTPUT_PATH = PROJECT_ROOT / "notes" / "data_summary.md"
MARKETS = {"Houston": 14, "Los Angeles": 20}
REQUIRED_FILES = {
    "commercial-property.jsonl",
    "apartment-listing.jsonl",
    "commercial-listing-summary.jsonl",
    "location-occupancy.jsonl",
}
Record = dict[str, object]


def is_nonempty(value: object) -> bool:
    """Treat nulls, NaN, blank strings and empty containers as missing, not zero."""
    if isinstance(value, (dict, list, str)):
        return bool(value.strip() if isinstance(value, str) else value)
    return value is not None and bool(pd.notna(value))


def values_at(record: object, path: str) -> list[object]:
    """Read a documented dotted field path, expanding lists marked with []."""
    values = [record]
    for part in path.split("."):
        expanded = []
        for value in values:
            if isinstance(value, dict):
                child = value.get(part.removesuffix("[]"))
                if part.endswith("[]"):
                    if isinstance(child, list):
                        expanded.extend(child)
                else:
                    expanded.append(child)
        values = expanded
    return values


def count_rows(records: list[Record], path: str) -> int:
    """Count parent records with at least one non-empty value at a field path."""
    return sum(any(is_nonempty(v) for v in values_at(row, path)) for row in records)


def numeric_table(records: list[Record], paths: list[str]) -> list[str]:
    """Report numeric value counts, keeping missing values distinct from zero."""
    lines = [
        "| Field | Values | Non-empty numeric | Non-zero | Zero |",
        "|---|---:|---:|---:|---:|",
    ]
    for path in paths:
        values = [v for row in records for v in values_at(row, path)]
        numbers = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce")
        valid = numbers.dropna()
        lines.append(
            f"| `{path}` | {len(values)} | {len(valid)} | "
            f"{int(valid.ne(0).sum())} | {int(valid.eq(0).sum())} |"
        )
    return lines


def property_summary(
    records: list[Record], property_types: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Describe market slices, documented building attributes and small examples."""
    overall: Counter[str] = Counter()
    markets: dict[str, Counter[str]] = {name: Counter() for name in MARKETS}
    for row in records:
        value = row.get("propertyType")
        code = str(int(value)) if is_nonempty(value) else "missing"
        overall[code] += 1
        for name, market_id in MARKETS.items():
            if values_at(row, "primaryAddress.researchMarketId") == [market_id]:
                markets[name][code] += 1
    lines = [
        "## Commercial properties",
        "",
        "Market filter: `primaryAddress.researchMarketId`; Houston = 14, Los Angeles = 20.",
        "These records cover the two cities in the kit, not the complete research markets.",
        "",
        "| propertyType | Overall | Houston (14) | Los Angeles (20) |",
        "|---|---:|---:|---:|",
    ]
    for code in sorted(overall, key=lambda c: int(c) if c != "missing" else -1):
        name = property_types.get(code, "Unknown or missing")
        lines.append(
            f"| {code} ({name}) | {overall[code]} | "
            f"{markets['Houston'][code]} | {markets['Los Angeles'][code]} |"
        )
    lines.extend([
        f"| **Total** | **{len(records)}** | **{sum(markets['Houston'].values())}** "
        f"| **{sum(markets['Los Angeles'].values())}** |",
        "",
        f"Non-empty counts below are property rows (denominator {len(records)}), "
        "counted once even when a list has multiple values; zero remains non-empty.",
        "",
        "| Attribute | Exact field path | Non-empty rows |",
        "|---|---|---:|",
    ])
    paths = [
        ("Latitude", "primaryAddress.latitude"),
        ("Longitude", "primaryAddress.longitude"),
        ("Gross building size (SF)", "grossBuildingAreaSquareFeet"),
        ("Rentable building size (SF)", "rentableBuildingAreaSquareFeet"),
        ("Typical floor size (SF)", "grossTypicalFloorSquareFeet"),
        ("Retail leasable size (SF)", "propertyRetail.grossLeasableAreaSquareFeet"),
        ("Construction years, all event types", "constructionEvents[].year"),
        ("Parking details", "parking"),
        ("Parking spaces by type", "parking[].totalSpaces"),
        ("Parking spaces / 1,000 SF", "parkingRatio"),
        ("Apartment units", "propertyApartment.totalNumUnits"),
        ("Units by apartment floor plan", "propertyApartment.models[].numUnits"),
        ("Hotel rooms (not apartment units)", "propertyHospitality.numUnits"),
        ("Stories above ground", "numStoriesAboveGrade"),
        ("Building class", "buildingClass"),
        ("Reported apartment vacancy (%)", "propertyApartment.vacancyPercent"),
    ]
    for label, path in paths:
        lines.append(f"| {label} | `{path}` | {count_rows(records, path)} |")
    built_rows = sum(
        any(event.get("eventType") == 2 and is_nonempty(event.get("year"))
            for event in values_at(row, "constructionEvents[]")
            if isinstance(event, dict))
        for row in records
    )
    lines.append(
        f"| Year built (ConstructionComplete) | `constructionEvents[].year` "
        f"where `eventType == 2` | {built_rows} |"
    )
    both_coordinates = sum(
        count_rows([row], "primaryAddress.latitude")
        and count_rows([row], "primaryAddress.longitude")
        for row in records
    )
    unknown_class = sum(row.get("buildingClass") == 0 for row in records)
    lines.extend([
        "",
        f"Both coordinates: {both_coordinates}/{len(records)}. There is no top-level "
        "`yearBuilt`; construction event 2 is ConstructionComplete; renovation years are separate.",
        "GBA measures the whole envelope; RBA is leasing area; retail uses GLA. "
        "None is a rooftop footprint. Apartment `totalNumUnits` counts units in active models.",
        f"`buildingClass`: 0 Unknown ({unknown_class} rows), 1 A, 2 B, 3 C, 4 F; "
        "class is market-relative and distinct from `buildingRating`.",
    ])
    examples = ["## Short example records", "", "Selected fields only; source propertyType codes are preserved.", ""]
    for code, name in ((5, "Office"), (11, "Multifamily (apartments)")):
        row = next((r for r in records if r.get("propertyType") == code), None)
        if row is None:
            examples.append(f"- {name}: no record in this input.")
            continue
        address = row.get("primaryAddress")
        example = {
            "id": row.get("id"),
            "propertyType": code,
            "primaryAddress": {
                key: address.get(key) if isinstance(address, dict) else None
                for key in ("researchMarketId", "latitude", "longitude")
            },
            "rentableBuildingAreaSquareFeet": row.get("rentableBuildingAreaSquareFeet"),
            "numStoriesAboveGrade": row.get("numStoriesAboveGrade"),
            "buildingClass": row.get("buildingClass"),
            "constructionEvents": [
                {"eventType": 2, "year": event.get("year")}
                for event in values_at(row, "constructionEvents[]")
                if isinstance(event, dict) and event.get("eventType") == 2
            ],
        }
        if code == 11:
            units = values_at(row, "propertyApartment.totalNumUnits")
            example["propertyApartment"] = {"totalNumUnits": units[0] if units else None}
        # pandas' JSON writer converts nullable numeric values to JSON null.
        encoded = pd.Series(example).to_json(force_ascii=False)
        examples.extend([f"{name}:", "```json", str(encoded), "```", ""])
    return lines, examples


def apartment_summary(records: list[Record]) -> list[str]:
    """Report monthly rent coverage, floor-plan bases and the property join."""
    property_ids = [row.get("propertyId") for row in records]
    linked = sum(is_nonempty(value) for value in property_ids)
    unique = len({value for value in property_ids if is_nonempty(value)})
    plans = [p for row in records for p in values_at(row, "apartmentModels[]") if isinstance(p, dict)]
    bases = Counter(p.get("rentPricingType") for p in plans)
    return [
        "## Apartment listings",
        "",
        f"One active community listing per `propertyId`; join to commercial-property `id`. "
        f"Present links: {linked}/{len(records)}; distinct properties: {unique}; floor plans: {len(plans)}.",
        "Rents are asking dollars/month. Zero means no rent given; exclude zero and null before averaging. "
        "Values counts scalar slots under existing parent objects, including null values.",
        "",
        *numeric_table(records, ["minRent", "maxRent", "apartmentModels[].minRent", "apartmentModels[].maxRent"]),
        "",
        "`monetaryUnit` provides currency (1 USD, 0 Unknown; the docs treat these US communities as USD). "
        "`apartmentModels[].rentPricingType` is a basis enum, not a rent amount: "
        f"1 PerUnit={bases[1]}, 2 PerBed={bases[2]}, 3 PerBedroom={bases[3]}; do not mix these bases.",
        "`apartmentModels[].id` joins `propertyApartment.models[].id`. "
        "Other useful listing fields: `yearBuilt`, `totalNumUnits`, `totalActiveUnits`, `numUnitsAvailable`; "
        "floor plans have `numUnits` and `numUnitsAvailable`. Availability is not measured vacancy.",
    ]


def listing_summary(records: list[Record]) -> list[str]:
    """Describe asking rents and advertised space using the documented rollups."""
    return [
        "## Commercial listing summaries",
        "",
        "`propertyId` joins commercial-property `id`; group listings also link via "
        "`forLeaseSummary.properties[].id`, `forLeaseSummary.spaces[].propertyId` and `forLeaseSummary.lands[].propertyId`.",
        "The following counts cover all listings/spaces, including off-market and lease-pending detail; "
        "Values counts scalar slots under existing parent objects, not property rows.",
        "",
        *numeric_table(records, [
            "forLeaseSummary.askingRateLow", "forLeaseSummary.askingRateHigh",
            "forLeaseSummary.totalAvailableSquareFeet", "forLeaseSummary.minDivisibleSquareFeet",
            "forLeaseSummary.maxContiguousSquareFeet", "forLeaseSummary.spaces[].rateLow",
            "forLeaseSummary.spaces[].rateHigh", "forLeaseSummary.spaces[].squareFeetMin",
            "forLeaseSummary.spaces[].squareFeetMax", "forLeaseSummary.spaces[].squareFeetMaxContiguous",
            "forLeaseSummary.properties[].totalAvailableSquareFeet",
            "forSaleSummary.totalAvailableSquareFeet", "auctionForSaleSummary.totalAvailableSquareFeet",
        ]),
        "",
        "Normalize using `forLeaseSummary.askingRateBasis` (1 Monthly, 2 Yearly), "
        "`askingRateMeasurementType` (1 SF, 2 acres, 3 square meters), `serviceType` and listing `monetaryUnit`; "
        "spaces carry `rateBasis`, `serviceType` and `monetaryUnit`. Zero rates mean unreported asking rent.",
        "For current availability, select lease listings with `listingStatus == 1` (Active), "
        "drop spaces with `dateOffMarket`, and state whether `isLeasePending` spaces are excluded. "
        "Stored rollups can disagree with filtered detail; deduplicate spaces across listings before summing.",
        "`totalAvailableSquareFeet` is marketed space, including occupied/future space; it is not vacant SF. "
        "Asking rent is not achieved contract/effective rent; normalize to $/SF/year and area-weight comparisons.",
    ]


def occupancy_summary(records: list[Record]) -> list[str]:
    """Explain tenancy history and why it does not directly provide vacancy rates."""
    open_ended = sum(not is_nonempty(row.get("toDate")) for row in records)
    return [
        "## Location occupancy: interpretation and rate limits",
        "",
        "Each row is a company's tenancy in a property over `fromDate` to `toDate`, not a property's occupancy percentage. "
        "`propertyId` joins commercial-property `id`; `tenantLocationId` joins company-detail `id` (null means unknown tenant).",
        f"`toDate` is null on {open_ended}/{len(records)} rows (still in place per the model doc). "
        "For an as-of analysis, apply `fromDate`/`toDate` windows rather than summing all historical tenancies. "
        "Use `moveInDate`, not lease sign date, for absorption.",
        "Occupied area is `sum(spaces[].squareFeetOccupied)` for the appropriate tenancy window. "
        "This is not a vacancy numerator, and marketed listing availability cannot substitute for vacant area.",
        "No direct, reliable per-property occupancy/vacancy rate is supplied by this file alone. "
        "A derived occupancy share requires complete current tenant coverage, non-overlapping/deduplicated spaces, "
        "and a compatible known property RBA (retail GLA) denominator; audit these before deriving a vacancy complement. "
        "For apartments, `propertyApartment.vacancyPercent` is a separately reported property-level figure; "
        "`numUnitsAvailable / totalNumUnits` measures advertised availability, not vacancy.",
    ]


def matterport_summary(csv_path: Path) -> list[str]:
    """Count distinct mapped properties without mistaking scans for properties."""
    frame = pd.read_csv(csv_path, dtype={"MatterportModelId": "string"})
    has_model = frame["MatterportModelId"].fillna("").str.strip().ne("")
    total = frame["PropertyId"].nunique()
    mapped = frame.loc[has_model, "PropertyId"].nunique()
    columns = ", ".join(f"`{column}`" for column in frame.columns)
    return [
        "## Commercial property / Matterport CSV",
        "",
        f"`commercial-properties.csv`: {len(frame)} rows; columns: {columns}.",
        f"Distinct `PropertyId`: {total}; properties with a non-blank `MatterportModelId`: {mapped}; "
        f"without a model: {total - mapped}. Non-blank model rows: {int(has_model.sum())}.",
        "One row represents a property/model scan mapping; multiple scans repeat a property. "
        "Join CSV `PropertyId` to commercial-property `id`; `MatterportModelId` identifies its optional 3D tour.",
    ]


def generate_summary(data_dir: Path = DATA_DIR, output_path: Path = OUTPUT_PATH) -> Path:
    """Load every JSON Lines file sequentially and write a report under 200 lines."""
    paths = sorted(data_dir.glob("*.jsonl"))
    missing = REQUIRED_FILES - {path.name for path in paths}
    if missing:
        raise FileNotFoundError(f"Missing required JSON Lines files: {', '.join(sorted(missing))}")
    enums = json.loads((data_dir.parent / "schemas" / "enums.json").read_text(encoding="utf-8"))
    inventory = ["| File | Rows | Top-level columns |", "|---|---:|---|"]
    sections: dict[str, list[str]] = {}
    examples: list[str] = []
    for path in paths:
        frame = pd.read_json(path, lines=True)
        columns = ", ".join(f"`{str(column).replace('|', '&#124;')}`" for column in frame.columns)
        inventory.append(f"| `{path.name}` | {len(frame)} | {columns} |")
        if path.name in REQUIRED_FILES:
            records = frame.to_dict(orient="records")
            if path.name == "commercial-property.jsonl":
                sections[path.name], examples = property_summary(records, enums["PropertyType"])
            elif path.name == "apartment-listing.jsonl":
                sections[path.name] = apartment_summary(records)
            elif path.name == "commercial-listing-summary.jsonl":
                sections[path.name] = listing_summary(records)
            else:
                sections[path.name] = occupancy_summary(records)
            del records
        del frame
    lines = [
        "# CoStar data summary", "",
        "Generated by `python scripts/explore_data.py` from local `data/costar/` files; no API calls or credentials are used.",
        "Counts are computed from JSON Lines with `pandas.read_json(path, lines=True)`. "
        "Non-empty excludes null/NaN, blank strings and empty lists/dicts; numeric zero remains present.",
        "Source interpretation: `data/costar/index.md`, model docs for commercial-property, apartment-listing, "
        "commercial-listing-summary and location-occupancy, `concepts/geography-and-markets.md`, and `schemas/enums.json`.",
        "", "## File inventory", "", *inventory, "",
    ]
    for name in ("commercial-property.jsonl", "apartment-listing.jsonl",
                 "commercial-listing-summary.jsonl", "location-occupancy.jsonl"):
        lines.extend([*sections[name], ""])
    lines.extend([*matterport_summary(data_dir.parent / "commercial-properties.csv"), "", *examples])
    report = "\n".join(lines).rstrip() + "\n"
    if len(report.splitlines()) >= 200:
        raise ValueError("Summary must stay under 200 lines; shorten the report before writing it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    destination = generate_summary()
    print(f"Wrote {destination}")
