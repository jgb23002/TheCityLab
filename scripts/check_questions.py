"""Count complete Multifamily research cohorts and observed parking types.

Run ``python scripts/check_questions.py`` to write notes/question_options.md.
Presence checks retain zero units and zero vacancy; only rent and class exclude
zero. The script reads local files only and never loads credentials.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "costar" / "data"
OUTPUT_PATH = PROJECT_ROOT / "notes" / "question_options.md"
MARKETS = {14: "Houston", 20: "Los Angeles"}


def is_present(value: object) -> bool:
    """Distinguish missing scalar values from meaningful zero values."""
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and bool(pd.notna(value))


def nested_value(record: object, *path: str) -> object:
    """Read a nested source field without treating a missing object as zero."""
    for key in path:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def has_completion_year(events: object) -> bool:
    """Check for a year on any ConstructionComplete event, counting a row once."""
    return isinstance(events, list) and any(
        isinstance(event, dict) and event.get("eventType") == 2
        and is_present(event.get("year"))
        for event in events
    )


def count_options(properties: pd.DataFrame, listings: pd.DataFrame) -> dict[str, object]:
    """Count distinct property cohorts and both parking entries and properties."""
    if properties["id"].isna().any() or properties["id"].duplicated().any():
        raise ValueError("Commercial property ids must be present and unique.")
    linked = listings.dropna(subset=["propertyId"])
    if linked["propertyId"].duplicated().any():
        raise ValueError("Apartment listings must have unique propertyId values.")
    rent_by_property = linked.set_index("propertyId")["maxRent"].to_dict()
    cities = {market: Counter() for market in MARKETS}
    entries = {market: Counter() for market in MARKETS}
    parking_properties = {market: Counter() for market in MARKETS}
    for record in properties.loc[properties["propertyType"].eq(11)].to_dict(orient="records"):
        market = nested_value(record, "primaryAddress", "researchMarketId")
        if market not in MARKETS:
            continue
        city = cities[market]
        city["multifamily"] += 1
        parking = record.get("parking")
        parking = [entry for entry in parking if isinstance(entry, dict)] if isinstance(parking, list) else []
        codes = []
        for entry in parking:
            value = entry.get("parkingType")
            code = str(int(value)) if is_present(value) else "missing"
            codes.append(code)
            entries[market][code] += 1
        parking_properties[market].update(set(codes))
        city["without_parking_entries"] += not bool(parking)
        units = nested_value(record, "propertyApartment", "totalNumUnits")
        grade = record.get("buildingClass")
        complete = (
            is_present(nested_value(record, "primaryAddress", "latitude"))
            and is_present(nested_value(record, "primaryAddress", "longitude"))
            and is_present(units)
            and has_completion_year(record.get("constructionEvents"))
            and is_present(grade) and grade != 0
        )
        if not complete:
            continue
        rent = rent_by_property.get(record["id"])
        has_rent = is_present(rent) and rent != 0
        has_vacancy = is_present(nested_value(record, "propertyApartment", "vacancyPercent"))
        city["controls_complete"] += 1
        city["with_rent"] += bool(has_rent)
        city["with_vacancy"] += bool(has_vacancy)
        city["with_both"] += bool(has_rent and has_vacancy)
        city["zero_units"] += units == 0
        if units > 0:
            city["positive_units_controls"] += 1
            city["positive_units_rent"] += bool(has_rent)
            city["positive_units_vacancy"] += bool(has_vacancy)
            city["positive_units_both"] += bool(has_rent and has_vacancy)
    return {"cities": cities, "parking_entries": entries, "parking_properties": parking_properties}


def render_report(
    counts: dict[str, object], parking_labels: dict[str, str], showcase_note_exists: bool
) -> str:
    """Explain the exact cohort rules and label all observed parking type codes."""
    lines = [
        "# Showcase question options", "",
        "Generated with `python scripts/check_questions.py` from local JSON Lines files using "
        "`pandas.read_json(path, lines=True)`. No API calls or credentials are used.",
        "Reference: `notes/data_summary.md` and the CoStar commercial-property/apartment-listing model docs.",
        ("Showcase reference: `notes/showcase_questions.md`." if showcase_note_exists else
         "`notes/showcase_questions.md` was absent; the supplied Q1–Q3 questions and "
         "`data/showcase/questions.json` provide the showcase context."),
        "", "## Completeness rules", "",
        "- Scope: `propertyType == 11` (Multifamily); market from `primaryAddress.researchMarketId` (14 Houston, 20 Los Angeles).",
        "- Required controls: non-null `primaryAddress.latitude`, `primaryAddress.longitude`, "
        "`propertyApartment.totalNumUnits`, a `constructionEvents[].year` where `eventType == 2` "
        "(ConstructionComplete), and non-null `buildingClass != 0`. Renovation events do not count.",
        "- (a) Rent: non-null `apartment-listing.maxRent != 0`, joined on "
        "`apartment-listing.propertyId == commercial-property.id`; (b) vacancy: non-null "
        "`propertyApartment.vacancyPercent`; (c) both conditions on the same property.",
        "- Counts are distinct properties. Missing listing links cannot supply rent but do not exclude "
        "a property from vacancy counts. Multiple completion years count a property once.",
        "- These are presence checks: zero units and zero vacancy count as present. No positive-unit, "
        "rent-positivity, vacancy-range or construction-status filter is added. Null/NaN and blanks are missing.",
        "", "## Eligible properties", "",
        "| City (market) | All Multifamily | Complete controls | + (a) non-zero maxRent | + (b) reported vacancy | + (c) both |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for market, name in MARKETS.items():
        city = counts["cities"][market]
        lines.append(
            f"| {name} ({market}) | {city['multifamily']} | {city['controls_complete']} | "
            f"{city['with_rent']} | {city['with_vacancy']} | {city['with_both']} |"
        )
    lines.extend([
        "", "Q1 uses column (a); Q2 uses column (b); a shared Q1/Q2 cohort uses column (c). "
        "The Los Angeles retrofit question needs complete controls without either outcome.",
        "", "## Positive-unit comparison", "",
        "The earlier sample preparer excludes zero unit totals. Adding only `totalNumUnits > 0` "
        "to the completeness rules above yields:", "",
        "| City | Complete controls | + rent | + vacancy | + both | Zero-unit rows in original complete-controls pool |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for market, name in MARKETS.items():
        city = counts["cities"][market]
        lines.append(
            f"| {name} | {city['positive_units_controls']} | {city['positive_units_rent']} | "
            f"{city['positive_units_vacancy']} | {city['positive_units_both']} | {city['zero_units']} |"
        )
    lines.extend([
        "", "## Observed parking types", "",
        "Scope: all Multifamily properties in the two markets, before completeness/outcome filtering. "
        "Entries count `parking[]` objects; properties count distinct properties with that type. "
        "One property may have several entries/types, so property columns are not additive.", "",
        "| parking[].parkingType | Documented label | Houston entries | Houston properties | LA entries | LA properties |",
        "|---|---|---:|---:|---:|---:|",
    ])
    entries = counts["parking_entries"]
    properties = counts["parking_properties"]
    codes = set(entries[14]) | set(entries[20])
    for code in sorted(codes, key=lambda c: int(c) if c != "missing" else -1):
        label = parking_labels.get(code, "Missing" if code == "missing" else "Undocumented value")
        lines.append(
            f"| {code} | {label} | {entries[14][code]} | {properties[14][code]} | "
            f"{entries[20][code]} | {properties[20][code]} |"
        )
    missing = counts["cities"]
    lines.extend([
        "",
        f"Properties with no parking entries: Houston {missing[14]['without_parking_entries']}; "
        f"Los Angeles {missing[20]['without_parking_entries']}.",
        "Surface parking codes are **1 Surface** and **2 SurfaceTandem**. "
        "For AI parking-space validation, sum `parking[].totalSpaces` only for those types. "
        "Covered, covered-tandem, garages, trailer and ambiguous reserved types are excluded. "
        "Absent or incomplete surface counts remain unknown, not zero.",
    ])
    return "\n".join(lines) + "\n"


def generate_report(data_dir: Path = DATA_DIR, output_path: Path = OUTPUT_PATH) -> Path:
    """Load the local kit, count eligible cohorts, and write the options report."""
    properties = pd.read_json(data_dir / "commercial-property.jsonl", lines=True)
    listings = pd.read_json(data_dir / "apartment-listing.jsonl", lines=True)
    labels = json.loads((data_dir.parent / "schemas" / "enums.json").read_text(encoding="utf-8"))["ParkingType"]
    counts = count_options(properties, listings)
    report = render_report(counts, labels, (PROJECT_ROOT / "notes" / "showcase_questions.md").is_file())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(f"Wrote {generate_report()}")
