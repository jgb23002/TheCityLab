"""Create reproducible CoStar apartment samples for the three showcase questions.

Run ``python scripts/prepare_showcase.py``. This reads only the local kit and
writes small derived CSV files and their selection manifest, without API calls.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "costar" / "data"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
SURFACE_PARKING_TYPES = (1, 2)  # Surface and SurfaceTandem, not covered/garage.
CITY_MARKETS = {"houston": 14, "los_angeles": 20}
FIELD_SOURCES = {
    "property_id": "commercial-property.id",
    "market_id": "commercial-property.primaryAddress.researchMarketId",
    "property_type": "commercial-property.propertyType",
    "latitude": "commercial-property.primaryAddress.latitude",
    "longitude": "commercial-property.primaryAddress.longitude",
    "total_num_units": "commercial-property.propertyApartment.totalNumUnits",
    "year_built": "commercial-property.constructionEvents[].year where eventType == 2; earliest completion year",
    "building_class": "commercial-property.buildingClass",
    "apartment_listing_id": "apartment-listing.id; joined on apartment-listing.propertyId == commercial-property.id",
    "max_rent": "apartment-listing.maxRent; dollars/month; zero or missing becomes unknown",
    "vacancy_percent": "commercial-property.propertyApartment.vacancyPercent; percent, including zero",
    "surface_parking_spaces": "sum(commercial-property.parking[].totalSpaces where parkingType in [1, 2]); unknown if absent or incomplete",
}
INTEGER_FIELDS = (
    "property_id", "market_id", "property_type", "total_num_units", "year_built",
    "building_class", "apartment_listing_id", "surface_parking_spaces",
)


def finite_number(value: object) -> float | None:
    """Return a finite scalar number, leaving missing and invalid values unknown."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def completion_year(events: object) -> int | None:
    """Use the earliest recorded ConstructionComplete year, excluding renovation."""
    years = []
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("eventType") == 2:
                year = finite_number(event.get("year"))
                if year is not None and year > 0 and year.is_integer():
                    years.append(int(year))
    return min(years) if years else None


def surface_parking_spaces(parking: object) -> int | None:
    """Sum known surface/tandem spaces; absent or incomplete counts stay unknown."""
    if not isinstance(parking, list):
        return None
    entries = [
        entry for entry in parking
        if isinstance(entry, dict) and entry.get("parkingType") in SURFACE_PARKING_TYPES
    ]
    if not entries:
        return None
    counts = [finite_number(entry.get("totalSpaces")) for entry in entries]
    if any(count is None or count < 0 or not count.is_integer() for count in counts):
        return None
    return int(sum(counts))


def prepare_candidates(properties: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """Join listing outcomes to apartment properties with complete valid controls."""
    if properties["id"].isna().any() or properties["id"].duplicated().any():
        raise ValueError("Commercial property ids must be present and unique.")
    linked = listings.dropna(subset=["propertyId"])
    if linked["propertyId"].duplicated().any():
        raise ValueError("Apartment listings must have unique propertyId values.")
    listing_by_property = linked.set_index("propertyId").to_dict(orient="index")
    rows = []
    for record in properties.loc[properties["propertyType"].eq(11)].to_dict(orient="records"):
        address = record.get("primaryAddress")
        apartment = record.get("propertyApartment")
        if not isinstance(address, dict) or not isinstance(apartment, dict):
            continue
        market = address.get("researchMarketId")
        latitude = finite_number(address.get("latitude"))
        longitude = finite_number(address.get("longitude"))
        units = finite_number(apartment.get("totalNumUnits"))
        year = completion_year(record.get("constructionEvents"))
        grade = finite_number(record.get("buildingClass"))
        if (market not in CITY_MARKETS.values()
                or latitude is None or not -90 <= latitude <= 90
                or longitude is None or not -180 <= longitude <= 180
                or units is None or units <= 0 or not units.is_integer()
                or year is None or grade not in (1, 2, 3, 4)):
            continue
        listing = listing_by_property.get(record["id"], {})
        rent = finite_number(listing.get("maxRent"))
        vacancy = finite_number(apartment.get("vacancyPercent"))
        rows.append({
            "property_id": int(record["id"]),
            "market_id": int(market),
            "property_type": 11,
            "latitude": latitude,
            "longitude": longitude,
            "total_num_units": int(units),
            "year_built": year,
            "building_class": int(grade),
            "apartment_listing_id": finite_number(listing.get("id")),
            "max_rent": rent if rent is not None and rent > 0 else None,
            "vacancy_percent": vacancy if vacancy is not None and 0 <= vacancy <= 100 else None,
            "surface_parking_spaces": surface_parking_spaces(record.get("parking")),
        })
    candidates = pd.DataFrame(rows, columns=list(FIELD_SOURCES))
    for column in INTEGER_FIELDS:
        candidates[column] = pd.array(candidates[column], dtype="Int64")
    return candidates


def build_samples(
    data_dir: Path = DATA_DIR, output_dir: Path = SAMPLE_DIR,
    size: int = 50, seed: int = 42,
) -> dict[str, object]:
    """Sample each city without replacement, validating both pools before writing."""
    if size < 1:
        raise ValueError("Sample size must be positive.")
    properties = pd.read_json(data_dir / "commercial-property.jsonl", lines=True)
    listings = pd.read_json(data_dir / "apartment-listing.jsonl", lines=True)
    candidates = prepare_candidates(properties, listings)
    samples = {}
    cities = {}
    for city, market_id in CITY_MARKETS.items():
        pool = candidates.loc[candidates["market_id"].eq(market_id)]
        controls_complete = len(pool)
        if city == "houston":
            pool = pool.dropna(subset=["max_rent", "vacancy_percent"])
        if len(pool) < size:
            raise ValueError(f"{city}: need {size} eligible properties, found {len(pool)}.")
        # Sort first so reordering source records cannot change the selection.
        sample = pool.sort_values("property_id").sample(n=size, random_state=seed)
        samples[city] = sample.sort_values("property_id")
        cities[city] = {
            "market_id": market_id,
            "controls_complete_count": controls_complete,
            "eligible_count": len(pool),
            "selected_count": len(sample),
            "required_outcomes": ["max_rent", "vacancy_percent"] if city == "houston" else [],
            "selected_with_rent": int(sample["max_rent"].notna().sum()),
            "selected_with_vacancy": int(sample["vacancy_percent"].notna().sum()),
            "selected_with_zero_vacancy": int(sample["vacancy_percent"].eq(0).sum()),
            "selected_with_surface_parking": int(sample["surface_parking_spaces"].notna().sum()),
            "file": f"{city}.csv",
        }
    manifest = {
        "sample_size_per_city": size,
        "seed": seed,
        "method": "Uniform sample without replacement using pandas sample(random_state=seed), after sorting by property_id.",
        "sampling_unit": "One CoStar Multifamily property/community, potentially containing multiple physical buildings.",
        "base_eligibility": "propertyType == 11; market 14 or 20; finite valid coordinates; positive units; construction event 2 year; known buildingClass in 1..4.",
        "outcome_eligibility": "Houston requires maxRent > 0 and reported vacancyPercent in [0, 100]. Los Angeles does not require either outcome.",
        "missing_values": "Blank CSV cells are unknown, never measured zero. Vacancy zero and explicitly reported zero surface spaces are valid.",
        "year_built_rule": "Earliest constructionEvents[].year with eventType == 2 (ConstructionComplete); exclude renovations.",
        "surface_parking_types": {"1": "Surface", "2": "SurfaceTandem"},
        "parking_validation": "Use only rows with known surface_parking_spaces; parking completeness does not determine sample selection.",
        "field_sources": FIELD_SOURCES,
        "cities": cities,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for city, sample in samples.items():
        sample.to_csv(output_dir / f"{city}.csv", index=False, na_rep="")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Generate samples using optional size and seed arguments; print counts only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_samples(size=args.size, seed=args.seed)
    for city, counts in manifest["cities"].items():
        print(f"{city}: selected {counts['selected_count']} of {counts['eligible_count']} eligible properties")


if __name__ == "__main__":
    main()
