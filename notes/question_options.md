# Question options by city and property type

Generated with `python scripts/pick_questions.py` using local CoStar JSON Lines loaded by `pandas.read_json(path, lines=True)`. Counts are unique property ids, not listing or space rows.
Read alongside `notes/data_summary.md`, `data/costar/models/commercial-property.md`, `apartment-listing.md`, `commercial-listing-summary.md` and `location-occupancy.md`.

## Selected property types

Top three types are chosen independently in each research market before outcome/control filtering. Ties use the smaller numeric propertyType code, giving exactly three types per city.

| City | Market | Property type | All properties |
|---|---:|---|---:|
| Houston | 14 | Multifamily (11) | 1348 |
| Houston | 14 | Office (5) | 191 |
| Houston | 14 | Industrial (2) | 98 |
| Los Angeles | 20 | Multifamily (11) | 1417 |
| Los Angeles | 20 | Office (5) | 98 |
| Los Angeles | 20 | Industrial (2) | 24 |

Los Angeles: GeneralRetail (6) ties at 24 properties but falls after the third type under the stated tie rule.

## Eligibility and outcome definitions

- All rows require finite `primaryAddress.latitude` and `primaryAddress.longitude`, positive `grossBuildingAreaSquareFeet` (GBA, square feet), and a recorded `constructionEvents[].year` with `eventType == 2` (ConstructionComplete). Use the earliest completion year; renovation years never substitute. No unit-count or building-class filter is applied.
- Building size uses GBA consistently, rather than mixing gross/rentable/leasable areas. The controls-only rows show the maximum cohort available before choosing a rent outcome.
- Apartment rent is the mean of the nonzero, nonmissing `minRent` and `maxRent` from `apartment-listing.jsonl`, joined by `propertyId` to property `id`. One reported bound suffices; both zero/missing means unavailable. This is a monthly asking-range midpoint, not an achieved or unit-weighted rent. The model doc treats Unknown currency on these US communities as USD; unit/bed pricing bases still need care in research.
- Commercial rent requires an explicitly per-square-foot lease quote: `forLeaseSummary.askingRateMeasurementType == 1`, with `askingRateLow`/`askingRateHigh` and `askingRateBasis` (1 Monthly, 2 Yearly), or a space's `rateLow`/`rateHigh` and `rateBasis`. Nonzero bounds are averaged and monthly rates multiplied by 12. Unknown time/area units and per-acre/per-meter quotes are excluded.
- A single-property rollup joins on listing `propertyId`. Group/portfolio rollups are not assigned to each member: only space quotes with their own `forLeaseSummary.spaces[].propertyId` qualify. A missing space propertyId falls back to the listing propertyId only for an unambiguous single-property listing.
- Commercial counts mean at least one attributable quote, counted once per building. They include Active and OffMarket records and off-market/pending space detail in this kit, so describe recorded quote coverage, not a current-market rent estimate. No rent averages are pooled across listings, currencies, serviceType or direct/sublet flavors; future analysis must separate/normalize those and area-weight any aggregation across spaces.

## Ranked options

Sorted globally by the final count, largest first. Zero means no qualifying rent outcome for that city/type; it does not mean observed zero rent.

| City | Property type | Outcome | Buildings with lat/lon + outcome + size + year built |
|---|---|---|---:|
| Los Angeles | Multifamily (11) | Controls only: building size + year built | 1414 |
| Houston | Multifamily (11) | Controls only: building size + year built | 1346 |
| Houston | Multifamily (11) | Apartment asking rent (monthly midpoint) | 977 |
| Los Angeles | Multifamily (11) | Apartment asking rent (monthly midpoint) | 749 |
| Houston | Office (5) | Controls only: building size + year built | 191 |
| Houston | Office (5) | Commercial asking rent per SF (annualized) | 115 |
| Houston | Industrial (2) | Controls only: building size + year built | 98 |
| Los Angeles | Office (5) | Controls only: building size + year built | 98 |
| Los Angeles | Multifamily (11) | Commercial asking rent per SF (annualized) | 63 |
| Los Angeles | Office (5) | Commercial asking rent per SF (annualized) | 48 |
| Los Angeles | Industrial (2) | Controls only: building size + year built | 24 |
| Houston | Industrial (2) | Commercial asking rent per SF (annualized) | 21 |
| Los Angeles | Industrial (2) | Commercial asking rent per SF (annualized) | 8 |
| Houston | Multifamily (11) | Commercial asking rent per SF (annualized) | 3 |
| Houston | Industrial (2) | Apartment asking rent (monthly midpoint) | 0 |
| Houston | Office (5) | Apartment asking rent (monthly midpoint) | 0 |
| Los Angeles | Industrial (2) | Apartment asking rent (monthly midpoint) | 0 |
| Los Angeles | Office (5) | Apartment asking rent (monthly midpoint) | 0 |

## Occupancy/vacancy outcome: unsupported from location-occupancy alone

The model describes tenancy windows (`fromDate`/`toDate`) and occupied space (`spaces[].squareFeetOccupied`), not a ready property occupancy/vacancy rate. A current occupied-area sum does not establish complete, non-overlapping tenant coverage or a compatible total inventory denominator. Restricted/suppressed-space rules also affect aggregate SF. The doc explicitly says occupied area is not a vacancy numerator.
No occupancy/vacancy rate is fabricated, and no numeric coverage row is generated for that unsupported outcome. `propertyApartment.vacancyPercent` and marketed available area are different sources and are not substituted for the requested location-occupancy outcome. The tenancy file is not aggregated for this report.
