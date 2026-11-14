# CityLab showcase

The three questions and their source fields are defined in `questions.json`.
`../samples/houston.csv` and `../samples/los_angeles.csv` each contain 50 distinct
Multifamily properties (`propertyType == 11`), drawn without replacement with
seed 42. `../samples/manifest.json` records eligibility counts, field mappings,
sampling rules, and validation coverage.

Both cities require coordinates, positive `propertyApartment.totalNumUnits`, a
ConstructionComplete year (`constructionEvents[].eventType == 2`), and a known
`buildingClass` (1–4). The year control uses the earliest completion year, never
a renovation year. Building class is a categorical control, not a numeric score.

Houston (market 14) requires both non-zero `apartment-listing.maxRent` and reported
`propertyApartment.vacancyPercent`. Zero vacancy is valid. The same 50 properties
support Q1 and Q2. Los Angeles (market 20) requires neither outcome; optional
rent/vacancy values are retained when known for the Q3 retrofit cohort.

For Q2 validation, `surface_parking_spaces` sums `parking[].totalSpaces` only for
types 1 (Surface) and 2 (SurfaceTandem). Covered parking, garages, reserved spaces
and other ambiguous types are excluded. Missing or incomplete surface counts
remain blank, and only known counts can validate AI parking-space measurements.
Parking completeness does not filter the 50-property research sample.

These are properties/communities, which can contain multiple physical buildings.
The kit is a selected set of properties in two cities, not a census of either
research market. Q3 will rank ten candidates within the sampled Los Angeles
cohort after aerial measurement and solar estimation. Green-cover, paved-area,
AI parking and solar measurements have not been computed. Research should report
the imagery/outcome dates and use consistent imagery areas and rent pricing
bases; `maxRent` is the highest asking rent, not typical or achieved rent.

Regenerate from the repository root with the local CoStar kit installed:

```sh
python scripts/prepare_showcase.py --size 50 --seed 42
pytest -q
```
