# CityLab contracts

These are application contracts, not provider response field names. Existing
schema classes and fields are preserved. All tool functions below are stubs
whose only behavior is to raise `NotImplementedError`; no file loading, network,
vision, statistical, solar or outreach logic is implemented here.

Tools return the named Pydantic models. At a JSON boundary, use
`model_dump(mode="json")` / `model_dump_json()` (per element for lists); never
return a DataFrame, image bytes, credentials or an SDK response object.

## Tool signatures

The Python declarations in these module sections are the authoritative,
machine-checked signatures, including parameter names, defaults and return types.
Every return type is a model in `backend/schemas.py`, a list of that model, or an
optional model. `None` from `get_property` will mean no matching property.

### backend.tools.costar_data

```python
def load_properties() -> list[Building]: ...
def get_property(id: int) -> Building | None: ...
def find_buildings(market_id: int, property_type: int, limit: int = 20) -> list[Building]: ...
```

Return schema: **Building**. Future loading uses local JSON Lines with
`pandas.read_json(path, lines=True)`; there is no CoStar API. Market ids come from
`primaryAddress.researchMarketId`, with Houston 14 and Los Angeles 20.

### backend.tools.vexcel

```python
def get_aerial(lat: float, lon: float, property_id: int) -> AerialImage: ...
def get_property_info(lat: float, lon: float) -> VexcelPropertyInfo: ...
```

Return schemas: **AerialImage** and **VexcelPropertyInfo**, respectively.
Imagery is represented by a saved path and metadata. Future requests use
longitude-first WKT, SRID 4326, header authentication, cached responses and
`USE_CACHE_ONLY`. Provider dotted keys remain literal keys in `attributes`,
`structures` and `raw_response`; they are not renamed into guessed fields.

### backend.tools.vision

```python
def measure_building(image_path: str, property_id: int) -> Measurement: ...
```

Return schema: **Measurement**, extending the existing **VisionResult** fields.
`Measurement.detections` contains **Detection** models. Detection `bbox` is a
four-number `[left, top, right, bottom]` box in source-image pixels; optional
`geometry_wkt` is a map outline when available. Confidence is 0–1; cover values
are percentages 0–100; parking counts are counts of spaces, not vehicles.
`roof_area_sq_ft` is square feet. Record the measurement area/basis and evidence
in inherited `observations`; parcel and 80 m image-square percentages are not
interchangeable. Unknown values remain null, not zero.

### backend.tools.solar

```python
def estimate_solar(building: Building, vision_result: Measurement) -> SolarResult: ...
def rank_buildings(estimates: list[SolarResult]) -> list[SolarResult]: ...
```

Return schema: **SolarResult**, extending **SolarEstimate**. Existing unit fields
are preserved: usable roof area in square feet, capacity in kW, annual energy in
kWh. `score` is a future documented ranking metric and `rank` is 1-based. Report
assumptions and caveats; a floor-area control is not a measured roof area.

### backend.tools.stats

```python
def run_study(question: str, buildings: list[Building], measurements: list[Measurement] | None = None) -> StatsResult: ...
```

Return schema: **StatsResult**, extending **Study**. The optional measurements
argument makes image-derived predictors available while preserving two-argument
calls for studies using CoStar attributes alone. `sample_size` is the analyzed
building count. `coefficients`, `p_values` and `confidence_intervals` are keyed by
the same term names; each interval is `[lower, upper]`. `controls` names the
actual controls used, with building class treated categorically when included.
Missing results remain null. No outcome, regression result or causal claim is
invented by a tool implementation.

### backend.tools.outreach

```python
def draft_outreach(building: Building, solar_estimate: SolarResult) -> OutreachDraft: ...
```

Return schema: **OutreachDraft**, with optional property id, subject,
`body_markdown` and caveats. This is a draft for review; there is no sending API.

### backend.tools.matterport

```python
def get_model(model_id: str) -> MatterportModel: ...
```

Return schema: **MatterportModel**. This optional tool is included so that every
tool stub in the package is covered. Its `id` is the Matterport model id and its
provider data belongs in `metadata` / `raw_response`.

## Building field mappings

Existing `Building` fields (`id`, `primary_address`, `property_type`, `market_id`,
`latitude`, `longitude`, `matterport_model_id`) retain their names and types.
The following optional application fields are added; these mappings were checked
against the local CoStar model docs, and are not new source-field claims:

| Application field | Documented CoStar source / meaning |
|---|---|
| `name` | commercial-property `buildingName` |
| `building_size_sq_ft` | `grossBuildingAreaSquareFeet` (GBA, not footprint or roof area) |
| `year_built` | earliest `constructionEvents[].year` with `eventType == 2` (ConstructionComplete), not renovation |
| `building_class` | `buildingClass`; 0 Unknown, 1 A, 2 B, 3 C, 4 F |
| `total_num_units` | `propertyApartment.totalNumUnits` (units counted across active apartment models) |
| `max_rent` | apartment-listing `maxRent`, dollars/month, joined on `propertyId == Building.id`; zero means unreported |
| `vacancy_percent` | `propertyApartment.vacancyPercent`; a reported zero is valid |
| `surface_parking_spaces` | sum of `parking[].totalSpaces` for types 1 Surface and 2 SurfaceTandem; absent/incomplete counts are unknown |

## Study and API schemas

**StudyResult** requires `id`, `question`, `buildings: list[Building]`,
`measurements: list[Measurement]`, `stats: StatsResult | None`,
`brief_markdown: str`, and `caveats: list[str]`. `stats` is a required key but can
be null for a retrofit-only study. Optional `solar_results: list[SolarResult]`
supports solar outputs; `is_mock` defaults to false. Join buildings to
measurements, detections and solar outputs using their `property_id` and
`Building.id`. All entity ids are required; `Detection.label` is also required.
Other entity measurements can be unknown.

| Future endpoint | Request model | Response model |
|---|---|---|
| `POST /api/ask` | **AskRequest** | **AskResponse** |
| `POST /api/correct` | **CorrectRequest** | **CorrectResponse** |

- **AskRequest**: required nonempty `question`; optional `market_id` and
  `property_type`; `limit` defaults to 50 (1–500); `mode` is `research` (default)
  or `retrofit`.
- **AskResponse**: required `study: StudyResult`; `events: list[AgentEvent]`
  defaults to an empty list.
- **CorrectRequest**: required `study_id`, `measurement_id` and
  `corrections: MeasurementCorrection`.
- **MeasurementCorrection**: optional `roof_area_sq_ft`, `green_cover_percent`,
  `paved_surface_percent`, `parking_space_count`, `detections`, and `note`.
  Serialize and apply patches with `model_dump(exclude_unset=True)` to distinguish
  omitted values (unchanged) from explicit null (clear to unknown).
- **CorrectResponse**: the same `study` and `events` fields as AskResponse, plus
  required `corrected_measurement: Measurement`; the updated study will carry
  that measurement and refreshed statistics/brief.

These are schemas only. `backend/main.py` still implements only health and static
files; no ask/correct handlers or correction/recomputation logic are added.
`backend/cache.py` remains a non-tool helper with its existing JsonValue/None
signatures; persistence is outside this contract task.

## Mock study and checks

`data/showcase/mock_study.json` is a **StudyResult** containing 12 fictional
Houston apartment properties, 12 synthetic measurements and illustrative
statistics. It sets `is_mock: true`, labels the brief and caveats as synthetic,
and makes no claim that these ids, imagery, measurements or statistics are real.
The brief contains three paragraphs. This fixture is for consumers to build
against; it does not result from running any tool.

```sh
pytest -q tests/test_contracts.py
pytest -q
```
