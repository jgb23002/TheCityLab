# CityLab contracts

These are application contracts, not provider response field names. Existing
schema classes and fields are preserved. The canonical tools below now include
local CoStar loading, cached Vexcel requests, vision inspection, regression,
solar estimates and outreach drafts. The optional Matterport tool remains a stub.

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

`parcel_bearing_deg(info: VexcelPropertyInfo) -> float | None` (internal
helper, not a `###`-checked contract since it returns `float | None` rather
than a schema model) computes the compass bearing (0–180°, since a line has
two ends) of the selected parcel polygon's longest edge from the geometry
already present in `raw_response["features"]`. It makes no new network call.
Used by `backend.tools.solar` as the panel azimuth when a polygon is
available, replacing the earlier hardcoded due-south assumption; returns
`None` (falls back to the stated due-south assumption) for a point-only or
missing parcel geometry.

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
def estimate_solar(building: Building, vision_result: Measurement | None = None) -> SolarResult: ...
def rank_buildings(estimates: list[SolarResult]) -> list[SolarResult]: ...
```

Return schema: **SolarResult**, extending **SolarEstimate**. Existing unit fields
are preserved: usable roof area in square feet, capacity in kW, annual energy in
kWh. `score` is a future documented ranking metric and `rank` is 1-based.
`payback_years` is optional simple payback in years, with its assumptions in
`assumptions`. Missing payback is unknown and sorts after known paybacks.
Report assumptions and caveats; `assumptions.roof_area_source` names which one
was used. Roof area prefers a real measurement — `vision_result.roof_area_sq_ft`
when `vision_result.observations["source"]` is a tagged real-measurement source
(e.g. Vexcel's detected structure footprint; see `scripts/measure_sample.py`) —
else falls back to CoStar building footprint fields
(`backend.tools.geometry.roof_footprint_sq_ft`). A vision *model's* photo-read
roof area (untagged `roof_area_sq_ft`) is never used as the roof-area source:
math first, model last. `vision_result` is optional; besides a measured roof
area, it supplies only caveats a formula can't derive (existing rooftop
obstructions or visible solar panels) and `observations["parcel_bearing_deg"]`
for azimuth. Its absence never blocks an estimate.

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
provider data belongs in `metadata` / `raw_response`. Requests each room's and
floor's `dimensionEstimates` (`area`, `width`, `depth`, `height`,
`lowerElevation`, `units`); `metadata["rooms"]`/`metadata["floors"]` hold the
raw GraphQL lists. Cached by model id via `backend.cache.cached_json_call`.

`building_height_ft(model: MatterportModel) -> float | None` (internal helper,
not a `###`-checked contract since it returns `float | None`) is the median of
rooms' positive `dimensionEstimates.height`; a room whose ceiling plane
couldn't be resolved reports `height: 0.0` and is excluded, not treated as a
real zero height. Used by `backend.agent.study` as a real-measurement height
source (`SolarResult.assumptions["estimated_building_height_ft"]`) for
buildings with a known `matterport_model_id`, falling back to
`backend.tools.geometry.building_height_ft`'s CoStar-story-count math
otherwise. Height is reported context only; it is not a `payback` input.

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
| `num_stories_above_grade` | `numStoriesAboveGrade` |
| `gross_typical_floor_sq_ft` | `grossTypicalFloorSquareFeet` (the building's most common floor size, not a sum) |
| `floor_areas_sq_ft` | `floors[].floorSize.squareFeet`; a list, one entry per floor, not summed |
| `building_height_ft` | `buildingHeightInFeet`; sparse (populated on a minority of records) but a real measurement when present |
| `lot_size` | `lotSize`; CoStar pre-converts square meters to square feet, so only `lotSizeMeasurementType` 1 (SquareFeet) and 2 (Acres) occur in practice |
| `lot_size_measurement_type` | `lotSizeMeasurementType`; 0 Unknown, 1 SquareFeet, 2 Acres |
| `building_construction_material` | `buildingConstructionMaterial`; 0 Unknown, 1 Masonry, 2 Metal, 3 ReinforcedConcrete, 4 Steel, 5 WoodFrame, 6 Brownstone, 7 Limestone |

### backend.tools.geometry (internal helper, not an agent tool)

`backend.tools.geometry` is a plain-arithmetic internal module consumed by
`backend.tools.solar`; it is not itself an agent-facing tool and its functions
return `float | None`, not a schema model, so it is documented here rather than
as a `###`-checked contract section. No network, no model call. Signatures:

```python
def roof_footprint_sq_ft(building: Building) -> float | None: ...
def building_height_ft(building: Building) -> float | None: ...
def lot_size_sq_ft(building: Building) -> float | None: ...
def open_ground_sq_ft(building: Building) -> float | None: ...
```

Per the project's "math first, model last" rule, this module is the roof-area/
height source for `backend.tools.solar`, replacing the earlier vision-model-
measured roof area. `roof_footprint_sq_ft` falls back through
`gross_typical_floor_sq_ft`, then the largest of `floor_areas_sq_ft`, then
`building_size_sq_ft` divided by `num_stories_above_grade`. `building_height_ft`
prefers CoStar's own measured `building_height_ft` field when present, else
`num_stories_above_grade` times `FLOOR_TO_FLOOR_FT` (a stated 12 ft assumption,
since CoStar's ceiling-height field is null on every sampled record). Every
function returns `None`, never 0 or a guess, when its required source fields
are missing.

## Study and API schemas

**StudyResult** requires `id`, `question`, `buildings: list[Building]`,
`measurements: list[Measurement]`, `stats: StatsResult | None`,
`brief_markdown: str`, and `caveats: list[str]`. `stats` is a required key but can
be null for a retrofit-only study. Optional `solar_results: list[SolarResult]`
and `outreach_drafts: list[OutreachDraft]` support solar rankings and unsent
owner-email drafts; `is_mock` defaults to false. Join buildings to
measurements, detections and solar outputs using their `property_id` and
`Building.id`. All entity ids are required; `Detection.label` is also required.
Other entity measurements can be unknown.

| Endpoint contract | Request model | Response model |
|---|---|---|
| `POST /api/ask` | **AskRequest** | **AskResponse** |
| `POST /api/correct` | **CorrectRequest** | **CorrectResponse** |

- **AskRequest**: required nonempty `question`; optional `market_id` and
  `property_type`; `limit` defaults to 50 (1–500); `mode` is `research` (default)
  or `retrofit`. `use_mock_tools` defaults to false; true explicitly selects
  offline synthetic tools for that web request.
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

`backend/main.py` serves health, the static frontend, `GET /api/questions`,
`GET /api/mock-study`, a no-imagery/model-call `GET /api/buildings` cohort preview,
and `POST /api/ask` using **AskRequest** and **AskResponse**.
The ask route runs `backend/agent/study.py` over the selected city cohort with
the merged CoStar, Vexcel, vision, statistics, solar and outreach tools.
Research gets full Building/Measurement records; retrofit returns ranked solar
results and a reviewable owner-email draft. Brief numbers come from tool
outputs. Mock mode is explicit and question-specific, not a fixed saved answer.
Progress events record completed workflow steps and are returned with the final
result; the web API does not claim to stream live events. The CLI planner remains
available separately. No correct handler or correction/recomputation is added.
Vision results reuse `data/cache/vision/m-<property_id>.json`, compatible with
`scripts/accuracy_check.py`; `USE_CACHE_ONLY` prohibits fresh Vexcel/vision calls.
Absent cached measurements remain unknown, and no usable measurements yields a
503 instead of a synthetic fallback. Cache-only outreach uses the existing local
template. All cache and local kit files remain ignored.

## Mock study and checks

`data/showcase/mock_study.json` is a **StudyResult** containing 12 fictional
Houston apartment properties, 12 synthetic measurements and illustrative
statistics. It sets `is_mock: true`, labels the brief and caveats as synthetic,
and makes no claim that these ids, imagery, measurements or statistics are real.
The brief contains three paragraphs. This fixture is for consumers to build
against; it does not result from running any tool.

## Mock tools and agent names

`backend/tools/mock.py` implements the same eleven canonical signatures with
fixed synthetic values and schema-object returns. It reads only the committed
mock study, adds a fictional Los Angeles cohort, and makes no network/model
calls or credential reads. Mock image references are `mock://` placeholders,
not image files. Regression output is explicitly marked as fixed fake numbers;
no regression is fitted. Solar paybacks are fixed examples, not calculations.

The agent's LangChain adapters serialize schema objects to JSON and expose these
aliases; other tool names match their canonical function names:

| Agent tool | Canonical contracted function |
|---|---|
| `sample_buildings` | `costar_data.find_buildings` |
| `vexcel_detect` | `vexcel.get_property_info` (precomputed detection evidence) |
| `vision_inspect` | `vision.measure_building` |
| `run_regression` | `stats.run_study` |

`USE_MOCK_TOOLS=true` (default) selects the mock functions; false routes to the
real implementations, requiring the local kit, credentials or cached responses.
Unavailable real tools/data fail explicitly. This flag substitutes the tools,
not the chat model: normal CLI runs still use the configured provider. Tests
inject a scripted fake chat model and prohibit network access.

The installed Deep Agents 0.7.38 `create_deep_agent` signature and docstring were
inspected before integration: CityLab supplies `model`, `tools`, `system_prompt`
and `name`; the default backend is in-memory StateBackend, not the host
filesystem. `get_chat_model("chat")` selects LLM_MODEL;
`get_chat_model("vision")` selects VISION_MODEL. The no-argument form still
selects chat for compatibility.
The optional `model_id=` keyword overrides the selected model for AI-tool calls.

```sh
USE_MOCK_TOOLS=true python -m backend.agent.planner "Do greener Houston apartments have higher asking rents?"
```

```sh
pytest -q tests/test_contracts.py
pytest -q
```
