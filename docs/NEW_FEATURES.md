# New features (Arohi): heat + cool roofs, lot fullness, Empty Lot Planner, change over time

From `CityLab_New_Feature_Ideas` (Nov 14). Every function returns plain JSON-serializable
dicts, so `backend/schemas.py` is unchanged. Everything caches under `data/cache/` and
works with `USE_CACHE_ONLY=true` (planner text falls back to a template offline).

Run all of it on real buildings and cache the results:

```sh
python -m scripts.feature_demo 23553 23564 23693        # Houston offices
python -m scripts.feature_demo 850411 275360 --no-detect # change over time examples
```

Results land in `data/cache/features/<id>.json`.

## For Jiya: how to wire it

| Step | Call | Gives |
|---|---|---|
| Vexcel facts | `roof_facts.roof_facts(vexcel.get_property_info(lat, lon))` | measured roof footprint, material, AC units/vents, existing solar, parcel tree %, paving %, vehicle count, photo date |
| Detection | `vexcel_detect.detect(lat, lon, ["parking space", "car"], id)` | boxes for Claude to check |
| Vision | `vision.inspect(run.image_path, id, run)` | now also `observations.roof_color`, `cars_seen`, `lot_fullness_percent` |
| 1. Lot fullness | `lot.lot_fullness(building, measurement, facts, capture_date)` | `% full`, flag `leased but looks empty` / `matches` / `differs` / `weekend photo, not judged`, card sentence |
| 1. Study | `lot.fullness_vs_vacancy(rows)` | Pearson/Spearman across buildings (weekday photos only) |
| 2. Planner | `lot.plan_empty_lot(building, fullness)` | spare spaces, spare land, AB 2097 / Houston rule check, ranked options with $, Bedrock recommendation |
| 3. Heat | `heat.rank_heat(buildings, measurements, facts_by_id)` | heat score 0-100, cool-roof cost/savings/payback, card sentence |
| 3. Retrofit | `heat.compare_retrofits(building, measurement, facts, plan["carport"])` | roof solar vs solar carport vs cool roof, by payback |
| 4. Change | `change.compare_dates(building)` / `change.summarize_changes(rows)` | before/after images, new solar, construction, demolition, lot fullness then vs now |

`capture_date`: detection and `get_aerial` use the newest capture, so date the photo with
the newest `urban`/`urban-r` entry of `vexcel_detect.list_collections(lat, lon)`.

## Guards (found on real buildings)

- Vexcel "parking space" boxes often outline whole lots, so Claude's stall count is used.
- Lot fullness only compares cars and spaces from the same area: an ~80 m photo is never
  divided by CoStar's whole-property space count. No visible lot means "not judged".
- Weekend photos are not flagged; holiday-week photos carry a caveat.
- `vacancy_percent` is only filled for apartments, so offices get fullness but no flag yet.

## Assumptions (all returned with results)

Cool roof: $0.75-1.50/sq ft (EPA), $0.20/sq ft extra over a normal re-coat, 0.30 / 0.15 / 0.05
kWh saved per sq ft per year for dark / medium / light roofs. Planner: 25% buffer over cars,
30 m² per space, rent $60 (Houston) / $120 (LA) per month at 50% take-up minus 20% app fee,
$300 per EV port per year, carport 3.5 kW per space at $3.00/W, greening $8/sq ft, Houston
drainage $0.032/sq ft/yr. Transit: rail stations in `data/transit/` (LA Metro GTFS, OSM for
Houston); Houston no-minimum areas are approximate boxes. "Likely eligible", not legal advice.
