# CityLab research agent (CLI planner)

`backend/agent/planner.py` runs a Deep Agents orchestrator that answers one question and
writes a markdown brief in which every number is cited to a tool result (`E1`, `E2`, ...).

## Switches

| Setting | Used by | Values |
|---|---|---|
| `TOOLS_MODE` / `--mode` | planner CLI and `POST /api/ask-agent` | `auto` (default), `real`, `mock` |
| `--intent` | planner CLI (`AskRequest.mode` on the API) | `auto` (default), `research`, `retrofit` |
| `USE_MOCK_TOOLS` | web study flow (`backend/agent/study.py`, `POST /api/ask`) | `true` / `false` |

- `auto` mode uses real tools and falls back to `backend/tools/mock.py` only on
  `NotImplementedError`. `mock` needs no keys. Mock results are labelled "(demo data)".
- `auto` intent picks retrofit when the question mentions solar, rooftop, retrofit or
  "best candidates", else research.

## Flows (subagents, in order)

**Research:** sampler (`sample_buildings`; complete cases only) -> inspector
(`vexcel_detect`, `vision_inspect`) per building -> statistician (`run_regression` once,
over the inspected buildings joined by property_id) -> writer.
The regression (`backend/tools/stats.py`) is statsmodels OLS with HC3 errors. Controls:
building size and year built, plus units and building class (categorical) when present.
A control that is >90% missing or collinear (VIF > 10; the later one is dropped) is
named in the caveats. Rows with missing values are dropped (listwise) and counted;
`sample_size` is the number of rows analyzed. Fewer rows than parameters + 5 is a failure.

**Retrofit (no regression):** sampler -> inspector (needs `roof_area_sq_ft`) -> solar
(`solar_payback` per inspected building, then `rank_buildings` once) -> outreach
(`draft_email` for the top N; N comes from the question, default and maximum 10) ->
retrofit writer (Question, Answer, Ranking table, Assumptions, Caveats, Next step).
Solar uses only the image-measured roof, never floor area. Buildings with no measured roof
are skipped and counted (`skipped_no_roof`). Drafts are never sent.

**Completeness guard:** research needs a regression with a coefficient and sample size;
retrofit needs a successful ranking and at least one email draft. If something is missing,
the orchestrator is asked once to run it. If it is still missing, the brief gets "Study
incomplete: ..." and the CLI prints `CITATION CHECK: FAIL (study incomplete)`.

## Citation rules (`backend/agent/citations.py`)

- Each number is followed by its tag, e.g. `$12.40 [E7]`. The tag must exist and that
  entry's JSON must contain the number (1% tolerance, rounding to the digits written, or
  x100 percent form). Ids and other whole numbers with 5+ digits must match exactly.
- A markdown table row is one unit: a tag in the row covers its numbers.
- Not checked: years 1900-2100, list enumerators, "95% CI", `METHOD_CONSTANT_PATTERNS`
  (e.g. "80 m"), ranks 1-10 in a table's "Rank" column, and the `## Question` body.
- The writer gets one fix attempt; problems left after that are listed as "Unverified numbers".

## Cache

Provider calls cache under `data/cache/` (git-ignored). Real vision measurements are reused
from `data/cache/vision/m-<property_id>.json`, so a repeat run makes no vision model call
(real outreach drafts still call the model). Errors and mocks are never cached.
`USE_CACHE_ONLY=true` turns a cache miss into an error.

## Measuring the sample

`python scripts/measure_sample.py --city houston|los_angeles [--n 50] [--workers 3] [--dry-run]`
measures each building in `data/samples/<city>.csv` once: Vexcel attributes, aerial
image, element detections (parking space, car, tree, pool, solar panel), then the vision
measurement. It saves `data/cache/vision/m-<id>.json`, which the agent, the web study and
the accuracy check all reuse. The run can be resumed: cached buildings are skipped,
failures are retried once and then recorded in `data/cache/measure_status.json` (rerun
them with `--retry-errors`), and errors or mocks are never cached. `--dry-run` estimates
paid calls without making any. With `USE_CACHE_ONLY=true` it only reports what is cached.

## Streaming

`POST /api/ask-stream` takes the same body as `/api/ask-agent` and returns server-sent
events (`text/event-stream`). `handoff` and `tool_call` frames arrive live, then
`citation_check`, then `evidence` (`[{id, tool, source, summary}]`) and `result` (the full
AskResponse). A failure sends one `error` frame and closes the stream. `: keepalive`
comments arrive every 15 s. Frames never contain tool arguments, raw Vexcel responses or
secrets. `GET /api/image/<id>` serves the building's aerial JPEG (`data/showcase/images/`,
then `data/cache/vexcel/images/`), or a placeholder SVG for ids in the mock cohort (990001-990012, 990101-990112).

```sh
curl -N -X POST localhost:8000/api/ask-stream -H "Content-Type: application/json" \
  -d '{"question":"Which 10 Los Angeles apartment buildings are the best rooftop-solar candidates?","mode":"retrofit","use_mock_tools":true}'
```

## Commands

```sh
python -m backend.agent.planner "In Houston, do apartment buildings with more parking spaces have higher asking rent?" --mode real --n 20 --json data/cache/runs/research.json
python -m backend.agent.planner "Which 10 Los Angeles apartment buildings are the best rooftop-solar candidates?" --mode mock
```

Keep `--json` output outside the repo or under `data/cache/`. Exit codes: 0 PASS, 2 FAIL, 1 error.
