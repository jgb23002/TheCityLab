# CityLab research agent (CLI planner)

`backend/agent/planner.py` runs a Deep Agents orchestrator that answers one research
question and writes a markdown brief in which every number is cited to a tool result.

## Tool modes: two separate switches

| Setting | Used by | Values |
|---|---|---|
| `TOOLS_MODE` | planner CLI (`backend/agent/registry.py`) | `auto` (default, also when blank), `real`, `mock` |
| `USE_MOCK_TOOLS` | web study flow (`backend/agent/study.py`, `POST /api/ask`) | `true` / `false` |

- `auto`: use the real tool, falling back to `backend/tools/mock.py` only if the real
  one raises `NotImplementedError` (logs `REAL TOOL NOT READY, USING MOCK: <tool>`).
- `real`: real tools only; failures surface as errors.
- `mock`: synthetic tools only; the brief must say "(demo data)" next to those results.
- `--mode` on the CLI overrides `TOOLS_MODE`. Neither setting changes the chat model.

## Subagents (in order)

1. **sampler**: `sample_buildings` once (n defaults to 40, max 60).
2. **inspector**: once per building: `vexcel_detect`, then `vision_inspect` (the tool
   fetches the aerial image itself). Image measurements only, never rent.
3. **statistician**: `run_regression` once; copies the result's values exactly.
4. **writer**: `list_evidence` / `read_evidence`, then the brief (Question, Answer,
   Method, Result, Confidence and caveats).

Every tool call is recorded in a per-run evidence ledger as `E1`, `E2`, ...
After the run, a guard checks that `sample_buildings`, at least one `vision_inspect` and
`run_regression` succeeded. If one is missing, it asks the orchestrator once to run it,
then adds the caveat "Study incomplete: <step> was not run."

## Citation rules (`backend/agent/citations.py`)

- Every number is followed by its tag, e.g. `$12.40 [E7]`; the tag must exist in the
  ledger and that entry's JSON must contain the number (1% tolerance, rounding to the
  digits written, or a x100 percent form).
- Not checked: years 1900-2100, list enumerators (`1.`, `1)`, `(1)`), "95% CI", and
  the methodology constants in `METHOD_CONSTANT_PATTERNS` (e.g. "80 m").
- On failure the writer gets one fix attempt; remaining problems are appended as
  "Unverified numbers" and the run is marked unverified.

## Cache

All provider calls cache under `data/cache/` (git-ignored). Real vision measurements
are reused from `data/cache/vision/m-<property_id>.json` (same format as
`scripts/accuracy_check.py` and the web study), so a repeat run makes no model call.
Errors and mock results are never cached. With `USE_CACHE_ONLY=true`, a missing cached
measurement is an error instead of a fresh call.

## Commands

```sh
python -m backend.agent.planner "In Houston, do apartment buildings with more parking spaces have higher asking rent?" --mode mock
python -m backend.agent.planner "<question>" --mode real --n 40 --json data/cache/runs/out.json
pytest -q
```

Keep `--json` output outside the repo or under `data/cache/`. The CLI exits with 0 on
`CITATION CHECK: PASS`, 2 on FAIL, and 1 on an error.
