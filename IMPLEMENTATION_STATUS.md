# CityLab implementation status

Audit and completion pass, 2026-11-14, branch `jiya`. Every statement below was checked by
running code, tests or live calls. "Verified" means it actually ran.

## How the pieces connect

```
frontend (plain JS, Leaflet, Chart.js, marked)
  └─ POST /api/ask-stream (SSE) · /api/ask-agent · /api/correct · /api/calibration · /api/image/{id}
       └─ backend/agent/agent_api.py ── replay (backend/agent/showcase.py) when USE_CACHE_ONLY or ?replay=true
            └─ planner.run_question ── DeepAgents create_deep_agent (orchestrator + subagents)
                 └─ registry tools ── backend/tools/* (CoStar kit, Vexcel, vision, stats, solar, outreach, Matterport)
                      └─ evidence Ledger (E1, E2, ...) ── citations.verify_brief ── StudyResult
```

The older `/api/ask` route (`backend/agent/study.py`) is a fixed, non-agent pipeline and is
unchanged.

## Audit: before this pass

| Area | State found | Evidence |
|---|---|---|
| Planner (DeepAgents) | Working: orchestrator + sampler/inspector/statistician/writer (research) and solar/outreach (retrofit) subagents, real tool calls, live SSE | tests + recorded live runs |
| Visible to-do plan | Missing. deepagents 0.7.38 adds no to-do tool by default; the agent had no `write_todos` | `deepagents/graph.py`, recorded runs had no plan |
| CoStar market data | Working for samples (local kit, `backend/tools/costar_data.py`) | tests, live runs |
| Single property / address | Missing (no intent, no lookup tool) | code trace |
| Owners / tenants | Missing | code trace |
| Vexcel + vision | Working: imagery, precomputed attributes, detections, vision JSON validated by pydantic, cached | tests, `scripts/measure_vision_sample.py` |
| Matterport | Stub (`NotImplementedError`) | `backend/tools/matterport.py` |
| Statistics | Working: OLS + HC3, controls size/age, listwise deletion, VIF | `tests/test_stats.py` |
| Rooftop solar + payback | Working (PVWatts when configured, regional fallback otherwise) | `tests/test_solar.py` |
| Other upgrades (carport, LED, window film, HVAC) | Missing | code trace |
| Reports / email | Research brief, retrofit brief, owner email (drafts only) | tests |
| Corrections | Working for research studies only; retrofit studies returned 422 | `backend/agent/correction.py` |
| Calibration from corrections | Missing | code trace |
| Frontend | Single-column page: progress, map, cards, ledger, chart, brief; no inspection panel, one chart | code + headless DOM run |
| Office vacancy (Scenario 1) | Not available in data: CoStar stores vacancy only for multifamily (`propertyApartment.vacancyPercent`); the sampler silently returned nothing | CoStar model docs, kit query (191 Houston offices, 0 with vacancy) |

## Implemented in this pass

| Feature | Files |
|---|---|
| Matterport client: GraphQL with Basic auth, cached (`data/cache/matterport`), `USE_CACHE_ONLY` honoured, PropertyId → model id from `commercial-properties.csv`; returns measured floors/rooms/ceiling heights only | `backend/tools/matterport.py` |
| `interior_inspect` tool: skips cleanly without a scan; lighting, windows and HVAC are reported as not observed (the API exposes no interior images) | `backend/agent/registry.py` |
| Property mode: `lookup_property` (CoStar id or street address), owners and property manager (`contacts[]` → company-detail), current tenants (location-occupancy), unavailable fields listed | `backend/agent/property_lookup.py`, `registry.py`, `planner.py`, `prompts.py` |
| Upgrade screening: rooftop solar (from the solar tool), solar carport (from measured parking, every assumption labelled with its source), LED / window film / HVAC marked "not estimated" with the missing input; ranked by simple payback | `backend/agent/upgrades.py` |
| Planner plan: LangChain `TodoListMiddleware` → real `write_todos`; plans and `write_file` notes are streamed and saved as events | `planner.py`, `events.py` |
| Scenario 1 honesty: sampling reports `missing_by_field` and an `unavailable_reason`; units are required only for multifamily | `registry.py` |
| Corrections for retrofit and property studies: deterministic rooftop re-estimate (no network), re-rank, re-screen upgrades, template re-draft of the email, regenerated cited ranking table | `backend/agent/correction.py` |
| Calibration from correction history (`GET /api/calibration`): per field, correction count, median relative correction, reliability flag, confidence multiplier. No retraining is claimed | `backend/agent/study_store.py`, `agent_api.py` |
| Provenance: per-building live / cached / demo / replay labels; replays are marked as recorded runs (SSE `replay` frame and `X-CityLab-Replay` header) | `events.py`, `agent_api.py`, `frontend/app.js` |
| Dashboard: activity panel (plan, delegations, tool calls), map (colour by measure, filter), building inspector (CoStar record, owners, aerial image with confirmed boxes, confidence, calibration, editable parking / green cover / roof area, interior, solar), upgrades table with assumptions, second chart, copy / download / print-to-PDF for reports and email | `frontend/index.html`, `app.js`, `style.css` |
| Offline showcase extended with the three judging scenarios (S1-S3) | `scripts/build_showcase.py`, `data/showcase/results/` |

## Verification results

See the bottom of this file (filled in after the final runs).

## Known limitations and external blockers

- **Office vacancy and rent:** the CoStar kit records vacancy and asking rent only for
  multifamily properties, and the frozen `Building` contract maps those fields to the
  apartment sources. Scenario 1 therefore ends "study incomplete" with that reason, which is
  the correct result. Office availability from listings is not the same measure as vacancy
  and was not substituted.
- **Matterport interior imagery:** the Model API used here returns measured floors and rooms
  but no image field (schema introspection is blocked), so lighting, windows and HVAC are not
  assessed. Ceiling height is shown only when the scan measured it.
- **LED, window film and HVAC:** never estimated. They need a fixture inventory, glazing area
  and equipment data that this pipeline does not have.
- **Simple payback** ignores incentives, financing, demand charges and operating costs. It is
  a screening number, not a feasibility study.
- **Parking counts** come from the area around the address point (about 80 to 100 m). On
  large communities they undercount (see `notes/accuracy.md` on the `ai-phase3` branch).
- **No linters or type checker** are configured in the project; ruff was run from a scratch
  folder for undefined names and unused imports only.
