This is Team Cook 

CityLab is our HackPrinceton Fall 2026 building research project. It combines
Vexcel aerial imagery and local CoStar data to investigate building questions
and identify rooftop-solar candidates. The web interface runs the merged data,
vision, regression, solar and outreach tools and displays a brief, completed
study steps, map, chart, evidence table, solar shortlist and owner-email draft.

## Run the frontend

1. Open a terminal in the project folder. If needed, create a Python 3.11 virtual
   environment with `python3.11 -m venv .venv`.
2. Activate the virtual environment:

   Mac or Linux:

   ```sh
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the existing project dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Start the server:

   ```sh
   uvicorn backend.main:app --reload
   ```

5. Open http://127.0.0.1:8000, choose a sample question, city, study type, data
   source and number of buildings, then click **Find the answer**.
   Hard-refresh the page if your browser still shows the old placeholder.

## What the page does now

The page streams a real DeepAgents run from `POST /api/ask-stream`:

- **Left, agent activity:** the planner's to-do list (DeepAgents `write_todos`), each
  subagent delegation, every tool call, and the citation check.
- **Centre, map:** sampled buildings, coloured by a chosen measure, with a filter.
- **Right, building inspection:** the CoStar record (owners and tenants for single
  properties), the aerial image with the boxes the vision model confirmed, confidence,
  editable measurements, the interior scan and solar estimates.
- **Below, analytics and reports:** two charts, the cited research brief, retrofit proposal
  or property report (copy, download, print to PDF), the upgrade table with every
  assumption, the solar ranking, the owner email draft (never sent) and the evidence ledger.

Three kinds of question work:

- **Research:** for example, Q1 and Q2.
- **Retrofit:** Q3, or "best rooftop solar" questions.
- **Single property:** a CoStar property id or a street address, for example "Evaluate
  property 23453 for building upgrades".

Correcting a measurement (parking spaces, green cover or roof area) re-runs the statistics,
or the solar estimates, ranking, upgrades and email, with no model call. Corrections are
saved under a new study id, and correction history feeds `GET /api/calibration`.

**Offline demo:** `USE_CACHE_ONLY=true uvicorn backend.main:app` replays the recorded
showcase runs (Q1-Q3 and the judging scenarios S1-S3) with no network calls. Replays are
labelled as recorded runs. See `docs/AGENT.md` and `IMPLEMENTATION_STATUS.md`.

## Older details

Open **Preview selected buildings** to see the actual CoStar sample immediately,
without starting an imagery or model call. Changing city or data source refreshes
that preview.

Press Ctrl+C in the terminal to stop the server. Fonts, Leaflet, Chart.js and map
tiles load from the internet. If those libraries cannot load, the brief,
statistics and evidence table still work, with a message in the map/chart area.

### Real data and mock demo

**Real data** uses the committed 50-building Houston or Los Angeles CoStar sample
from `data/samples/`. If the local CoStar kit is available in `data/costar/`, the
merged CoStar loader enriches those same property ids with full building fields.
Existing cached vision measurements are reused. Fresh measurements call Vexcel
and the configured vision model, so copy `.env.example` to `.env` if needed and
configure `VEXCEL_API_KEY`, `LLM_PROVIDER`, `VISION_MODEL` and provider credentials.
Start with a small building count while checking your configuration. No keys or
local kit/cache files belong in Git.

Q1 runs the green-cover/rent regression; Q2 runs paved-surface/vacancy regression,
retaining reported zero vacancy. Both use full Building records for the units,
year-built and categorical class controls. Q3 measures Los Angeles roofs, ranks
solar payback, shows up to ten known-payback candidates and drafts an email for
the top candidate. Emails are drafts only and are never sent.

**Mock demo** runs the same question-specific workflow with synthetic Houston or
Los Angeles tools. It needs no keys, model calls, network access or CoStar kit;
all results are labeled synthetic. It no longer returns one fixed Houston
answer for every question. `GET /api/mock-study` remains a saved fixture preview.

The web workflow is in `backend/agent/study.py`. It calls the merged typed tools
directly and builds the brief only from returned numbers. The existing Deep
Agents CLI remains available for conversational planning. The web request's
`use_mock_tools` flag selects its data source; `USE_MOCK_TOOLS` still controls that
workflow. The dashboard itself uses the agent routes below.

To prohibit all Vexcel and vision calls and use existing measurements only:

```sh
USE_CACHE_ONLY=true uvicorn backend.main:app --reload
```

Missing measurements stay unknown. If none are usable, the API returns a clear
503 error instead of substituting fake data. Solar uses the existing documented
fallback yield when PVWatts is unavailable, and cache-only outreach uses the
merged tool's local template without a model call.

### API routes

| Route | What it returns |
| --- | --- |
| `GET /api/health` | `{"ok": true}` |
| `GET /api/questions` | The three existing showcase questions and their metadata |
| `GET /api/mock-study` | The validated, saved mock `StudyResult` |
| `GET /api/buildings?market_id=14&limit=50` | The selected CoStar cohort without imagery/model calls; `use_mock_tools=true` previews synthetic records |
| `POST /api/ask` | A question-specific `AskResponse` containing a `StudyResult` and actual completed-step events (fixed pipeline, no agent) |
| `POST /api/ask-agent` | The DeepAgents planner's `AskResponse` (research, retrofit or single property) |
| `POST /api/ask-stream` | The same, as server-sent events: plan, handoff, tool_call, details, citation_check, evidence, result |
| `POST /api/correct` | Applies a measurement correction to a saved study and recomputes it (no model call) |
| `GET /api/calibration` | Correction-history calibration per measured field |
| `GET /api/image/{id}` | A building's aerial image (showcase, then cache) or a demo placeholder |

`POST /api/ask` accepts the existing `AskRequest` contract, for example
`{"question": "Do greener Houston apartments have higher asking rents?", "market_id": 14, "limit": 12, "use_mock_tools": true}`.
Use `mode: "retrofit"` and `market_id: 20` for the Los Angeles solar workflow.
Invalid request bodies return 422; missing preview files return 404; unavailable
analysis returns 503. Completed events are shown after the request finishes;
the active indicator while waiting is not a simulated tool-execution replay.

### Frontend files

- `frontend/index.html`: supplied page layout and animated aerial illustration.
- `frontend/style.css`: supplied survey-map styling, responsive layout and animations.
- `frontend/app.js`: typed study requests, completed-step history, brief, map,
  research/solar charts, rankings, email drafts and evidence tables.

### Checks

```sh
pytest -q
node --check frontend/app.js
```
