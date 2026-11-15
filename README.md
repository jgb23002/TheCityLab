# CityLab

**Find the apartment roofs that are wasting money, do the math, and tell the owner.**

CityLab is an AI research agent that scans every sampled apartment community in a city,
measures each roof from aerial imagery, simulates a full year of sun on every roof face,
places real-size solar panels, and ranks the buildings by what solar would be worth. For
the best roofs it builds a 3D model and drafts an email to the owner.

Built at **HackPrinceton Fall 2026** by Team Cook.

| Track | How CityLab fits |
| --- | --- |
| 🌱 **Sustainability** (main track) | Finds where rooftop solar pays off, city-wide, before anyone climbs a ladder |
| 🏢 **CoStar** | Every building, owner, unit count and lot size comes from the CoStar dataset |
| 🦜 **LangChain** | A DeepAgents (LangChain) agent runs the analysis; LangSmith traces every run |
| ☁️ **Amazon: Best Practical AI Innovation** | A Bedrock-powered agent a solar sales team could use every day, from start to finish |

---

## The problem

Apartment roofs are some of the largest unused solar surfaces in a city, but finding the
good ones is slow and expensive. Owners don't know if their roof is worth it, and
installers can't pay for a site visit to every building just to find out. The data to
answer the question already exists. It's scattered across a real-estate database, aerial
photos and public weather records.

## What CityLab does

Pick **Houston** or **Los Angeles** and click **Find the roofs**. CityLab's agent then:

1. **Discovers** 50 apartment communities from CoStar, with units, lot size and owner.
2. **Follows the sun** every 10 minutes for a full year, with NASA cloud data for the city.
3. **Traces every roof** from Vexcel aerial data and splits sloped roofs into real faces.
4. **Places real-size panels** (400 W, 1.72 × 1.13 m) only on faces that get at least
   80% of the best sun, with a 3 ft fire setback and room left for rooftop AC units.
5. **Prices each site:** yearly kWh from NREL PVWatts, installed cost, savings and payback.
6. **Ranks the sites**, builds the top 5 in 3D, and drafts the owner email (never sent).
7. **Checks itself** against NREL's own solar model.

Every step streams live into the page as the agent works, with the real numbers each
tool returned.

### Results

| | Houston | Los Angeles |
| --- | --- | --- |
| Apartment communities analyzed | 43 of 50 | 46 of 50 |
| Roofs traced / roof faces scored | 505 / 2,272 | 59 / 98 |
| Panels placed | 119,955 | 23,272 |
| Total estimated savings | **$5.87M / year** | **$3.29M / year** |
| #1 site | 33 Thirty Three: $412K/yr, pays back in 13–22 yrs | HillCreste Apartments: $531K/yr, pays back in 5–8 yrs |

Los Angeles pays back faster because power costs more there ($0.22/kWh vs $0.09/kWh) and
it gets more sun (10% cloud loss vs 23% in Houston).

### Why you can trust the numbers

- **Matches NREL PVWatts**, the U.S. government's solar calculator. Roof faces differ by
  a median of **1.9%** (Houston) and **3.3%** (LA). Both models rank roofs the same way
  (rank agreement 0.98 and 0.96) and pick the same best roof.
- **Matches real installations.** On **53 Los Angeles homes that already have solar**,
  Vexcel's AI detected the installed panels and we measured where they sit. On the median
  home, **95% of the panel area** is on faces CityLab recommends. **No** installed panels
  sit on faces CityLab rates below 60% of the best sun.
- **Matches NASA.** Our sun model's flat-roof sunlight equals NASA POWER's measurement to
  within 1 kWh/m²/year in both cities.
- **Never invents a number.** The AI model decides the order of the steps and writes a
  short summary. Every number on the page comes from a tool, never from the model.

---

## Sponsor tracks

### 🌱 Sustainability: main track

Solar on existing roofs is one of the cheapest ways to add clean power to a city, because
no new land is needed. The hard part is knowing *which* roofs are worth it. CityLab turns
a city-wide screening job into a few minutes of computation:

- It finds the sunny faces of each roof and leaves out the shaded ones, so no panel goes
  where it would waste money.
- It sizes each system from the real roof, not a rule of thumb.
- It shows the payback and the savings per apartment per month. If the owner saves on
  power, they don't have to raise the rent.
- It drafts the email that starts the conversation with the owner.

### 🏢 CoStar

CoStar's dataset is the backbone of CityLab. Every building starts as a CoStar record:

| CoStar data | What CityLab uses it for |
| --- | --- |
| Commercial property records (Houston market 14, LA market 20) | The sample of apartment communities, location, units, floors, year built |
| Lot size | Matching each property to the right Vexcel parcel |
| True owner contacts and sale records | Who the owner email goes to (owner, else property manager) |
| Units | Savings per apartment per month |

Code: [backend/tools/costar_data.py](backend/tools/costar_data.py), the `owners()`
lookup in [scripts/build_solar_demo.py](scripts/build_solar_demo.py), and the samples in
[backend/samples.py](backend/samples.py).

### 🦜 LangChain

The solar analysis is run by an agent built with **DeepAgents**, LangChain's agent
toolkit, on LangChain and LangGraph:

- **Agent:** `create_deep_agent` with a system prompt and five LangChain tools, one per
  stage of the analysis: `find_apartment_buildings`, `measure_year_of_sun`,
  `trace_roofs_and_place_panels`, `rank_roofs_by_savings` and `check_against_nrel`.
- **Planning:** LangChain's `write_todos` tool gives the agent a to-do list, which is
  shown on the page.
- **Streaming:** a LangChain callback forwards each plan and tool call to the browser as
  it happens.
- **Guard rails:** if the agent skips a stage, the backend runs it and labels it
  "run by the backend". If the model is unavailable, the same tools run in order and the
  page says why.
- **LangSmith:** every run is traced as `citylab-solar-city` in the `citylab` project,
  showing each model call, the plan and each tool's inputs and outputs.

The original question-box app at `/classic` uses a larger DeepAgents setup with
subagents (sampler, inspector, statistician, writer) and a citation check.

Code: [backend/agent/solar_city.py](backend/agent/solar_city.py) (solar agent) and
[backend/agent/planner.py](backend/agent/planner.py) (research agent).

### ☁️ Amazon: Best Practical AI Innovation

> *"Build AI that someone would use every day, not AI that only works in a demo."*

**Who uses it, and when.** A solar installer's sales team, or a property manager with a
portfolio of buildings. Every morning they pick a city and click one button. They get the
roofs worth calling about, ranked by savings, with the payback, a 3D view of the panels
and a drafted email to the right owner or manager. Without CityLab, building that list
means a site visit or a manual estimate for every building.

**An agent that works from start to finish.** The CityLab agent runs on **Amazon
Bedrock** (Claude Haiku 4.5, through LangChain's `ChatBedrockConverse`).

| What Amazon asks for | What the CityLab agent does |
| --- | --- |
| **Reasons and plans** | Reads the goal, writes its own to-do list (`write_todos`) and picks the order of its tools |
| **Takes action** | Runs five real tools across CoStar, Vexcel, NOAA, NASA and NREL, then ranks the roofs and drafts the owner emails |
| **Takes multi-step actions** | Goes from "which roofs in Houston?" to a ranked list with 3D models and emails, with no human steps in between |
| **Adapts** | If a tool fails, the agent gets the error and carries on. If it skips a step, the backend finishes it and says so. If the model is unavailable, the same tools still run |
| **Learns from feedback** | In the research app (`/classic`), users can correct a measurement. Each correction re-runs the analysis without a model call and feeds a calibration report (`GET /api/calibration`) that tracks how often, and by how much, each measurement is wrong |
| **Feels simple** | One button. The page shows each step as it happens, in plain words with real numbers |

**How it scores on Amazon's four criteria**

- **Creativity:** it combines a real-estate database, aerial imagery, a year of sun
  simulation and government solar data. No one source answers the question alone.
- **Usefulness:** it screens a whole city in minutes instead of one site visit per
  building, for a decision worth hundreds of thousands of dollars a year per building ($5.87M a year in total across
  43 Houston sites).
- **Technical implementation:** every number comes from a tool, never from the model.
  Results match NREL PVWatts (median gap 1.9% in Houston) and 53 real Los Angeles solar
  installations. There are 439 automated tests. Bedrock calls retry with adaptive
  backoff (up to 10 attempts), so busy periods don't break a run.
- **User experience:** one click, a live progress panel, a map, a 3D roof you can orbit
  and a ready-to-send email draft. The second visit is faster, because every provider
  call is cached.

**Built with:** Amazon Bedrock, LangChain + DeepAgents (agentic framework), LangSmith
(tracing) and open-source tools (FastAPI, Shapely, Three.js, Leaflet).

Code: [backend/llm.py](backend/llm.py) (Bedrock setup) and
[backend/agent/solar_city.py](backend/agent/solar_city.py) (the agent).

---

## How it works

```
Browser (frontend/solar/)                    FastAPI backend
──────────────────────────                   ─────────────────────────────────────────────
"Find the roofs"  ──POST /api/solar-city/stream──▶  DeepAgents agent (Amazon Bedrock model)
                                                     │  plans with write_todos
"Working on it"   ◀── live steps (SSE) ──────────────┤  calls 5 LangChain tools:
                                                     │   1 CoStar buildings, owners
Map, 3D roofs,    ◀── result: ranked roofs ──────────┤   2 Sun every 10 min + NASA POWER weather
savings card,                                        │   3 Vexcel roofs → faces → panels
owner email                                          │   4 Rank, 3D models, owner email
                                                     │   5 Check against NREL PVWatts
                                                     └─▶ LangSmith trace
```

| Data source | Used for |
| --- | --- |
| **CoStar** | Buildings, units, lot sizes, owners |
| **Vexcel** | Parcel roof outlines and shapes, rooftop AC units, aerial photos, panel detection |
| **NOAA** equations | Sun position every 10 minutes for a year |
| **NASA POWER** | Monthly cloud cover and sunlight |
| **NREL PVWatts V8** | Yearly energy per roof face, and the accuracy check |

All Vexcel and NREL calls are cached under `data/cache/`, so a repeat run costs nothing.

---

## Run it

**Requirements:** Python 3.11, the CoStar hackathon kit unzipped into `data/costar/`, and
API keys.

```sh
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # then fill in your keys
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/**.

| Address | Page |
| --- | --- |
| `/` | CityLab Solar: the main experience |
| `/classic` | The original research app (ask a question about a city's buildings) |

**Keys in `.env`:**
- `VEXCEL_API_KEY`
- `NREL_API_KEY`
- AWS credentials with `LLM_PROVIDER=bedrock`, `LLM_MODEL` and `VISION_MODEL`
- Optional: `LANGSMITH_API_KEY` with `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=citylab`

Keys are read only through [backend/config.py](backend/config.py) and are never printed
or committed.

**Offline mode** (no model or paid API calls):

```sh
USE_CACHE_ONLY=true uvicorn backend.main:app --reload
```

The backend runs the same tools on cached data only. If a city's roofs aren't cached, the
page shows the recorded run and labels it as recorded.

**Rebuild a city's recorded data:**

```sh
python scripts/build_solar_demo.py --city houston       # or los_angeles
```

## Tests

```sh
pytest -q
```

There are 439 tests. They cover the agent with a scripted model, the live stream, the
sun model, roof geometry, the finance math and the API routes. They run offline in about
10 seconds.

## Project structure

```
backend/
  main.py               FastAPI app; serves the solar page at /
  agent/solar_city.py   Solar agent (DeepAgents), its 5 tools and the live stream
  agent/planner.py      Research agent for /classic (subagents, citation check)
  llm.py                Model setup (Amazon Bedrock, via LangChain)
  sun.py                Sun position and NASA POWER weather
  tools/                CoStar, Vexcel, solar finance, geometry, outreach, statistics
  vexcel_detect.py      Vexcel parcel matching and panel detection
scripts/
  build_solar_demo.py   The staged solar pipeline (CityRun) used by the agent and the CLI
  roof_solar_view.py    Roof faces, panel fitting, PVWatts per face
  validate_installs.py  Real-install test drawings
  detect_installs.py    Real-install test, measured with Vexcel's panel detector
frontend/
  solar/                CityLab Solar page (map, 3D roofs, live agent steps)
  index.html, app.js    Research app at /classic
data/samples/           The fixed 50-building samples for each city
docs/CONTRACTS.md       Tool signatures (checked by tests/test_contracts.py)
tests/                  pytest suite
```

## Assumptions and limits

- **Costs and prices:** installed cost is $2.00/W (range $1.80–$2.40). Power price is
  $0.09/kWh in Houston and $0.22/kWh in LA. No tax credits are included.
- **Simple payback:** every kWh produced is assumed to replace a kWh bought from the grid.
- **Sloped roofs:** assumed to be 27°, measured from roof shading on one building.
- **Roof outlines** are Vexcel's detected footprints. They can be off under trees or at
  complex rooflines.
- **Not modelled:** shade from neighbouring buildings and trees.
- **Not an engineering design.** CityLab is a screening tool that decides which roofs are
  worth a site visit. A site visit still makes the final call.

## Data and privacy

CoStar data, the Vexcel cache and `.env` are licensed or secret and are never committed
(see [.gitignore](.gitignore)). The repository contains only code, fixed building samples,
the recorded page data and the photos the page displays. Owner emails are drafts only;
CityLab never sends anything.
