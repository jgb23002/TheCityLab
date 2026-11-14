# CityLab

CityLab is a HackPrinceton Fall 2026 project: an AI research agent that measures
buildings from Vexcel aerial images, tests questions against CoStar data, and
ranks buildings for rooftop solar.

## Data and integrations

- CoStar is a local hackathon kit in `data/costar/`, not an API. Load
  `data/costar/data/*.jsonl` with `pandas.read_json(path, lines=True)`.
- Read `data/costar/index.md`, `data/costar/models/*.md`,
  `data/costar/concepts/*.md`, and `data/costar/schemas/` before using the data.
  Never invent source field names; verify them in `data/costar/models/*.md`.
- Houston is `primaryAddress.researchMarketId == 14`. Source field names use
  camelCase; `propertyType` and other enums are stored as integers.
- `data/costar/commercial-properties.csv` maps `PropertyId` to
  `MatterportModelId`. `PropertyId` joins to the commercial property's `id`.
- Vexcel is a live API at `https://api.vexcelgroup.com/v2`. Read `VEXCEL_API_KEY`
  from the environment and send it in the `X-API-KEY` header. WKT uses longitude
  first and `srid=4326`. Read the local quickstart at
  `notes/vexcel/vexcel_quickstart.py`; do not copy it into tracked files.
- Cache every Vexcel call under `data/cache/` and honor `USE_CACHE_ONLY`.
- Matterport is optional for now. Its GraphQL endpoint is
  `https://api.matterport.com/api/models/graph`, with HTTP Basic authentication
  using `MP_TOKEN_ID` and `MP_TOKEN_SECRET`.

## Coding rules

- Never hard-code or print keys. Read credentials from `.env` through the
  environment settings in `backend/config.py`.
- Never commit `.env`, `data/costar/`, `data/cache/`, or `notes/vexcel/`.
  Keep the supplied kit and quickstart local; do not copy them into the repo.
- Every tool function must have type hints and a docstring and return
  JSON-serializable data. Return image paths and metadata instead of bytes.
- The initial scaffold contains signatures and TODOs only. Implement behavior
  when a task explicitly calls for it.
- Add a pytest test for each change and run `pytest -q`.
- Finish each task with a summary of files changed and how to test.
