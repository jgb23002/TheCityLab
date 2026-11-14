# CityLab

CityLab is a HackPrinceton Fall 2026 project for an AI research agent that
measures buildings from Vexcel aerial images, tests questions against local
CoStar data, and ranks buildings for rooftop solar. This repository currently
contains typed stubs, output schemas, a health endpoint, and a static placeholder.

## Setup

1. Create and activate a virtual environment with Python 3.11 or newer:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create your local settings and fill in keys as needed:
   ```sh
   cp -n .env.example .env
   ```
   If `.env` already exists, keep it and add any missing names from the example.
4. Put the CoStar hackathon kit in `data/costar/`, with its JSON Lines files
   under `data/costar/data/`. The kit, Vexcel notes in `notes/vexcel/`, and cache
   in `data/cache/` stay local and git-ignored. CoStar uses local files; Vexcel
   uses a live API; Matterport is optional. The scaffold makes no external calls.
5. Run the app from the repository root:
   ```sh
   uvicorn backend.main:app --reload
   ```
   Open `http://127.0.0.1:8000/` for the placeholder or
   `http://127.0.0.1:8000/api/health` for `{"ok": true}`.

Run the smoke tests from the repository root:

```sh
pytest -q
```

The [showcase definitions](data/showcase/README.md) cover the Houston research
questions and Los Angeles solar shortlist. Generate their 50-property samples
with `python scripts/prepare_showcase.py --size 50 --seed 42`; selection rules
and source-field mappings are saved in `data/samples/manifest.json`.

To describe an aerial JPEG, set `LLM_PROVIDER` (`bedrock`, `google`, or
`anthropic`) and a vision-capable `LLM_MODEL` in `.env`, then run:

```sh
python -m backend.llm <image_path>
```

Bedrock uses `AWS_REGION` and your AWS credentials. Google uses `GOOGLE_API_KEY`.
For Anthropic, add its standard `ANTHROPIC_API_KEY` to `.env`. The command asks
"Describe this aerial image in one sentence." and prints the text reply.
