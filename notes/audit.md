# Finish-Line Build Guide audit

Audited 2026-11-14 (America/New_York) against the supplied checklist and the local repository/worktree. Checklist totals: **8 PASS, 5 FAIL, 1 PARTIAL**; the D14 file rows expand its inventory and are not extra checklist items.

Only `notes/audit.md` was written in the project. Bytecode and pytest-cache writes were disabled; test fixtures used system temporary directories. The exploration script's output was captured in memory instead of replacing its report. Live probes used exactly one operation per provider, with retries disabled; the read-only instruction took precedence over writing a Vexcel cache entry. No credentials, reference files or data responses were copied into the report. File size/mtime snapshots confirmed the other 141 project files were unchanged (excluding `.git` and `.venv`).

| Item | Status | Evidence |
|---|---|---|
| A1 Git branch / main | FAIL | Current branch `jiya` at `56607b7`; local `main` is `940eab1` (3 commits behind), and 9 pre-existing modified/untracked paths contain additional work; no fetch, commit, staging or merge was performed. |
| A2 Ignore rules | PASS | `git check-ignore --no-index` confirms `.env`, `.venv/`, `data/costar/`, `data/cache/` and `notes/vexcel/` are ignored. |
| A3 Tracked files / secret scan | PASS | `git ls-files` contains none of the private paths; AKIA/aws_secret and broader secret-shape scans found no credentials; four assignment-name hits are two blank template entries and two `os.getenv` reads. |
| A4 Environment template coverage | FAIL | All 14 explicit environment-read names are present, but SDK-used `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are absent; guide setting `VISION_MODEL` is also absent; keys are blank and the only default value is `USE_CACHE_ONLY=false`. |
| B5 Interpreter / activation / imports | FAIL | `.venv/bin/python` is Python 3.11.5 and all 18 requirements import; however the inherited audit shell has `VIRTUAL_ENV` MISSING and plain `python --version` fails (permission denied); select/activate `.venv` before bare commands. |
| B6 Tests | PASS | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider` exits 0: 82 passed in 0.83s; pytest cache and bytecode writes were disabled. |
| B7 Local settings / formatting / models | PASS | VEXCEL_API_KEY SET; MP_TOKEN_ID SET; MP_TOKEN_SECRET SET; AWS_ACCESS_KEY_ID SET; AWS_SECRET_ACCESS_KEY SET; AWS_REGION SET; LLM_PROVIDER SET; LLM_MODEL SET; VISION_MODEL SET; all are unique single-line printable-ASCII values without spaces/hidden characters; both model IDs MATCH the requested IDs. |
| C8 Data exploration | PASS | `scripts/explore_data.py` entry point completed against the local kit with only its final write redirected into memory; its 129-line result exactly matches existing `notes/data_summary.md` (129 lines, under 200), which was not changed. |
| C9 Question picker / options table | PASS | `scripts/pick_questions.py` exists; `notes/question_options.md` contains the city, property-type, outcome and complete-building-count table sorted largest first. |
| C10 Showcase definitions | FAIL | `data/showcase/questions.json` contains exactly Q1, Q2 and Q3; `notes/showcase_questions.md` is MISSING. |
| C11 Vexcel probe / artifacts | FAIL | `scripts/vexcel_test.py` has X-API-KEY, five 429 retries, property/extract, ortho/extract and cache/sample writers; custom-elements `elements/detect` is absent; sample JSON/JPEG counts are 0/0, and existing `notes/vexcel_fields.md` documents earlier 401 failures rather than observed fields. |
| C12 LLM module / CLI / mocked test | PASS | `backend/llm.py` implements `get_chat_model()` and `ask_with_image()`; `python -B -m backend.llm --help` exits 0, and mocked provider/image/CLI tests pass; image calls currently select `LLM_MODEL`, not the separate `VISION_MODEL` setting. |
| C13 Tiny live checks | PASS | Exactly one Bedrock Converse call on configured LLM_MODEL with `Say OK`: HTTP 200, OK; exactly one Vexcel property/extract GET using the first Houston sample point: HTTP 200, features=8; retries/redirects disabled and neither response saved. |
| D14 Implementation inventory | PARTIAL | All 8 tool sources and 3 agent sources remain stubs; frontend contains 2 comment-only files and a working HTML placeholder, not the research interface; every source file is classified below (ignored bytecode excluded). |
| D14 backend/tools/__init__.py | PARTIAL | STUB/package scaffold: 3 lines, docstring and TODO only. |
| D14 backend/tools/costar_data.py | PARTIAL | STUB: 22 lines; `load_properties`, `get_property` and `find_buildings` contain documented TODOs and ellipsis bodies. |
| D14 backend/tools/matterport.py | PARTIAL | STUB: 9 lines; `get_model` has a TODO and ellipsis body. |
| D14 backend/tools/outreach.py | PARTIAL | STUB: 10 lines; `draft_outreach` has a TODO and ellipsis body. |
| D14 backend/tools/solar.py | PARTIAL | STUB: 17 lines; `estimate_solar` and `rank_buildings` have TODOs and ellipsis bodies. |
| D14 backend/tools/stats.py | PARTIAL | STUB: 10 lines; `run_study` has a TODO and ellipsis body. |
| D14 backend/tools/vexcel.py | PARTIAL | STUB: 15 lines; `get_aerial` and `get_property_info` have TODOs and ellipsis bodies; the separate probe script does not implement these tools. |
| D14 backend/tools/vision.py | PARTIAL | STUB: 8 lines; `measure_building` has a TODO and ellipsis body. |
| D14 backend/agent/__init__.py | PARTIAL | STUB/package scaffold: 3 lines, docstring and TODO only. |
| D14 backend/agent/planner.py | PARTIAL | STUB: 8 lines; `plan_research` has a TODO and ellipsis body. |
| D14 backend/agent/prompts.py | PARTIAL | STUB: 5 lines; TODO plus empty `SYSTEM_PROMPT` and `PLANNER_PROMPT` strings. |
| D14 frontend/app.js | PARTIAL | STUB: 2 comment/TODO lines; no executable UI or API integration. |
| D14 frontend/index.html | PARTIAL | STUB application / real static HTML shell: 16 lines displaying `CityLab is running`, with a TODO for the research interface. |
| D14 frontend/style.css | PARTIAL | STUB: 2 comment/TODO lines; no CSS rules. |

The required model comparisons both matched: LLM_MODEL should be `us.anthropic.claude-haiku-4-5-20251001-v1:0`; VISION_MODEL should be `us.anthropic.claude-sonnet-4-6`. No secret values are shown. `VISION_MODEL` is configured locally but is not currently consumed by `backend/config.py` or `backend/llm.py`.

`git log --oneline -8` returned all four existing commits:

```text
56607b7 save changes
8dc0f7b Data summary and showcase questions
b4a9920 Project structure, AGENTS.md, health check
940eab1 Project setup
```

Local main contains only `940eab1 Project setup`. The 9 pre-existing working-tree changes are separate from this newly generated audit report. The successful fresh Vexcel probe supersedes earlier authentication failures for current connectivity; it does not populate the missing sample artifacts or update `notes/vexcel_fields.md` during this read-only audit.

## Fix first

1. **A1 — Preserve the latest work on main.** Review and commit the intended CityLab changes on `jiya`, then update local `main` to include them; never stage secrets or ignored kit/cache/reference data.
2. **C11 — Complete the Vexcel deliverable.** Add custom-elements detection following the local quickstart, then use the now-working authentication to generate the three property JSONs, three JPEGs and an observed-field inventory; retain caching and secret-free logs.
3. **A4 — Complete `.env.example`.** Add blank entries for SDK credential names `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `ANTHROPIC_API_KEY`, plus `VISION_MODEL`; preserve blank secret values.
4. **C10 — Add the missing showcase Markdown.** Create `notes/showcase_questions.md` consistent with the existing Q1–Q3 JSON and the agreed Houston/Los Angeles cohort rules.
5. **B5 — Activate the project environment.** Use `source .venv/bin/activate`; confirm bare `python --version` resolves to Python 3.11.5 and `pytest` uses this environment; no dependency installation is needed.

## Exact next Codex prompt

```text
Read notes/audit.md and AGENTS.md. Fix A1 only: inspect all current modified and untracked files on jiya, review their diffs, and preserve the intended CityLab source, templates, documentation and tests in a commit. Never stage .env, .venv/, data/costar/, data/cache/, notes/vexcel/ or any credential value. Activate .venv and run pytest -q before committing. Then fast-forward local main so it contains the saved work if the history permits; if main has diverged, report that fact without resetting, overwriting changes or force-updating refs. Do not push or delete branches. Report the new commit IDs, final git status, and proof that main contains the saved work.
```

