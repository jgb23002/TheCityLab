"""Offline showcase: look up a saved Q1-Q3 result and replay it without any network call.

scripts/build_showcase.py writes data/showcase/results/q<N>.json and index.json. When
config.USE_CACHE_ONLY is true, or a request asks for replay=true and its question is in
the index, the agent routes serve the saved result instead of running the planner. A
replayed study is also stored like a fresh run so POST /api/correct works on it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend import config
from backend.agent.evidence import Ledger
from backend.agent.study_store import StudyRun, save_study_run
from backend.schemas import AgentEvent, AskResponse, StudyResult

NOT_IN_SHOWCASE = "Demo mode: this question is not in the offline showcase."


def showcase_dir() -> Path:
    """data/showcase (resolved at call time so tests can relocate it)."""
    return config.PROJECT_ROOT / "data" / "showcase"


def normalize_question(text: str) -> str:
    """Lowercase, trim, punctuation removed (treated as a space), whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", (text or "").lower()).split())


def load_index() -> dict[str, dict[str, Any]]:
    """Normalised question -> {id, file, question, mode}; empty when no showcase is built."""
    try:
        data = json.loads((showcase_dir() / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("entries") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def lookup(question: str) -> dict[str, Any] | None:
    """The saved result for a question (matched after normalisation), or None."""
    entry = load_index().get(normalize_question(question))
    if not entry:
        return None
    root = showcase_dir().resolve()
    path = (root / str(entry.get("file", ""))).resolve()
    if root not in path.parents or not path.is_file():  # index entries can only point inside the showcase
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        StudyResult.model_validate(saved["study"])
    except (OSError, ValueError, KeyError):
        return None
    return saved


def remember(saved: dict[str, Any]) -> StudyResult:
    """Store a replayed study and ledger like a fresh run (so corrections work); return the study."""
    study = StudyResult.model_validate(saved["study"])
    save_study_run(StudyRun(study=study, ledger=Ledger.from_entries(saved.get("ledger") or [])))
    return study


def response_from(saved: dict[str, Any]) -> AskResponse:
    """The saved AskResponse (study + events), stored for the correction loop."""
    study = remember(saved)
    events = [AgentEvent.model_validate(e) for e in saved.get("events") or []]
    return AskResponse(study=study, events=events)
