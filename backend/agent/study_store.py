"""Save and load finished studies (study + evidence ledger) under data/cache/studies/.

Files are JSON, git-ignored (data/cache/), and redacted before writing. Corrected
studies are saved under a new id "<original id>-c<N>" with a link to the original.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend import config
from backend.agent.evidence import Ledger
from backend.agent.registry import redact
from backend.schemas import StudyResult

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class StudyNotFound(LookupError):
    """No saved study has this id (or the id is not a valid study id)."""


@dataclass
class StudyRun:
    """A saved study with its ledger and lineage ({original_id, corrected_from, correction_count})."""

    study: StudyResult
    ledger: Ledger
    meta: dict[str, Any] = field(default_factory=dict)


def studies_dir() -> Path:
    """data/cache/studies (resolved at call time so tests can move the cache)."""
    return config.CACHE_DIR / "studies"


def study_path(study_id: str) -> Path:
    """Path for a study id; rejects anything that is not a plain id (no traversal)."""
    if not _SAFE_ID.match(study_id or ""):
        raise StudyNotFound("Unknown study id.")
    return studies_dir() / f"{study_id}.json"


def save_study_run(run: StudyRun) -> Path:
    """Write {study, ledger, meta} for run.study.id, redacting anything secret-shaped."""
    meta = {"original_id": run.study.id, "corrected_from": None, "correction_count": 0, **run.meta}
    data = redact({"study": run.study.model_dump(mode="json"), "ledger": run.ledger.to_json(), "meta": meta})
    path = study_path(run.study.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def load_study_run(study_id: str) -> StudyRun:
    """Load a saved study run; raises StudyNotFound for unknown or invalid ids."""
    path = study_path(study_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        study = StudyResult.model_validate(data["study"])
        ledger = Ledger.from_entries(data.get("ledger") or [])
    except (OSError, ValueError, KeyError, TypeError):
        raise StudyNotFound("Unknown study id.") from None
    return StudyRun(study=study, ledger=ledger, meta=dict(data.get("meta") or {}))


def next_corrected_id(run: StudyRun) -> tuple[str, int]:
    """New id "<original id>-c<N>" for a correction of *run* (N continues across corrections)."""
    root = run.meta.get("original_id") or run.study.id
    n = int(run.meta.get("correction_count") or 0) + 1
    while study_path(f"{root}-c{n}").exists():
        n += 1
    return f"{root}-c{n}", n
