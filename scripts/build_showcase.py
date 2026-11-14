"""Build the offline showcase: run Q1-Q3 for real once, save everything needed to replay them.

Usage: python scripts/build_showcase.py [--n 40] [--only Q1,Q3] [--force] [--images-only]

For each question in data/showcase/questions.json the planner runs in real mode
(cached Vexcel/vision results are reused). Each run's {study, ledger, events,
evidence} is saved to data/showcase/results/q<N>.json, with secrets redacted and raw
Vexcel responses removed. Only the aerial images of the studies' buildings are copied
to data/showcase/images/property_<id>.jpg (max 1024 px, JPEG quality 80), and
data/showcase/index.json maps each normalised question to its result file.

Resumable: an existing result file or image is kept unless --force. The script prints
the showcase size and exits non-zero if it is over MAX_BYTES (25 MB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/build_showcase.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from backend import config  # noqa: E402
from backend.agent.registry import redact  # noqa: E402
from backend.agent.showcase import normalize_question  # noqa: E402,F401

MAX_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PX = 1024
JPEG_QUALITY = 80
DROP_KEYS = {"raw_response"}  # large provider payloads never go into the showcase


def showcase_dir() -> Path:
    """data/showcase (resolved at call time so tests can relocate it)."""
    return config.PROJECT_ROOT / "data" / "showcase"




def strip_raw(value: Any) -> Any:
    """Remove raw provider responses at any depth."""
    if isinstance(value, dict):
        return {k: strip_raw(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [strip_raw(v) for v in value]
    return value


def result_payload(question: dict[str, Any], run: Any) -> dict[str, Any]:
    """Everything replay needs: study, ledger, events and evidence (redacted, no raw responses)."""
    from backend.agent.agent_api import events_from_run
    from backend.agent.events import evidence_list

    return redact(strip_raw({
        "question_id": question["id"], "question": question["question"], "mode": question["mode"],
        "generated_at": datetime.now(timezone.utc).isoformat(), "verified": run.verified,
        "complete": run.complete, "problems": run.problems,
        "study": run.study.model_dump(mode="json"), "ledger": run.ledger.to_json(),
        "events": [e.model_dump(mode="json") for e in events_from_run(run)],
        "evidence": evidence_list(run.ledger.all()),
    }))


def write_json(path: Path, data: Any) -> None:
    """Atomic JSON write (a crash never leaves a half-written result)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")  # compact: size counts
    os.replace(tmp, path)


def copy_image(property_id: int, force: bool = False) -> str:
    """Resize one cached aerial JPEG into the showcase: 'copied', 'kept' or 'missing'."""
    target = showcase_dir() / "images" / f"property_{property_id}.jpg"
    if target.exists() and not force:
        return "kept"
    source = config.CACHE_DIR / "vexcel" / "images" / f"property_{property_id}.jpg"
    if not source.is_file():
        return "missing"
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))  # keeps aspect ratio; never enlarges
        img.save(target, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return "copied"


def showcase_size() -> int:
    """Bytes of the generated showcase (results, images and index)."""
    root = showcase_dir()
    files = [root / "index.json", *(root / "results").glob("*.json"), *(root / "images").glob("*.jpg")]
    return sum(f.stat().st_size for f in files if f.is_file())


def check_size(limit: int = MAX_BYTES) -> int:
    """Return the showcase size; raise SystemExit if it is over *limit*."""
    size = showcase_size()
    if size > limit:
        raise SystemExit(f"Showcase is {size / 2**20:.1f} MB, over the {limit / 2**20:.0f} MB limit.")
    return size


def build_index(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """index.json: normalised question -> result file, for every result that exists."""
    entries = {}
    for q in questions:
        name = f"results/{q['id'].lower()}.json"
        if (showcase_dir() / name).is_file():
            entries[normalize_question(q["question"])] = {"id": q["id"], "file": name, "question": q["question"],
                                                          "mode": q["mode"]}
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "entries": entries}


def main(argv: list[str] | None = None) -> int:
    """Run (or resume) the showcase build and enforce the size limit."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--only", default="", help="comma-separated question ids, e.g. Q1,Q3")
    ap.add_argument("--force", action="store_true", help="re-run questions and re-copy images")
    ap.add_argument("--images-only", action="store_true", help="skip planner runs; refresh images and index")
    args = ap.parse_args(argv)

    from backend.agent import planner  # imported here so --help works without model settings
    from backend.agent import agent_api, events  # noqa: F401  load now, before any long run

    questions = json.loads((showcase_dir() / "questions.json").read_text(encoding="utf-8"))["questions"]
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    for q in questions:
        if wanted and q["id"] not in wanted:
            continue
        path = showcase_dir() / "results" / f"{q['id'].lower()}.json"
        if args.images_only or (path.exists() and not args.force):
            print(f"{q['id']}: kept {path.name}" if path.exists() else f"{q['id']}: no result yet")
            continue
        print(f"{q['id']}: running the planner (real mode, n={args.n}) ...", flush=True)
        run = planner.run_question(q["question"], mode="real", n=args.n, intent=q["mode"])
        write_json(path, result_payload(q, run))
        print(f"{q['id']}: saved {path.name} - {planner.verdict_line(run)}", flush=True)

    counts = {"copied": 0, "kept": 0, "missing": 0}
    for path in sorted((showcase_dir() / "results").glob("q*.json")):
        study = json.loads(path.read_text(encoding="utf-8"))["study"]
        for building in study["buildings"]:
            counts[copy_image(building["id"], args.force)] += 1
    write_json(showcase_dir() / "index.json", build_index(questions))
    print(f"Images: {counts['copied']} copied, {counts['kept']} kept, {counts['missing']} missing (no cached image).")
    size = check_size()
    print(f"Showcase size: {size / 2**20:.1f} MB (limit {MAX_BYTES / 2**20:.0f} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
