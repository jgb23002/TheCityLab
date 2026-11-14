"""Compare Sky Inspector parking counts with CoStar's recorded surface parking.

Usage: python -m scripts.accuracy_check --n 10 [--market 14] [--min 10] [--max 80] [--refresh]

CoStar's surface spaces (parking[] types 1 Surface and 2 SurfaceTandem) are the
reference. Only properties with a modest lot are sampled so the lot fits in the
~80 m aerial square. Vision results are cached in data/cache/vision/ so reruns
are free. Results are written to data/cache/accuracy.json (git-ignored) and a
one-sentence summary for the pitch is printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from backend import config
from backend.schemas import Measurement
from backend.tools import vision

SURFACE_TYPES = (1, 2)


def surface_spaces(parking: Any) -> int | None:
    """Sum CoStar totalSpaces for surface parking records; None when none are reported."""
    if not isinstance(parking, list):
        return None
    values = [p.get("totalSpaces") for p in parking if isinstance(p, dict) and p.get("parkingType") in SURFACE_TYPES]
    values = [v for v in values if isinstance(v, (int, float))]
    return int(sum(values)) if values else None


def sample_properties(market: int, n: int, low: int, high: int) -> list[dict[str, Any]]:
    """Pick n existing properties in a market with low..high surface spaces, spread across ids."""
    df = pd.read_json(config.COSTAR_DATA_DIR / "commercial-property.jsonl", lines=True)
    rows = []
    for rec in df.to_dict("records"):
        addr = rec.get("primaryAddress") or {}
        spaces = surface_spaces(rec.get("parking"))
        if addr.get("researchMarketId") != market or spaces is None or not low <= spaces <= high:
            continue
        if addr.get("latitude") is None or addr.get("longitude") is None:
            continue
        rows.append({"id": int(rec["id"]), "name": rec.get("buildingName"), "costar_surface_spaces": spaces,
                     "latitude": float(addr["latitude"]), "longitude": float(addr["longitude"])})
    rows.sort(key=lambda r: r["id"])
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def image_for(prop: dict[str, Any]) -> str | None:
    """Return the cached aerial JPEG, fetching it through vexcel.get_aerial when possible."""
    path = config.CACHE_DIR / "vexcel" / "images" / f"property_{prop['id']}.jpg"
    if path.exists():
        return str(path)
    try:
        from backend.tools import vexcel

        result = vexcel.get_aerial(prop["latitude"], prop["longitude"], prop["id"])
        found = result.get("image_path") if isinstance(result, dict) else getattr(result, "image_path", None)
        return str(found) if found else None
    except Exception as exc:
        print(f"  {prop['id']}: no aerial image ({type(exc).__name__})")
        return None


def measure(prop: dict[str, Any], image: str, refresh: bool) -> Measurement:
    """Run the Sky Inspector once per property and cache the result."""
    cache = config.CACHE_DIR / "vision" / f"m-{prop['id']}.json"
    if cache.exists() and not refresh:
        return Measurement.model_validate_json(cache.read_text())
    result = vision.measure_building(image, prop["id"])
    if not (result.observations or {}).get("error"):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(result.model_dump_json())
    return result


def summarize(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Error statistics for (predicted, reference) parking counts."""
    if not pairs:
        return {"n": 0}
    errors = [p - r for p, r in pairs]
    abs_errors = [abs(e) for e in errors]
    pct = [abs(p - r) / r * 100 for p, r in pairs if r > 0]
    return {
        "n": len(pairs),
        "mean_abs_error": round(statistics.mean(abs_errors), 1),
        "median_abs_error": round(statistics.median(abs_errors), 1),
        "mean_signed_error": round(statistics.mean(errors), 1),
        "median_abs_pct_error": round(statistics.median(pct), 1) if pct else None,
        "within_20_pct": sum(1 for p, r in pairs if r > 0 and abs(p - r) <= 0.2 * r),
    }


def pitch_sentence(s: dict[str, Any]) -> str:
    """One sentence for the pitch and README."""
    if not s.get("n"):
        return "No buildings could be checked."
    bias = "over" if s["mean_signed_error"] > 0 else "under"
    return (f"On {s['n']} Houston buildings, the Sky Inspector's parking count was within "
            f"{s['median_abs_error']:g} spaces of CoStar's record at the median "
            f"(mean error {s['mean_abs_error']:g} spaces, median {s['median_abs_pct_error']:g}%), "
            f"{s['within_20_pct']} of {s['n']} within 20%, tending to {bias}count by "
            f"{abs(s['mean_signed_error']):g} spaces.")


def main(argv: list[str] | None = None) -> int:
    """Run the check and print a per-building table plus the pitch sentence."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--market", type=int, default=14)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min", type=int, default=10)
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--refresh", action="store_true", help="re-run vision even if cached")
    args = ap.parse_args(argv)

    rows, pairs = [], []
    for prop in sample_properties(args.market, args.n, args.min, args.max):
        image = image_for(prop)
        if not image:
            continue
        m = measure(prop, image, args.refresh)
        predicted = m.parking_space_count
        rows.append({**prop, "predicted": predicted, "confidence": m.confidence,
                     "error": (m.observations or {}).get("error")})
        flag = "" if predicted is not None else "  (no count)"
        print(f"  {prop['id']:>9}  CoStar {prop['costar_surface_spaces']:>4}  vision {predicted if predicted is not None else '-':>4}{flag}")
        if predicted is not None:
            pairs.append((predicted, prop["costar_surface_spaces"]))

    summary = summarize(pairs)
    out = Path(config.CACHE_DIR) / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary))
    print(pitch_sentence(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
