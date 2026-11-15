"""Measured check: on roofs that already have solar, are the real panels on the sides our model picks?

No judging by eye. For every sloped LA roof Vexcel flags with existing panels:
  1. our model splits the roof into its two sides and scores each (share of the best sun);
  2. Vexcel's own object detector (elements/detect, prompt "solar panel") finds the panels
     on the same aerial photo;
  3. we measure how many square metres of detected panel fall on each side.

Panel area comes from Vexcel's pixel masks (the exact panel shape), not the boxes around them.
A roof "matches" when at least MATCH_SHARE of its detected panel area sits on sides we pick.
Roofs where the detector finds no panels are reported as "not found", not dropped silently.
Structures whose outline is 70%+ covered by panels (parking canopies, ground mounts) are not roofs
and are left out. The same building photographed twice (centres within DUP_M metres) is counted once.

Reads frontend/solar/installs_check.json (from validate_installs.py) and writes the measured
fields back into it, and paints the detected panels on its photos (so run validate_installs.py first,
or the photos get painted twice). Run: python scripts/detect_installs.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from shapely import contains_xy  # noqa: E402
from shapely import wkt as shapely_wkt  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from backend import vexcel_detect  # noqa: E402
from backend.tools import vexcel  # noqa: E402
from scripts.roof_solar_view import M_PER_DEG_LAT, roof_sides  # noqa: E402
from scripts.validate_installs import pv_roofs  # noqa: E402

CHECK = PROJECT_ROOT / "frontend" / "solar" / "installs_check.json"
MATCH_SHARE = 0.8   # at least 80% of the detected panel area on our sides
DUP_M = 15.0
SQFT_TO_M2 = 0.09290304
# Panels covering 70%+ of the whole outline = the structure IS the solar array (parking canopy,
# ground mount), not a roof with panels on it. Real roofs here top out near 50%.
STRUCTURE_COVER = 0.7


def local_poly(g, c) -> Polygon:
    kx = M_PER_DEG_LAT * math.cos(math.radians(c.y))
    return Polygon([((x - c.x) * kx, (y - c.y) * M_PER_DEG_LAT) for x, y in g.exterior.coords])


def mark_panels(row: dict, g, drawn: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Draw the detected panel pixels (light blue) on the roof photo validate_installs.py saved,
    so what we measured is visible. Same frame as that script: roof bounds + 6 m."""
    path = PROJECT_ROOT / "frontend" / "solar" / row["image"]
    if not path.exists() or not drawn:
        return
    pad = 6 / M_PER_DEG_LAT
    b = g.bounds
    x0, y0, x1, y1 = b[0] - pad * 1.2, b[1] - pad, b[2] + pad * 1.2, b[3] + pad
    img = Image.open(path).convert("RGB")
    W, H = img.size
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for lon, lat in drawn:
        for u, v in zip((lon - x0) / (x1 - x0) * W, (y1 - lat) / (y1 - y0) * H):
            d.point((float(u), float(v)), fill=(56, 189, 248, 150))
    Image.alpha_composite(img.convert("RGBA"), over).convert("RGB").save(path, quality=88)


def main() -> None:
    check = json.loads(CHECK.read_text())
    roofs = []
    for r in pv_roofs():
        g = shapely_wkt.loads(r["wkt"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        roofs.append((g, r["shape"]))

    seen: list[tuple[float, float]] = []
    for row in check["roofs"]:
        # the footprint this row was drawn from (same centroid)
        g, shape = min(roofs, key=lambda q: abs(q[0].centroid.y - row["lat"]) + abs(q[0].centroid.x - row["lon"]))
        c = g.centroid
        kx = M_PER_DEG_LAT * math.cos(math.radians(c.y))
        if any(math.hypot((c.x - lo) * kx, (c.y - la) * M_PER_DEG_LAT) < DUP_M for la, lo in seen):
            row["measured"] = {"status": "duplicate"}
            print(f"roof {row['n']}: duplicate photo of an earlier building")
            continue
        seen.append((c.y, c.x))

        footprint = local_poly(g, c)
        cover = row["pv_area_sqft"] * SQFT_TO_M2 / footprint.area
        if cover >= STRUCTURE_COVER:
            row["measured"] = {"status": "panel structure", "panel_cover_pct": round(100 * cover)}
            print(f"roof {row['n']}: panels cover {100 * cover:.0f}% of it -> a canopy or ground mount, not a roof")
            continue
        sides = roof_sides(footprint, shape)
        if len(sides) < 2:
            row["measured"] = {"status": "no two sides"}
            continue

        pad = 4 / M_PER_DEG_LAT
        b = g.bounds
        area_wkt = box(b[0] - pad * 1.2, b[1] - pad, b[2] + pad * 1.2, b[3] + pad).wkt
        try:  # Vexcel's default (newest) photo; asking for a specific capture date returns 404
            data = vexcel._client.request_json(
                "POST", vexcel_detect.DETECT_ENDPOINT,
                vexcel_detect._detect_payload(area_wkt, ["solar panel"],
                                              include=("bbox", "confidence", "pixel-bbox", "pixel-mask")), timeout=300)
        except Exception as exc:  # report, never invent
            row["measured"] = {"status": f"detector error ({type(exc).__name__})"}
            print(f"roof {row['n']}: detector error {exc}")
            continue

        # Every panel pixel from Vexcel's mask -> a point on the ground -> which roof side it is on.
        per_side = [0.0 for _ in sides]
        boxes = 0
        drawn: list[tuple[np.ndarray, np.ndarray]] = []  # panel pixels on the roof, lon/lat, for the photo
        for feat in data.get("features") or []:
            for det in ((feat or {}).get("properties") or {}).get("detections") or []:
                if not det.get("pixel-mask") or not det.get("bbox"):
                    continue
                try:
                    gb = shapely_wkt.loads(det["bbox"])
                    m = np.asarray(Image.open(io.BytesIO(base64.b64decode(det["pixel-mask"]))).convert("L")) > 128
                except Exception:
                    continue
                h, w = m.shape
                if not m.any():
                    continue
                boxes += 1
                x0, y0, x1, y1 = gb.bounds
                jj, ii = np.nonzero(m)
                lon = x0 + (ii + 0.5) / w * (x1 - x0)
                lat = y1 - (jj + 0.5) / h * (y1 - y0)
                X, Y = (lon - c.x) * kx, (lat - c.y) * M_PER_DEG_LAT
                px_m2 = ((x1 - x0) * kx / w) * ((y1 - y0) * M_PER_DEG_LAT / h)
                for k, sd in enumerate(sides):
                    per_side[k] += float(contains_xy(sd["poly"], X, Y).sum()) * px_m2
                on_roof = contains_xy(footprint, X, Y)
                drawn.append((lon[on_roof], lat[on_roof]))
        mark_panels(row, g, drawn)
        for a, out in zip(per_side, row["sides"]):
            out["detected_panel_m2"] = round(a, 1)
        panels = range(boxes)
        total = sum(per_side)
        if total < 1:
            row["measured"] = {"status": "not found", "boxes": len(panels)}
            print(f"roof {row['n']}: detector found no panels on the roof")
            continue
        on_ours = sum(a for a, s in zip(per_side, row["sides"]) if s["ours"])
        share = on_ours / total
        row["measured"] = {"status": "match" if share >= MATCH_SHARE else "differs",
                           "share_on_our_sides": round(100 * share), "panel_m2": round(total, 1), "boxes": len(panels)}
        print(f"roof {row['n']}: {len(panels)} boxes, {total:.0f} m2 of panels, {100 * share:.0f}% on our sides -> {row['measured']['status']}")

    tested = [r for r in check["roofs"] if r.get("measured", {}).get("status") in ("match", "differs")]
    check["measured_summary"] = {
        "method": "Vexcel elements/detect ('solar panel') pixel masks, each panel pixel placed on the Vexcel roof outline, area per roof side",
        "match_share_pct": round(100 * MATCH_SHARE),
        "roofs_flagged": len(check["roofs"]),
        "panel_structures_left_out": sum(r.get("measured", {}).get("status") == "panel structure" for r in check["roofs"]),
        "tested": len(tested),
        "match": sum(r["measured"]["status"] == "match" for r in tested),
        "panel_area_on_our_sides_pct": round(100 * sum(r["measured"]["share_on_our_sides"] * r["measured"]["panel_m2"] for r in tested)
                                             / max(1, sum(r["measured"]["panel_m2"] for r in tested)) / 100) if tested else None,
    }
    # Where the real panels sit, by our sun score for that face: shows whether installers ever use
    # faces we rate poorly, or only ones just under our cut-off.
    bands = {"80+": 0.0, "70-79": 0.0, "60-69": 0.0, "under 60": 0.0}
    for r in tested:
        for sd in r["sides"]:
            p = sd["pct"]
            bands["80+" if p >= 80 else "70-79" if p >= 70 else "60-69" if p >= 60 else "under 60"] += sd.get("detected_panel_m2", 0)
    total = sum(bands.values()) or 1
    check["measured_summary"]["panel_area_by_our_score_pct"] = {k: round(100 * v / total) for k, v in bands.items()}
    shares = sorted(r["measured"]["share_on_our_sides"] for r in tested)
    check["measured_summary"]["median_roof_share_pct"] = shares[len(shares) // 2] if shares else None
    CHECK.write_text(json.dumps(check, indent=1))
    print(json.dumps(check["measured_summary"], indent=1))


if __name__ == "__main__":
    main()
