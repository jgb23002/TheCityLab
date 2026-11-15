"""Check our roof-side choice against buildings that ALREADY have solar panels.

Vexcel marks roofs where its aerial model sees solar panels (roof-objects.solar-panel-photovoltaic).
For each sloped roof with real panels we: split it into its two sides, score each side with our
sun model (share of the best sun), and draw our choice on the real top-down photo. Then a person
(or the blue-panel pixel test below) checks which side the real panels are on.

Writes frontend/solar/views/install_<n>_before.jpg (the plain photo, no overlay),
frontend/solar/views/install_<n>.jpg (our choice drawn on it) -- both Vexcel imagery,
git-ignored -- and frontend/solar/installs_check.json (numbers only).

Run: python scripts/validate_installs.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from shapely import wkt as shapely_wkt  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from backend import config, sun, vexcel_detect  # noqa: E402
from scripts.roof_solar_view import (  # noqa: E402
    KEEP_PERCENT, M_PER_DEG_LAT, ortho_image, roof_sides,
)

OUT_DIR = PROJECT_ROOT / "frontend" / "solar" / "views"
PITCHED_TILT_DEG = 27.0


def pv_roofs() -> list[dict[str, Any]]:
    """Sloped structures where Vexcel sees solar panels, one entry per roof."""
    out, seen = [], set()
    for f in glob.glob(str(config.CACHE_DIR / "vexcel" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for feat in d.get("features") or []:
            for st in (feat.get("properties") or {}).get("structures") or []:
                try:
                    area = float(st.get("roof-objects.solar-panel-photovoltaic.area"))
                except (TypeError, ValueError):
                    continue
                shape = st.get("roof-characteristics.shape.value")
                if area <= 0 or shape not in ("gable", "hip") or not st.get("footprint.wkt"):
                    continue
                key = st["footprint.wkt"][:80]
                if key in seen:
                    continue
                seen.add(key)
                out.append({"wkt": st["footprint.wkt"], "shape": shape, "pv_area_sqft": round(area),
                            "pv_count": st.get("roof-objects.solar-panel-photovoltaic.count"),
                            "pv_score": st.get("roof-objects.solar-panel-photovoltaic.score")})
    return out


def main() -> dict[str, Any]:
    roofs = pv_roofs()
    print(f"{len(roofs)} sloped roofs with existing panels")
    lat0 = np.mean([shapely_wkt.loads(r["wkt"]).centroid.y for r in roofs])
    lon0 = np.mean([shapely_wkt.loads(r["wkt"]).centroid.x for r in roofs])
    steps = sun.sky_year(lat0, lon0)
    best_tilt = sun.sun_summary(lat0, lon0)["best_fixed_tilt_deg"]
    best_year = sun.plane_sunlight(steps, best_tilt, 180)["kwh_m2_year"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ".gitignore").write_text("*\n")
    results = []
    for n, r in enumerate(roofs):
        g = shapely_wkt.loads(r["wkt"])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        c = g.centroid
        kx = M_PER_DEG_LAT * math.cos(math.radians(c.y))
        local = Polygon([((x - c.x) * kx, (y - c.y) * M_PER_DEG_LAT) for x, y in g.exterior.coords])
        sides = []
        for sd in roof_sides(local, r['shape']):
            pct = round(100 * sun.plane_sunlight(steps, PITCHED_TILT_DEG, sd["facing_deg"])["kwh_m2_year"] / best_year)
            sides.append({"facing": sd["facing_deg"], "pct": pct, "poly": sd["poly"], "ours": pct >= KEEP_PERCENT})
        if len(sides) < 2:
            continue
        pad = 6 / M_PER_DEG_LAT
        b = g.bounds
        bounds = (b[0] - pad * 1.2, b[1] - pad, b[2] + pad * 1.2, b[3] + pad)
        path = OUT_DIR / f"install_{n}.jpg"
        before_path = OUT_DIR / f"install_{n}_before.jpg"
        try:
            cols = vexcel_detect.list_collections(c.y, c.x)
            col = next(cc for cc in cols if cc["layer"] in ("urban", "urban-r"))
            img = ortho_image(box(*bounds).wkt, col["collection"], col["layer"])
        except Exception as exc:
            print(f"  roof {n}: no photo ({type(exc).__name__})")
            continue
        W, H = img.size
        img.resize((W * 2, H * 2)).convert("RGB").save(before_path, quality=88)  # plain photo, no overlay

        def px(x: float, y: float) -> tuple[float, float]:
            lon, lat = c.x + x / kx, c.y + y / M_PER_DEG_LAT
            return ((lon - bounds[0]) / (bounds[2] - bounds[0]) * W, (bounds[3] - lat) / (bounds[3] - bounds[1]) * H)

        # Blue-panel pixel test: solar cells are dark and bluish (blue clearly above red).
        a = np.asarray(img, dtype=np.int32)
        bluish = (a[..., 2] - a[..., 0] > 12) & (a.sum(axis=2) < 330)
        for s in sides:
            m = Image.new("L", (W, H), 0)
            ImageDraw.Draw(m).polygon([px(x, y) for x, y in s["poly"].exterior.coords], fill=1)
            m = np.asarray(m, bool)
            s["panel_pixels_pct"] = round(100 * float(bluish[m].mean()), 1) if m.sum() else 0.0

        # Draw our choice: green outline = we'd put panels, orange = we'd skip; label = share of best sun.
        big = img.resize((W * 2, H * 2))
        d = ImageDraw.Draw(big)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
        except OSError:
            font = ImageFont.load_default()
        for s in sides:
            pts = [(u * 2, v * 2) for u, v in (px(x, y) for x, y in s["poly"].exterior.coords)]
            d.line(pts, fill=(70, 185, 124) if s["ours"] else (229, 83, 75), width=5)
            cx, cy = s["poly"].centroid.coords[0]
            u, v = px(cx, cy)
            label = f"sun {s['pct']}%"
            d.rectangle(d.textbbox((u * 2, v * 2), label, font=font, anchor="mm"), fill=(11, 13, 16))
            d.text((u * 2, v * 2), label, fill=(237, 234, 228), font=font, anchor="mm")
        big.thumbnail((900, 900))
        big.save(path, quality=88)
        results.append({"n": n, "image": f"views/{path.name}", "before": f"views/{before_path.name}",
                        "lat": round(c.y, 6), "lon": round(c.x, 6), "shape": r["shape"],
                        "pv_area_sqft": r["pv_area_sqft"], "pv_count": r["pv_count"], "pv_score": r["pv_score"],
                        "collection": col["collection"],
                        "sides": [{"facing": s["facing"], "pct": s["pct"], "ours": s["ours"],
                                   "panel_pixels_pct": s["panel_pixels_pct"]} for s in sides]})
        print(f"  roof {n}: {r['shape']}, sides " + ", ".join(
            f"{s['facing']:.0f}deg {s['pct']}% ours={s['ours']} blue={s['panel_pixels_pct']}%" for s in sides))
    out = {"city": "Los Angeles", "best_kwh_m2_year": best_year, "keep_pct": KEEP_PERCENT, "roofs": results}
    (PROJECT_ROOT / "frontend" / "solar" / "installs_check.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
