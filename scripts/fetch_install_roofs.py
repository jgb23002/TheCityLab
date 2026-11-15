"""Fetch Vexcel roof data for LA house neighbourhoods so the real-install test has more roofs.

Vexcel reports existing solar panels (roof-objects.solar-panel-photovoltaic) only on small requests:
a 120 m square returns the field, a 300 m+ square (500-parcel cap) comes back without it. The tiles
below are single-family areas in central / south / west LA, picked by map before looking at any
results. Every response is cached under data/cache/vexcel/, where validate_installs.py finds it.
Run: python scripts/fetch_install_roofs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapely.geometry import box  # noqa: E402

from backend.tools import vexcel  # noqa: E402

HALF_M = 60   # 120 m squares: on bigger requests Vexcel caps at 500 parcels and drops the solar field
GRID = 3      # a 3 x 3 grid of squares around each neighbourhood centre
TILES = {
    "Mid-City": (34.040, -118.355),
    "Leimert Park": (34.005, -118.330),
    "View Park": (33.995, -118.345),
    "Baldwin Hills": (34.010, -118.360),
    "Mar Vista": (34.002, -118.430),
    "Westchester": (33.960, -118.405),
    "Palms": (34.025, -118.410),
    "Cheviot Hills": (34.040, -118.410),
    "Hancock Park": (34.070, -118.330),
    "Los Feliz": (34.110, -118.290),
    "Eagle Rock": (34.140, -118.205),
    "Highland Park": (34.115, -118.190),
    "El Sereno": (34.080, -118.180),
    "Hyde Park": (33.985, -118.320),
    "Inglewood": (33.960, -118.350),
    "Culver City": (34.010, -118.395),
}


def main() -> None:
    import math
    total = 0
    for name, (lat0, lon0) in TILES.items():
        found = 0
        for i in range(GRID):
            for j in range(GRID):
                lat = lat0 + (i - GRID // 2) * 2 * HALF_M / 111_320
                lon = lon0 + (j - GRID // 2) * 2 * HALF_M / (111_320 * math.cos(math.radians(lat0)))
                area = box(*vexcel._square_bounds(lat, lon, half_m=HALF_M)).wkt
                try:
                    d = vexcel._client.request_json("GET", "property/extract", {
                        "wkt": area, "srid": 4326, "layer": vexcel.LAYERS, "latest": "true",
                        "include": "precomputed", "unit-system": "imperial"}, timeout=120)
                except Exception as exc:
                    print(f"{name} {i},{j}: error {type(exc).__name__}")
                    continue
                for f in d.get("features") or []:
                    for st in (f.get("properties") or {}).get("structures") or []:
                        a = st.get("roof-objects.solar-panel-photovoltaic.area")
                        if a is not None and float(a) > 0 and st.get("roof-characteristics.shape.value") in ("gable", "hip"):
                            found += 1
        total += found
        print(f"{name}: {found} sloped roofs with panels (may repeat across touching squares)")
    print("total", total)


if __name__ == "__main__":
    main()
