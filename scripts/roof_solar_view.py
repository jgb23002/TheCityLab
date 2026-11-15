"""Measure building height, roof slope and per-roof-side solar output from one aerial photo.

Run ``python scripts/roof_solar_view.py --id 10572052``.

Everything is measured from the Vexcel top-down photo with geometry; no
model is used:

1. **Sun at capture time.** The direction shadows point is found by testing
   every compass direction for the darkest strip next to the buildings. Sun
   direction = shadow direction - 180 deg; the time of day on the capture date
   with that sun direction gives the sun height (``backend/sun.py``).
2. **Wall (eave) height** = shadow length x tan(sun height), from shadow rays
   cast off the walls that face away from the sun.
3. **Ridge height** from the tip of the triangular shadow a gable end casts;
   **roof slope** = atan((ridge - eave) / (building width / 2)).
4. **Second slope estimate from shading**: the roof side facing the sun is
   brighter; the brightness ratio of the two sides fixes the slope.
5. **Solar per roof side**: each side's own tilt and facing direction go to
   NREL PVWatts (typical-year weather, so cloud cover is included).

Vexcel's precomputed data gives footprints and roof shape but no heights
for these buildings (0 of 505 structures), and no elevation image exists,
which is why heights are measured from shadows here.

Output: ``data/cache/roof_view/property_<id>_report.jpg`` (+ ``.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import shapely  # noqa: E402
from shapely import wkt as shapely_wkt  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from backend import config, sun, vexcel_detect  # noqa: E402
from backend.samples import sample_buildings  # noqa: E402
from backend.tools import solar, vexcel  # noqa: E402
from scripts.measure_sample import lot_sizes  # noqa: E402

M_PER_DEG_LAT = 111_320.0
FLOOR_TO_FLOOR_M = 3.0  # assumption for the story check (about 10 ft)
DIFFUSE_SHARE = (0.1, 0.2, 0.3)  # clear-sky diffuse / direct light, low-mid-high
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------ coordinates

class Frame:
    """Local metres (x east, y north) <-> image pixels for a north-up image."""

    def __init__(self, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> None:
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds
        self.w, self.h = size
        self.lat0 = (self.min_lat + self.max_lat) / 2
        self.m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(self.lat0))
        self.width_m = (self.max_lon - self.min_lon) * self.m_per_deg_lon
        self.height_m = (self.max_lat - self.min_lat) * M_PER_DEG_LAT
        self.mpp_x, self.mpp_y = self.width_m / self.w, self.height_m / self.h

    def to_m(self, lon: float, lat: float) -> tuple[float, float]:
        return ((lon - self.min_lon) * self.m_per_deg_lon, (lat - self.min_lat) * M_PER_DEG_LAT)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        return x / self.mpp_x, self.h - y / self.mpp_y

    def poly_m(self, geom_wkt: str) -> Polygon:
        g = shapely_wkt.loads(geom_wkt)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        return Polygon([self.to_m(lon, lat) for lon, lat in g.exterior.coords])

    def poly_px(self, poly: Polygon) -> list[tuple[float, float]]:
        return [self.to_px(x, y) for x, y in poly.exterior.coords]


def unit(az_deg: float) -> np.ndarray:
    """Compass direction (0 = north, 90 = east) as an (east, north) unit vector."""
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a)])


# ------------------------------------------------------------ imagery

def ortho_image(area_wkt: str, collection: str, layer: str) -> Image.Image:
    data = vexcel._client.request_bytes("POST", "ortho/extract", {
        "wkt": area_wkt, "srid": 4326, "layer": layer, "collection": collection,
        "buffer-meter": 0, "crop": "bounding_box", "image-format": "jpeg",
    })
    return Image.open(io.BytesIO(data)).convert("RGB")


def oblique_image(area_wkt: str, direction: str) -> Image.Image | None:
    try:
        data = vexcel._client.request_bytes("POST", "oriented/extract", {
            "wkt": area_wkt, "srid": 4326, "layer": "urban", "product-type": direction,
            "rotation": "natural", "buffer-meter": 10, "crop": "bbox", "image-format": "jpeg",
        })
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        print(f"oblique {direction} unavailable ({type(exc).__name__})")
        return None


def linear_luminance(img: Image.Image) -> np.ndarray:
    """Undo JPEG gamma so brightness is proportional to light."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    lin = np.power(a, 2.2)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256, range=(0, 1))
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(edges[np.nanargmax(between)])


def mask_of(polys: list[Polygon], frame: Frame, scale: float = 1.0) -> np.ndarray:
    m = Image.new("L", (int(frame.w * scale), int(frame.h * scale)), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        d.polygon([(x * scale, y * scale) for x, y in frame.poly_px(p)], fill=255)
    return np.asarray(m) > 0


# ------------------------------------------------------------ sun from shadows

def shadow_direction(polys: list[Polygon], dark: np.ndarray, frame: Frame) -> float:
    """Compass direction shadows point, from the darkest 2.5 m strip next to the buildings."""
    scale = 0.25
    small_dark = np.asarray(Image.fromarray(dark.astype(np.uint8) * 255).resize(
        (int(frame.w * scale), int(frame.h * scale)))) > 127
    footprint = mask_of(polys, frame, scale)
    best = (-1.0, 0.0)
    for az in range(0, 360, 2):
        v = unit(az) * 2.5
        moved = [Polygon([(x + v[0], y + v[1]) for x, y in p.exterior.coords]) for p in polys]
        strip = mask_of(moved, frame, scale) & ~footprint
        if strip.sum() > 50:
            best = max(best, (float(small_dark[strip].mean()), float(az)))
    return best[1]


def vexcel_capture_time(lat: float, lon: float, collection: str, date: str) -> tuple[datetime, str]:
    """Capture time of *collection* at this point from Vexcel's coverage record.

    Vexcel lists the first and last capture time of the collection at the
    point; the one on the photo's estimated date is used. If neither matches,
    solar noon on that date is used and labelled as an assumption.
    """
    data = vexcel._client.request_json("GET", "ortho/collections", {
        "wkt": f"POINT({lon} {lat})", "srid": 4326, "layer": vexcel.LAYERS,
        "include": "collection,first-capture-date,last-capture-date,layer"})
    for f in data.get("features") or []:
        p = (f or {}).get("properties") or {}
        if p.get("collection") != collection:
            continue
        for key in ("first-capture-date", "last-capture-date"):
            if str(p.get(key, ""))[:10] == date:
                return datetime.fromisoformat(p[key]).replace(tzinfo=timezone.utc), f"Vexcel {key}"
    noon = sun._day_facts(lat, lon, datetime.fromisoformat(date).date())["solar_noon_utc"]
    when = datetime.fromisoformat(f"{date}T{noon}").replace(tzinfo=timezone.utc)
    return when, "assumed solar noon (Vexcel gave no time for this date)"


def sun_from_shadow(shadow_az: float, lat: float, lon: float, day: datetime) -> dict[str, Any]:
    """Time on *day* when the sun stands opposite the shadows, and its height then."""
    target = (shadow_az + 180) % 360
    best = None
    for minute in range(0, 24 * 60, 2):
        when = day.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc) + timedelta(minutes=minute)
        pos = sun.sun_position(lat, lon, when)
        if pos["elevation_deg"] <= 5:
            continue
        err = abs((pos["azimuth_deg"] - target + 180) % 360 - 180)
        if best is None or err < best[0]:
            best = (err, when, pos)
    _, when, pos = best
    return {"sun_azimuth_deg": pos["azimuth_deg"], "sun_elevation_deg": pos["elevation_deg"],
            "time_utc": when.strftime("%Y-%m-%d %H:%M"), "shadow_azimuth_deg": shadow_az}


# ------------------------------------------------------------ heights

JUMP = 2.5  # shadow -> sunlit ground is at least this many times brighter (same surface ~4-6x)


def shadow_rays(poly: Polygon, footprints: np.ndarray, lum: np.ndarray, dark: np.ndarray,
                frame: Frame, s: np.ndarray, step_m: float = 0.75) -> list[dict[str, Any]]:
    """Shadow length from points along walls facing away from the sun.

    Each ray starts at the wall and walks away from the sun. It must start in
    shadow; the shadow ends at the first sharp brightness jump (x2.5 that
    holds for 0.5 m). Rays that reach another building or the image edge
    are dropped. A fixed dark/light cut-off is not used because unlit grass
    is as dark as shadowed concrete.
    """
    rays = []
    coords = list(poly.exterior.coords)
    ccw = poly.exterior.is_ccw
    px_step = min(frame.mpp_x, frame.mpp_y)
    dists = np.arange(0.3, 40, px_step)
    win = max(1, int(0.3 / px_step))
    hold = max(1, int(0.5 / px_step))
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        edge = np.array([x2 - x1, y2 - y1])
        length = float(np.hypot(*edge))
        if length < 3:
            continue
        normal = np.array([edge[1], -edge[0]]) / length  # outward for CCW rings
        if not ccw:
            normal = -normal
        if normal @ s < 0.3:
            continue
        for t in np.arange(1.0, length - 1.0, step_m):
            start = np.array([x1, y1]) + edge / length * t
            pts = start[None, :] + dists[:, None] * s[None, :]
            pxs = pts[:, 0] / frame.mpp_x
            pys = frame.h - pts[:, 1] / frame.mpp_y
            inside = (pxs >= 0) & (pxs < frame.w) & (pys >= 0) & (pys < frame.h)
            n = int(np.argmin(inside)) if not inside.all() else len(dists)
            if n < hold * 4:
                continue
            ix, iy = pxs[:n].astype(int), pys[:n].astype(int)
            hits = footprints[iy, ix]
            start_dark = dark[iy[: 2 * win], ix[: 2 * win]].mean() > 0.7
            if not start_dark:
                continue  # no shadow at the wall: tree, light, or wall faces the sun
            prof = np.convolve(lum[iy, ix], np.ones(win) / win, mode="same")
            base = float(np.median(prof[: 3 * win]))
            bright = prof > base * JUMP
            end = None
            for k in range(2 * win, n - hold):
                if hits[k]:
                    break  # reached another building before the shadow ended
                if bright[k] and bright[k:k + hold].mean() > 0.8:
                    end = k
                    break
            if end is not None:
                rays.append({"t": float(t), "edge_dir": edge / length, "length_m": float(dists[end])})
    return rays


def ridge_axis(poly: Polygon) -> tuple[np.ndarray, float, float]:
    """Unit vector along the long side of the minimum rectangle, its length and width (m)."""
    rect = list(poly.minimum_rotated_rectangle.exterior.coords)
    e1 = np.array(rect[1]) - np.array(rect[0])
    e2 = np.array(rect[2]) - np.array(rect[1])
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    return (e1 / l1, l1, l2) if l1 >= l2 else (e2 / l2, l2, l1)


def measure_heights(poly: Polygon, rays: list[dict[str, Any]], tan_e: float) -> dict[str, Any]:
    axis, length, width = ridge_axis(poly)
    eave = [r["length_m"] for r in rays if abs(r["edge_dir"] @ axis) > 0.9]  # walls along the ridge
    gable = [r["length_m"] for r in rays if abs(r["edge_dir"] @ axis) < 0.4]  # gable ends
    out: dict[str, Any] = {"length_m": round(length, 1), "width_m": round(width, 1),
                           "eave_rays": len(eave), "gable_rays": len(gable)}
    eave_h = ridge_h = None
    if len(eave) >= 8:
        q1, med, q3 = np.percentile(eave, [25, 50, 75])
        out["eave_spread"] = round(float((q3 - q1) / med), 2)
        if (q3 - q1) / med <= 0.35:  # rays agree -> trust the number
            eave_h = float(med) * tan_e
    if len(gable) >= 5:
        ridge_h = float(np.percentile(gable, 85)) * tan_e
    out["eave_height_m"] = round(eave_h, 1) if eave_h else None
    out["ridge_height_m"] = round(ridge_h, 1) if ridge_h else None
    if eave_h and ridge_h and ridge_h > eave_h:
        slope = math.degrees(math.atan((ridge_h - eave_h) / (width / 2)))
        if 5 <= slope <= 45:
            out["slope_from_shadow_deg"] = round(slope, 1)
    return out


# ------------------------------------------------------------ roof sides

def roof_sides(poly: Polygon) -> list[dict[str, Any]]:
    """Split a gable footprint along its ridge into two sides with their facing directions."""
    axis, _, _ = ridge_axis(poly)
    c = np.array(poly.centroid.coords[0])
    normal = np.array([axis[1], -axis[0]])
    big = 1000
    sides = []
    for sign in (1, -1):
        n = normal * sign
        half = Polygon([c - axis * big, c + axis * big, c + axis * big + n * big, c - axis * big + n * big])
        part = poly.intersection(half)
        if part.is_empty:
            continue
        if part.geom_type != "Polygon":
            part = max(part.geoms, key=lambda g: g.area)
        sides.append({"poly": part, "facing_deg": round(math.degrees(math.atan2(n[0], n[1])) % 360, 1)})
    return sides


def side_brightness(part: Polygon, lum: np.ndarray, frame: Frame) -> float | None:
    inner = part.buffer(-0.8)
    if inner.is_empty or inner.area < 4:
        return None
    m = mask_of([inner] if inner.geom_type == "Polygon" else list(inner.geoms), frame)
    return float(np.median(lum[m])) if m.sum() > 50 else None


def slope_from_shading(ratio: float, facing: tuple[float, float], sun_az: float, sun_el: float,
                       diffuse: float) -> float | None:
    """Slope where (cos i_a + d) / (cos i_b + d) matches the brightness ratio of sides a/b."""
    def predicted(p: float) -> float:
        pr, er = math.radians(p), math.radians(sun_el)
        d = diffuse * (1 + math.cos(pr)) / 2

        def cos_i(face: float) -> float:
            return max(0.0, math.cos(pr) * math.sin(er)
                       + math.sin(pr) * math.cos(er) * math.cos(math.radians(sun_az - face)))
        return (cos_i(facing[0]) + d) / (cos_i(facing[1]) + d)

    grid = [(abs(predicted(p) - ratio), p) for p in np.arange(0, 60.5, 0.5)]
    err, best = min(grid)
    return round(float(best), 1) if err < 0.05 * ratio else None


# ------------------------------------------------------------ solar

# A common 400 W panel, 1.13 m x 1.72 m, mounted flush with the roof, long side down the slope.
PANEL_W_M, PANEL_L_M, PANEL_KW = 1.13, 1.72, 0.40
PANEL_AREA_M2 = PANEL_W_M * PANEL_L_M
GAP_M = 0.02
SETBACK_M = 0.91  # 3 ft fire-access path from roof edges and ridge (IFC/IRC style)
PERFORMANCE_RATIO = 0.86  # 14% losses (wiring, inverter, heat, dirt), same as solar.py
KEEP_PERCENT = 80  # put panels only where yearly sun >= 80% of the best possible


def layout_panels(side: Polygon, facing_deg: float, tilt_deg: float,
                  row_cover: float = 1.0) -> list[Polygon]:
    """Fit as many whole panels as possible on one roof side, after the fire setback.

    Panels run in rows along the ridge; seen from above, the down-slope length
    shrinks by cos(tilt). *row_cover* < 1 leaves walkway/shade gaps between rows
    (tilted racks on a flat roof: about 0.7 of the row area holds panels).
    """
    inner = side.buffer(-SETBACK_M, join_style=2)
    if inner.is_empty:
        return []
    if inner.geom_type != "Polygon":
        inner = max(inner.geoms, key=lambda g: g.area)
    down = unit(facing_deg)  # down the slope, seen from above
    along = np.array([down[1], -down[0]])
    proj_len = PANEL_L_M * math.cos(math.radians(tilt_deg))
    pts = np.array(inner.exterior.coords)
    u, v = pts @ along, pts @ down
    best: list[Polygon] = []
    for du in (0.0, PANEL_W_M / 2):  # try two offsets, keep the better one
        for dv in (0.0, proj_len / 2):
            got = []
            for u0 in np.arange(u.min() + du, u.max() - PANEL_W_M + 1e-6, PANEL_W_M + GAP_M):
                for v0 in np.arange(v.min() + dv, v.max() - proj_len + 1e-6, proj_len / row_cover + GAP_M):
                    corners = [along * uu + down * vv for uu, vv in
                               ((u0, v0), (u0 + PANEL_W_M, v0), (u0 + PANEL_W_M, v0 + proj_len), (u0, v0 + proj_len))]
                    rect = Polygon(corners)
                    if inner.contains(rect):
                        got.append(rect)
            if len(got) > len(best):
                best = got
    return best


# Panel sizes we can lay out. Typical datasheet sizes for 182 mm-cell modules (not one brand):
# 108 cells ~ 1722 x 1134 mm (~400 W), 144 cells ~ 2278 x 1134 mm (~550 W).
PANEL_TYPES: dict[str, dict[str, Any]] = {
    "400w": {"name": "400 W", "length_m": 1.722, "width_m": 1.134, "kw": 0.40},
    "550w": {"name": "550 W", "length_m": 2.278, "width_m": 1.134, "kw": 0.55},
}
WINTER_DECLINATION_DEG = -23.44  # Dec 21, the shortest day


def winter_row_gap(lat: float, tilt_deg: float, slope_len_m: float, facing_deg: float) -> float:
    """Gap between tilted rows on a flat roof so no row shades the next one from 10 am to
    2 pm (solar time) on Dec 21, the day the sun is lowest. Plain sun geometry."""
    if tilt_deg <= 0:
        return GAP_M
    phi, dec = math.radians(lat), math.radians(WINTER_DECLINATION_DEG)
    rise = slope_len_m * math.sin(math.radians(tilt_deg))
    worst = 0.0
    for hour_angle in (-30.0, 30.0):  # 10 am and 2 pm
        h = math.radians(hour_angle)
        sin_el = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(h)
        el = math.asin(max(-1.0, min(1.0, sin_el)))
        if el <= math.radians(3):
            continue
        # sun azimuth from north, clockwise
        az = math.degrees(math.atan2(math.sin(h), math.cos(h) * math.sin(phi) - math.tan(dec) * math.cos(phi))) + 180
        along = max(0.0, math.cos(math.radians(az - facing_deg)))  # shadow length along the row direction
        worst = max(worst, rise * along / math.tan(el))
    return max(GAP_M, worst)


def fit_panels(side: Polygon, facing_deg: float, tilt_deg: float, panel: dict[str, Any] | None = None,
               row_gap_m: float = GAP_M) -> dict[str, Any]:
    """Pack as many whole panels as fit on one roof side, inside the fire setback.

    Tries the panel upright (long side down the slope) and sideways, and 4 x 4 grid
    starting offsets for each, then keeps the arrangement with the most panels. Rows run
    across the slope; *row_gap_m* is the clear gap between rows (seen from above).
    Returns {"panels": [Polygon], "orientation", "rows", "row_gap_m", "panel"}.
    """
    panel = panel or PANEL_TYPES["400w"]
    empty = {"panels": [], "orientation": None, "rows": 0, "row_gap_m": round(row_gap_m, 2), "panel": panel}
    inner = side.buffer(-SETBACK_M, join_style=2)
    if inner.is_empty:
        return empty
    if inner.geom_type != "Polygon":
        inner = max(inner.geoms, key=lambda g: g.area)
    shapely.prepare(inner)
    down = unit(facing_deg)  # down the slope, seen from above
    along = np.array([down[1], -down[0]])
    pts = np.array(inner.exterior.coords)
    u, v = pts @ along, pts @ down
    cos_t = math.cos(math.radians(tilt_deg))
    best = empty
    for orientation in ("portrait", "landscape"):
        across, slope = ((panel["width_m"], panel["length_m"]) if orientation == "portrait"
                         else (panel["length_m"], panel["width_m"]))
        proj = slope * cos_t  # panel depth seen from above
        step_u, step_v = across + GAP_M, proj + row_gap_m
        for du in np.linspace(0, step_u, 4, endpoint=False):
            for dv in np.linspace(0, step_v, 4, endpoint=False):
                us = np.arange(u.min() + du, u.max() - across + 1e-6, step_u)
                vs = np.arange(v.min() + dv, v.max() - proj + 1e-6, step_v)
                if not len(us) or not len(vs):
                    continue
                uu, vv = np.meshgrid(us, vs)
                uu, vv = uu.ravel(), vv.ravel()
                corners = np.stack([
                    np.outer(uu, along) + np.outer(vv, down),
                    np.outer(uu + across, along) + np.outer(vv, down),
                    np.outer(uu + across, along) + np.outer(vv + proj, down),
                    np.outer(uu, along) + np.outer(vv + proj, down),
                ], axis=1)
                rects = shapely.polygons(corners)
                ok = shapely.contains(inner, rects)
                if ok.sum() > len(best["panels"]):
                    best = {"panels": list(rects[ok]), "orientation": orientation,
                            "rows": int(len(np.unique(np.round(vv[ok], 3)))), "row_gap_m": round(row_gap_m, 2),
                            "panel": panel}
    return best


def neighbour_shade(side: Polygon, near: list[tuple[int, Polygon, float]], my_height: float,
                    steps: list[dict[str, Any]]) -> list[float]:
    """Share of the roof side in a taller neighbour's shadow at every time step (0-1)."""
    taller = [(p, h - my_height) for _, p, h in near if h - my_height > 0.3]
    out = []
    for st in steps:
        if not taller or st["elevation_deg"] < 3:
            out.append(0.0)
            continue
        d = unit(st["azimuth_deg"] + 180) / math.tan(math.radians(st["elevation_deg"]))
        covered = 0.0
        for p, dh in taller:
            moved = Polygon([(x + d[0] * dh, y + d[1] * dh) for x, y in p.exterior.coords])
            shadow = p.union(moved).convex_hull
            if shadow.intersects(side):
                covered = max(covered, shadow.intersection(side).area / side.area)
        out.append(min(covered, 1.0))
    return out


def pvwatts_kwh_per_kw(lat: float, lon: float, tilt: float, azimuth: float) -> float | None:
    """NREL PVWatts yearly kWh per installed kW for one tilt/direction (cached)."""
    params = {"system_capacity": 1, "module_type": 0, "losses": solar.SYSTEM_LOSSES_PCT,
              "array_type": 1, "tilt": round(tilt), "azimuth": round(azimuth / 5) * 5 % 360,
              "lat": round(lat, 3), "lon": round(lon, 3)}
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
    path = config.CACHE_DIR / "nrel" / f"pvwatts_plane_{digest}.json"
    if path.exists():
        return json.loads(path.read_text())["ac_annual"]
    if config.USE_CACHE_ONLY or not config.NREL_API_KEY:
        return None
    try:
        resp = requests.get(solar.PVWATTS_URL, params={**params, "api_key": config.NREL_API_KEY}, timeout=30)
        resp.raise_for_status()
        ac = float(resp.json()["outputs"]["ac_annual"])
    except Exception as exc:  # never print the URL: it holds the key
        print(f"PVWatts failed ({type(exc).__name__})")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "ac_annual": ac}))
    return ac


# ------------------------------------------------------------ drawing

def color_scale(value: float, lo: float, hi: float) -> tuple[int, int, int]:
    """Red (low) -> yellow -> green (high)."""
    t = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 1.0
    if t < 0.5:
        return 215, int(60 + 330 * t), 50
    return int(215 - 330 * (t - 0.5)), 200, 60


def height_color(h: float, lo: float, hi: float) -> tuple[int, int, int]:
    """Dark blue (low) -> cyan -> white (high)."""
    t = max(0.0, min(1.0, (h - lo) / (hi - lo))) if hi > lo else 0.5
    return int(30 + 225 * t), int(70 + 185 * t), int(160 + 95 * t)


def label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int) -> None:
    f = font(size)
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=f, anchor="mm")
    draw.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(0, 0, 0, 170))
    draw.text(xy, text, font=f, fill=(255, 255, 255, 255), anchor="mm")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], az: float, length: float,
          color: tuple[int, int, int], width: int = 6) -> None:
    v = unit(az)
    end = (start[0] + v[0] * length, start[1] - v[1] * length)
    draw.line([start, end], fill=color, width=width)
    for side in (150, -150):
        w = unit(az + side)
        draw.line([end, (end[0] + w[0] * length * 0.25, end[1] - w[1] * length * 0.25)], fill=color, width=width)


# ------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, default=10572052)
    parser.add_argument("--city", default="houston")
    args = parser.parse_args(argv)

    b = next(x for x in sample_buildings(args.city, 11, n=1000) if x.id == args.id)
    lot = lot_sizes([b.id]).get(b.id)
    parcel = vexcel_detect.property_parcel(b.latitude, b.longitude, lot) if lot else None
    if parcel is None:
        raise SystemExit("No Vexcel parcel matches this property's lot size.")
    props = parcel["properties"]
    structures = [s for s in props.get("structures") or [] if s.get("footprint.wkt")]

    # Newest top-down capture at this property, and the date Vexcel estimates for it.
    cols = vexcel_detect.list_collections(b.latitude, b.longitude)
    col = next(c for c in cols if c["layer"] in ("urban", "urban-r"))
    est = props.get("metadata.ortho-date-estimate") or col["date"]

    area = shapely_wkt.loads(parcel["wkt"])
    pad = 15 / M_PER_DEG_LAT
    bounds = (area.bounds[0] - pad, area.bounds[1] - pad, area.bounds[2] + pad, area.bounds[3] + pad)
    img = ortho_image(box(*bounds).wkt, col["collection"], col["layer"])
    frame = Frame(bounds, img.size)
    polys = [frame.poly_m(s["footprint.wkt"]) for s in structures]
    lum = linear_luminance(img)
    footprint = mask_of(polys, frame)
    threshold = otsu(lum[~footprint])
    dark = lum < threshold

    # 1. Sun at capture time from latitude/longitude + Vexcel's capture time (NOAA math).
    centre = area.centroid
    site_lat, site_lon = centre.y, centre.x
    capture_time, time_source = vexcel_capture_time(site_lat, site_lon, col["collection"], est[:10])
    pos = sun.sun_position(site_lat, site_lon, capture_time)
    vexcel_shadow_az = round((pos["azimuth_deg"] + 180) % 360, 1)
    # Check: which way do the shadows in the photo actually point?
    shadow_az = shadow_direction(polys, dark, frame)
    gap = abs((vexcel_shadow_az - shadow_az + 180) % 360 - 180)
    if gap > 5:
        # Vexcel's time is for the collection at this spot, not this exact image. Keep the
        # lat/lon sun math; only move the clock to the time when it matches the photo's shadows.
        matched = sun_from_shadow(shadow_az, site_lat, site_lon, capture_time)
        capture_time = datetime.fromisoformat(matched["time_utc"]).replace(tzinfo=timezone.utc)
        time_source = (f"clock time matched to photo shadows; Vexcel's {time_source.split()[-1]} "
                       f"gave shadows {gap:.0f}° off")
        pos = sun.sun_position(site_lat, site_lon, capture_time)
    sun_now = {"sun_azimuth_deg": pos["azimuth_deg"], "sun_elevation_deg": pos["elevation_deg"],
               "time_utc": capture_time.strftime("%Y-%m-%d %H:%M"), "time_source": time_source,
               "latitude": round(site_lat, 5), "longitude": round(site_lon, 5),
               "photo_shadow_azimuth_deg": shadow_az, "vexcel_time_shadow_azimuth_deg": vexcel_shadow_az,
               "math_shadow_azimuth_deg": round((pos["azimuth_deg"] + 180) % 360, 1)}
    tan_e = math.tan(math.radians(sun_now["sun_elevation_deg"]))
    s = unit(sun_now["sun_azimuth_deg"] + 180)  # shadows point away from the sun
    path = sun.sun_summary(site_lat, site_lon)

    # 2-4. Heights and slope per building; 5. solar per roof side.
    best_kwh = pvwatts_kwh_per_kw(b.latitude, b.longitude, sun.sun_summary(b.latitude, b.longitude)["best_fixed_tilt_deg"], 180)
    flat_kwh = pvwatts_kwh_per_kw(b.latitude, b.longitude, solar.PANEL_TILT_DEG, solar.PANEL_AZIMUTH_DEG)
    usable = solar.USABLE_FRACTION[None]
    buildings, all_rays, side_sets = [], [], []
    for i, (st, poly) in enumerate(zip(structures, polys)):
        rays = shadow_rays(poly, footprint, lum, dark, frame, s)
        all_rays.append(rays)
        h = measure_heights(poly, rays, tan_e)
        shape = st.get("roof-characteristics.shape.value")
        h.update(index=i + 1, roof_shape=shape, footprint_sq_ft=round(st.get("footprint.area") or poly.area * 10.7639),
                 ground_elevation_ft=st.get("ground-elevation.value"))
        if h["eave_height_m"]:
            h["stories_estimate"] = round(h["eave_height_m"] / FLOOR_TO_FLOOR_M, 1)

        sides = roof_sides(poly) if shape in ("gable", "hip") else []
        if len(sides) == 2:
            bright = [side_brightness(sd["poly"], lum, frame) for sd in sides]
            facing = (sides[0]["facing_deg"], sides[1]["facing_deg"])
            # Shading only tells the slope when one side faces the sun and the other faces away.
            h["shading_sensitive"] = abs(math.cos(math.radians(sun_now["sun_azimuth_deg"] - facing[0]))) > 0.6
            if all(bright):
                fits = [slope_from_shading(bright[0] / bright[1], facing, sun_now["sun_azimuth_deg"],
                                           sun_now["sun_elevation_deg"], d) for d in DIFFUSE_SHARE]
                if fits[1] is not None:
                    h["slope_from_shading_deg"] = fits[1]
        side_sets.append(sides)
        buildings.append(h)

    # One slope for the complex: buildings built together share a roof pitch; use only reliable readings.
    readings = [bd["slope_from_shadow_deg"] for bd in buildings if bd.get("slope_from_shadow_deg")]
    readings += [bd["slope_from_shading_deg"] for bd in buildings
                 if bd.get("shading_sensitive") and bd.get("slope_from_shading_deg")
                 and 5 <= bd["slope_from_shading_deg"] <= 45]
    complex_slope = round(float(np.median(readings)), 1) if readings else None
    slope_range = [round(float(v), 1) for v in np.percentile(readings, [25, 75])] if len(readings) >= 3 else None

    # Heights for shading: measured walls; buildings marked "?" get the median measured wall.
    measured = [bd["eave_height_m"] for bd in buildings if bd.get("eave_height_m")]
    default_eave = float(np.median(measured)) if measured else FLOOR_TO_FLOOR_M
    for i, (st, poly, h, sides) in enumerate(zip(structures, polys, buildings, side_sets)):
        slope = h.get("slope_from_shadow_deg") or complex_slope
        h["slope_used_deg"] = slope if sides else None
        h["slope_source"] = ("own shadow" if h.get("slope_from_shadow_deg") else
                             "complex median" if sides and complex_slope else "flat")
        eave = h.get("eave_height_m") or default_eave
        rise = (ridge_axis(poly)[2] / 2) * math.tan(math.radians(slope or 0)) if sides else 0.0
        h["height_for_shading_m"] = {"eave": round(eave, 1), "ridge": round(eave + rise, 1),
                                     "eave_assumed": not h.get("eave_height_m")}

    # Whole-year sun at this latitude/longitude, with NASA's measured monthly weather.
    steps = sun.sky_year(site_lat, site_lon)
    best_tilt = sun.sun_summary(site_lat, site_lon)["best_fixed_tilt_deg"]
    best_year = sun.plane_sunlight(steps, best_tilt, 180)["kwh_m2_year"]

    sides_out = []
    for i, (st, poly, h, sides) in enumerate(zip(structures, polys, buildings, side_sets)):
        if h["roof_shape"] == "flat" or not sides:
            sides = [{"poly": poly, "facing_deg": 180.0}]
            tilt = solar.PANEL_TILT_DEG
        else:
            tilt = h["slope_used_deg"] or 0.0
        my_mid = h["height_for_shading_m"]["eave"] + (h["height_for_shading_m"]["ridge"] - h["height_for_shading_m"]["eave"]) / 2
        near = [(j, polys[j], buildings[j]["height_for_shading_m"]["ridge"]) for j in range(len(polys))
                if j != i and polys[j].distance(poly) < 40]
        for sd in sides:
            shade = neighbour_shade(sd["poly"], near, my_mid, steps)
            year = sun.plane_sunlight(steps, tilt, sd["facing_deg"], shade)
            panels = layout_panels(sd["poly"], sd["facing_deg"], tilt)
            kwh_pvw = pvwatts_kwh_per_kw(b.latitude, b.longitude, tilt, sd["facing_deg"])
            plane_m2 = sd["poly"].area / max(math.cos(math.radians(tilt if h["roof_shape"] != "flat" else 0)), 0.2)
            pct = round(100 * year["kwh_m2_year"] / best_year)
            kw = len(panels) * PANEL_KW
            sides_out.append({
                "building": i + 1, "facing_deg": sd["facing_deg"], "tilt_deg": round(tilt, 1),
                "surface_m2": round(plane_m2, 1), "sun_kwh_m2_year": year["kwh_m2_year"],
                "shade_loss_kwh_m2": year["shade_loss_kwh_m2"], "vs_best_percent": pct,
                "panels_fit": len(panels), "panel_area_m2": round(len(panels) * PANEL_AREA_M2, 1),
                "capacity_kw": round(kw, 1),
                # Energy: NREL PVWatts yield for this tilt/direction (it also counts heat and glass
                # losses), cut by the neighbour shade found above; our own model if PVWatts is unavailable.
                "annual_kwh": round(kw * kwh_pvw * year["kwh_m2_year"]
                                    / (year["kwh_m2_year"] + year["shade_loss_kwh_m2"])) if kwh_pvw
                else round(kw * year["kwh_m2_year"] * PERFORMANCE_RATIO),
                "pvwatts_kwh_per_kw": round(kwh_pvw) if kwh_pvw else None,
                "chosen": pct >= KEEP_PERCENT, "poly": sd["poly"], "panels": panels})

    chosen = [sd for sd in sides_out if sd["chosen"]]
    footprint_sq_ft = sum(bd["footprint_sq_ft"] for bd in buildings)
    flat_kw = footprint_sq_ft * usable * solar.WATTS_PER_USABLE_SQ_FT / 1000
    ours = [sd["sun_kwh_m2_year"] * PERFORMANCE_RATIO for sd in sides_out]
    pvw = [sd["pvwatts_kwh_per_kw"] for sd in sides_out]
    agree = [o / p for o, p in zip(ours, pvw) if p]
    totals = {
        "roof_sides": len(sides_out), "chosen_sides": len(chosen),
        "roof_surface_m2": round(sum(sd["surface_m2"] for sd in sides_out)),
        "panels": sum(sd["panels_fit"] for sd in chosen),
        "panel_area_m2": round(sum(sd["panel_area_m2"] for sd in chosen)),
        "capacity_kw": round(sum(sd["capacity_kw"] for sd in chosen)),
        "annual_kwh": round(sum(sd["annual_kwh"] for sd in chosen)),
        "all_sides_panels": sum(sd["panels_fit"] for sd in sides_out),
        "all_sides_annual_kwh": round(sum(sd["annual_kwh"] for sd in sides_out)),
        "best_sun_kwh_m2_year": best_year, "best_tilt_deg": best_tilt,
        "flat_assumption_kw": round(flat_kw),
        "flat_assumption_annual_kwh": round(flat_kw * flat_kwh) if flat_kwh else None,
        "best_direction_kwh_per_kw": round(best_kwh) if best_kwh else None,
        "our_model_vs_pvwatts": round(float(np.median(agree)), 2) if agree else None,
        "shading_heights_assumed": sum(1 for bd in buildings if bd["height_for_shading_m"]["eave_assumed"]),
    }
    clim = sun.climate(b.latitude, b.longitude) or {}
    eaves = [bd["eave_height_m"] for bd in buildings if bd.get("eave_height_m")]
    slopes = [bd["slope_used_deg"] for bd in buildings if bd.get("slope_used_deg")]

    # ------------------------------------------------ the picture
    out_dir = config.CACHE_DIR / "roof_view"
    out_dir.mkdir(parents=True, exist_ok=True)
    W = 1400
    scale = W / img.width
    base = img.resize((W, int(img.height * scale)))
    fs = max(16, int(W / 70))

    def px(x: float, y: float) -> tuple[float, float]:
        a, c = frame.to_px(x, y)
        return a * scale, c * scale

    # Panel A: height (depth) map + shadow rays.
    a = base.convert("RGBA")
    ov = Image.new("RGBA", a.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    lo, hi = (min(eaves), max(eaves)) if eaves else (0, 1)
    for bd, poly in zip(buildings, polys):
        col_h = height_color(bd["eave_height_m"], lo, hi) if bd.get("eave_height_m") else (120, 120, 120)
        d.polygon([px(x, y) for x, y in poly.exterior.coords], fill=(*col_h, 150), outline=(255, 255, 255, 255))
    for i, poly in enumerate(polys):  # every 4th shadow ray, only on measured buildings
        if not buildings[i].get("eave_height_m"):
            continue
        for r in all_rays[i][::4]:
            start = None
            for (x1, y1), (x2, y2) in zip(list(poly.exterior.coords)[:-1], list(poly.exterior.coords)[1:]):
                e = np.array([x2 - x1, y2 - y1])
                if np.allclose(e / max(np.hypot(*e), 1e-9), r["edge_dir"]):
                    start = np.array([x1, y1]) + r["edge_dir"] * r["t"]
                    break
            if start is not None:
                d.line([px(*start), px(*(start + s * r["length_m"]))], fill=(255, 220, 0, 255), width=3)
    for bd, poly in zip(buildings, polys):
        c = poly.centroid
        if bd.get("eave_height_m"):
            floors = max(1, round(bd["eave_height_m"] / FLOOR_TO_FLOOR_M))
            txt = f"{bd['eave_height_m']} m = {floors} fl"
        else:
            txt = "?"
        if bd.get("slope_used_deg"):
            txt += f" | {bd['slope_used_deg']:.0f}°"
        label(d, px(c.x, c.y), txt, fs)
    a = Image.alpha_composite(a, ov)
    da = ImageDraw.Draw(a)
    # Sun compass for this latitude/longitude: rise/set directions in June and December.
    cx, cy, rad = W - 150, a.height - 190, 105
    da.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(0, 0, 0, 150), outline=(255, 255, 255, 255), width=2)
    for az_, col_, name in ((path["summer_solstice"]["sunrise_azimuth_deg"], (255, 230, 90), "Jun rise"),
                            (path["summer_solstice"]["sunset_azimuth_deg"], (255, 230, 90), "Jun set"),
                            (path["winter_solstice"]["sunrise_azimuth_deg"], (255, 150, 60), "Dec rise"),
                            (path["winter_solstice"]["sunset_azimuth_deg"], (255, 150, 60), "Dec set"),
                            (180.0, (255, 255, 255), "noon")):
        v = unit(az_)
        da.line([(cx, cy), (cx + v[0] * rad, cy - v[1] * rad)], fill=col_, width=3)
        da.text((cx + v[0] * (rad + 8), cy - v[1] * (rad + 8)), name, font=font(fs - 4), fill=(255, 255, 255),
                anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    arrow(da, (cx, cy), sun_now["sun_azimuth_deg"], rad * 0.8, (255, 60, 60), width=5)
    da.text((cx, cy - rad - 8), "N", font=font(fs), fill=(255, 255, 255), anchor="md", stroke_width=3, stroke_fill=(0, 0, 0))
    label(da, (cx, cy + rad + 40), f"red = sun at photo time: {sun_now['sun_azimuth_deg']:.0f}°, "
          f"{sun_now['sun_elevation_deg']:.0f}° up", fs - 4)

    # Panel B: roof sides coloured by yearly sunlight; panels drawn where it is worth it.
    bimg = base.convert("RGBA")
    ov = Image.new("RGBA", bimg.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for sd in sides_out:
        d.polygon([px(x, y) for x, y in sd["poly"].exterior.coords],
                  fill=(*color_scale(sd["vs_best_percent"], 70, 100), 120 if sd["chosen"] else 150),
                  outline=(255, 255, 255, 255))
    for sd in sides_out:
        if sd["chosen"]:
            for rect in sd["panels"]:
                d.polygon([px(x, y) for x, y in rect.exterior.coords], fill=(25, 45, 110, 235),
                          outline=(170, 200, 255, 255))
    for sd in sides_out:
        c = sd["poly"].centroid
        txt = f"{sd['vs_best_percent']}%" + (f" · {sd['panels_fit']}" if sd["chosen"] else "")
        label(d, px(c.x, c.y), txt, fs - 4)
    bimg = Image.alpha_composite(bimg, ov)

    # Panel C: oblique photo (looking along most ridges, so gable slopes show in profile).
    axes = [ridge_axis(p)[0] for p, bd in zip(polys, buildings) if bd["roof_shape"] in ("gable", "hip")]
    ew = sum(abs(ax[0]) for ax in axes) >= sum(abs(ax[1]) for ax in axes)
    direction = "oblique-west" if ew else "oblique-north"
    obl = oblique_image(parcel["wkt"], direction)

    # Compose 2 x 2.
    title_h, cap_h = 70, 46
    ph = a.height
    if obl is not None:
        obl.thumbnail((W, ph))
    ph2 = max(obl.height if obl is not None else 0, 760)
    canvas = Image.new("RGB", (W * 2 + 60, title_h + ph + ph2 + cap_h * 2 + 40), (250, 250, 248))
    cd = ImageDraw.Draw(canvas)
    cd.text((20, 18), f"Property {b.id}: height, roof slope and solar per roof side, measured from the aerial photo",
            font=font(30), fill=(25, 25, 25))
    cd.text((20, title_h), "A. Depth: wall height from shadow length (yellow rays) | roof slope", font=font(22), fill=(25, 25, 25))
    canvas.paste(a.convert("RGB"), (20, title_h + cap_h - 10))
    cd.text((W + 10, title_h), "B. Yearly sun on each roof side (% of best) · number of panels that fit (blue)",
            font=font(22), fill=(25, 25, 25))
    canvas.paste(bimg.convert("RGB"), (W + 10, title_h + cap_h - 10))
    y2 = title_h + ph + cap_h + 20
    if obl is not None:
        cd.text((20, y2), f"C. Oblique photo ({direction.replace('-', ' ')}): check the roof slant by eye",
                font=font(22), fill=(25, 25, 25))
        canvas.paste(obl, (20, y2 + cap_h - 10))

    lines = [
        "D. How every number was calculated (no AI model)",
        "",
        f"Photo: Vexcel {col['collection']}, estimated capture {est[:10]}",
        f"Place (Vexcel parcel centre): {site_lat:.4f}, {site_lon:.4f}",
        f"Photo time: {sun_now['time_utc']} UTC ({time_source})",
        f"Sun from lat/lon + time (NOAA): direction {sun_now['sun_azimuth_deg']:.0f}°, "
        f"{sun_now['sun_elevation_deg']:.1f}° above horizon",
        f"   check: math says shadows point {sun_now['math_shadow_azimuth_deg']:.0f}°, "
        f"photo shadows point {shadow_az:.0f}°",
        f"Sun path here: Jun noon {path['summer_solstice']['noon_elevation_deg']}° up, "
        f"Dec noon {path['winter_solstice']['noon_elevation_deg']}° up",
        "Wall height = shadow length x tan(sun height)",
        (f"   walls {min(eaves):.1f}-{max(eaves):.1f} m -> "
         + ", ".join(f"{n} building(s) {fl} floor{'s' if fl > 1 else ''}" for fl, n in sorted(
             {fl: sum(1 for e in eaves if max(1, round(e / FLOOR_TO_FLOOR_M)) == fl)
              for fl in {max(1, round(e / FLOOR_TO_FLOOR_M)) for e in eaves}}.items()))
         + " (3 m per floor)") if eaves else "   no wall height measured",
        f"   measured on {len(eaves)} of {len(buildings)} buildings ('?' = shadow blocked by cars/trees)",
        "Roof slope: sunny side vs shady side brightness (+ gable shadow tips)",
        f"   complex slope {complex_slope}° (middle half of readings {slope_range[0]}-{slope_range[1]}°, "
        f"{len(readings)} roofs)" if complex_slope and slope_range else "   slope not measured",
        "",
        f"Whole year at this lat/lon: sun position every 10 min, 1 day per month ({len(steps)} positions)",
        f"   x NASA monthly sunlight & clouds (cloud cover {clim.get('cloud_cover_percent', '?')}%, "
        f"clouds remove {clim.get('cloud_loss_percent', '?')}%)",
        f"   + shade from taller neighbours (measured heights; {totals['shading_heights_assumed']} assumed)",
        f"   best possible: {totals['best_sun_kwh_m2_year']:.0f} kWh/m² a year (tilt {totals['best_tilt_deg']}°, facing south)",
        f"   cross-check: our model / NREL PVWatts = {totals['our_model_vs_pvwatts']}",
        "",
        f"Panels: 400 W, 1.13 x 1.72 m, 3 ft fire gap from edges/ridge, only where sun >= {KEEP_PERCENT}% of best",
        f"   {totals['chosen_sides']} of {totals['roof_sides']} roof sides -> {totals['panels']:,} panels, "
        f"{totals['panel_area_m2']:,} m² of panels",
        f"   {totals['capacity_kw']} kW, {totals['annual_kwh']:,} kWh per year",
        f"   (panels on every side: {totals['all_sides_panels']:,} panels, {totals['all_sides_annual_kwh']:,} kWh)",
        f"Old 'every roof is flat' guess: {totals['flat_assumption_kw']} kW, "
        f"{(totals['flat_assumption_annual_kwh'] or 0):,} kWh per year",
        "",
        "Limits: trees not in shade model; one photo; slope is one value for the complex.",
    ]
    need = y2 + 36 * len(lines) + 40
    if need > canvas.height:  # grow the canvas so no text is cut off
        bigger = Image.new("RGB", (canvas.width, need), (250, 250, 248))
        bigger.paste(canvas, (0, 0))
        canvas, cd = bigger, ImageDraw.Draw(bigger)
    yy = y2
    for i, line in enumerate(lines):
        cd.text((W + 40, yy), line, font=font(26 if i == 0 else 22), fill=(25, 25, 25) if i == 0 else (60, 60, 60))
        yy += 36

    report = out_dir / f"property_{b.id}_report.jpg"
    canvas.save(report, quality=88)
    result = {"property_id": b.id, "capture": {"collection": col["collection"], "date_estimate": est[:10], **sun_now},
              "dark_threshold": round(threshold, 4), "buildings": buildings,
              "roof_sides": [{k: v for k, v in sd.items() if k not in ("poly", "panels")} for sd in sides_out],
              "totals": totals, "climate": {k: clim.get(k) for k in ("cloud_cover_percent", "cloud_loss_percent")},
              "report_image": str(report)}
    (out_dir / f"property_{b.id}.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({"sun": sun_now, "median_wall_m": statistics.median(eaves) if eaves else None,
                      "median_slope_deg": statistics.median(slopes) if slopes else None, **totals}, indent=1))
    print(f"saved: {report}")
    return result


if __name__ == "__main__":
    main()
