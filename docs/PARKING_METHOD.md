# Counting parking spaces from aerial photos

**Result:** combining car positions across 4 Vexcel aerial captures counted
**190 spaces where a hand count found 201 (95%)** on one Houston apartment
complex (CoStar property 1189335).

Owner: Dev (data). Code: `backend/vexcel_detect.py`. Status: working
prototype, tested on 1 property, 9 of 18 grid squares hand-counted.

## Why we needed this

The accuracy check compares parking measured from photos with CoStar's
recorded parking (`parking[].totalSpaces`). Vexcel's precomputed attributes do
not include a parking-space count: they give `vehicles.count` (cars visible on
one date) and `hardscapes.percent` (all paving). So spaces have to be counted
from the imagery.

## What we tried

All tests used property 1189335 (431 units, 13.0 acres; CoStar: 288 surface +
99 covered spaces).

| # | Method | Result | Why it fell short |
|---|---|---|---|
| 1 | `sam3` "parking space" on an 80 m square | 4 | Boxes whole lots, not spaces |
| 2 | `sam3` "car" on one large image | 84 | Small objects missed on a big image; only cars present that day |
| 3 | Same, split into ~80 m tiles | 128 cars + 26 empty | Better recall; empty spaces still mostly missed |
| 4 | "parking lot" area (masks) / 300–350 sq ft per space | 92–108 | Apartment parking is strips along drives, not "lots" |
| 5 | **Cars across 4 capture dates, merged by location** | **190 vs 201 hand count** | **Chosen** |

## The method (#5)

1. **Find the property outline.** CoStar's point can sit off-centre or in a
   small neighbouring parcel. `property_outline()` searches Vexcel parcels
   around the point and picks the one whose area is closest to CoStar
   `lotSize` (within 25%). For 1189335: 13.11 acres vs CoStar 13.02.
2. **Pick dates.** `ortho/collections` lists every capture of the site (16 for
   1189335, back to 2017). We use the 4 newest sharp (7.5 cm) captures:
   Jan 2026, Oct 2025, Mar 2025, Sep 2024.
3. **Detect cars in ~80 m tiles** for each date (`elements/detect`, model
   `sam3`, label "car"). Tiles are needed because the detector misses small
   objects on one large image.
4. **Keep spots inside the outline.** This drops traffic on public roads and
   neighbours' cars.
5. **Merge across dates.** A space that holds a car on any date is a real
   space. Two detections are the same space if they overlap, or sit within
   half a space width sideways (1.3 m) and under one space length along the
   car (3.5 m). These limits come from a standard 9 × 18 ft space; captures
   shift 1–3 m between dates, and shade can split one car into two boxes.
6. **Add empty spaces** that no car covered, from "empty parking spot" on the
   newest capture.

## Validation

Dev hand-counted spaces (with and without cars) square by square on a 3 × 6
grid over the property photo. Sky count vs hand count:

| Square | Hand | Sky | Off by |
|---|---|---|---|
| B2 | 21 | 16 | −5 |
| C2 | 0 | 0 | 0 |
| C3 | 41 | 40 | −1 |
| D1 | 22 | 18 | −4 |
| D2 | 22 | 27 | +5 |
| D3 | 22 | 25 | +3 |
| E1 | 19 | 16 | −3 |
| E3 | 24 | 23 | −1 |
| F1 | 30 | 25 | −5 |
| **Total** | **201** | **190** | **−11 (95%)** |

How to read it: 95% is the **total**. Single squares range from 76% to 123%
because over- and under-counts partly cancel; the average per-square error is
about 13%.

Remaining errors, seen in the images:

- **Under-counts:** cars half-hidden in deep building shade; spaces never
  occupied on any of the 4 dates.
- **Over-counts:** a dumpster, a car stopped in a drive lane, a box on a
  sidewalk, boxes on accessible-space hatching.
- **Not visible at all:** spaces under carport roofs.

## Limits

- **One property, 9 of 18 squares.** Promising, not proven. The full-property
  comparison with CoStar (288 + 99) needs the remaining squares counted.
- **Garden-style complexes only.** Property 1146902 keeps parking in courtyard
  garages; the sky method found 46–102 vs CoStar's 195 surface spaces. Nothing
  stacked under a roof can be seen from above. Such buildings should be
  flagged, not measured.
- **Unmarked curb parking** (cars along a curb with no painted lines) is
  counted by the method. Whether it should count depends on the comparison.
- **Cost:** about 50–70 Vexcel calls per property (tiles × 4 dates, plus empty
  spots). Fine for a handful of validation properties, not for every building.
  Every call is cached, so re-runs and the demo use no network.

## How to run

```python
from backend import vexcel_detect as vd

outline = vd.property_outline(lat, lon, lot_acres)      # CoStar lotSize, acres
result = vd.spaces_from_dates(lat, lon, property_id, n_dates=4, outline_wkt=outline)

result.spaces                  # distinct spots (the number validated above)
result.dates                   # cars found per capture date
result.annotated_image_path    # newest photo with every spot boxed
```

The hand-count sheet and grid image are in `data/cache/handcount/`
(git-ignored).

## Pitch line

> "We count parking from the sky by combining car positions across 4 aerial
> photo dates. Against a hand count of 201 spaces, it was 95% accurate."

Say alongside it: tested on one complex; garage buildings are flagged rather
than measured.

## Future work

1. **Train our own parking-space model.** Use the hand-counted squares as
   labels and fine-tune a detector (for example YOLO) on Vexcel imagery to box
   individual painted spaces, empty or occupied. Needs: a few hundred
   labelled tiles, permission from Vexcel to train on their imagery, and about
   a day of labelling plus training. Expected gain: count empty spaces
   directly instead of waiting for a car to park there.
2. **Validate on more properties.** Hand-count 5–10 garden-style complexes and
   report the error per property, not just one total.
3. **Count carport spaces.** Detect carport roofs and divide their length by
   a 9 ft space width, or use the oblique (angled) imagery Vexcel offers,
   which can see under roof edges. Compare with CoStar's covered count
   separately.
4. **Line detection.** Count painted stripes with classic image processing
   (OpenCV line detection) on the cached photos, and merge with the car spots.
   It is free and catches spaces that were never occupied.
5. **Align captures before merging.** Register each date's image to the newest
   one so the 1–3 m shifts disappear, instead of absorbing them with merge
   limits.
6. **Classify parking type automatically.** Flag garage and podium buildings
   (courtyard decks, few visible cars per unit) so they are excluded instead
   of under-counted.
7. **More dates.** Sixteen captures exist for 1189335; adding more raises the
   chance that every space was occupied at least once.
8. **Claude as a reviewer.** Have Claude check doubtful boxes (dumpsters,
   lanes, hatching) on cropped tiles before they are counted.
