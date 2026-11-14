# Sun position and solar climate: Houston vs Los Angeles

Owner: Dev (data). Code: `backend/sun.py`, `scripts/solar_conditions.py`.
No model is used: sun position is math, weather is public NASA data.

## Result (median of 50 sampled buildings per city)

| | Houston | Los Angeles |
|---|---|---|
| Latitude | 29.76 | 34.06 |
| Sunlight reaching the ground, kWh/m²/day | **4.65** | **5.65** |
| Sunlight on a cloudless day, kWh/m²/day | 6.03 | 6.28 |
| Cloud cover | **52%** | **30%** |
| Sunlight lost to clouds | **23%** | **10%** |
| Average temperature | 20.5 °C | 17.4 °C |
| Noon sun height, Jun 21 / Dec 21 | 83.7° / 36.8° | 79.4° / 32.5° |
| Day length, Jun 21 / Dec 21 | 14.1 h / 10.2 h | 14.4 h / 9.9 h |
| Best fixed panel tilt (facing south) | 25.7° | 29.0° |

**Finding:** Houston's sun is higher (it is further south), yet Los Angeles
gets about 22% more sunlight on the ground, because Houston loses 23% of
its sunlight to clouds versus 10% in LA. Clouds matter more than latitude.

## How it is computed

- **Sun position** (task 2): NOAA solar position equations from latitude,
  longitude and time: declination, equation of time, hour angle. They give
  the sun's height and compass direction at any moment, noon height,
  day length, and sunrise/sunset direction. Checked against the textbook
  rule (noon height = 90 − |latitude − declination|) in `tests/test_sun.py`.
- **Weather** (task 3): NASA POWER climatology (power.larc.nasa.gov, free,
  no key): long-term monthly averages of all-sky sunlight
  (`ALLSKY_SFC_SW_DWN`), clear-sky sunlight (`CLRSKY_SFC_SW_DWN`), cloud
  amount (`CLOUD_AMT`), clearness index and temperature.
  Sunlight lost to clouds = 1 − all-sky / clear-sky.
- **Best tilt:** 0.76 × latitude + 3.1° (Jacobson & Jadhav 2018), facing
  due south.

## Limits

- NASA POWER cells are about 50 km wide, so all buildings in one city share
  the same weather. This compares cities, not individual roofs.
- No shading from neighbouring buildings or trees.
- Long-term averages, not a specific year.

## How to run

```bash
python scripts/solar_conditions.py      # writes data/cache/solar_conditions/
```

```python
from backend import sun
sun.sun_summary(lat, lon)        # sun geometry
sun.climate(lat, lon)            # sunlight, clouds, temperature (cached)
sun.sun_position(lat, lon, when) # height + direction at a timezone-aware moment
```
