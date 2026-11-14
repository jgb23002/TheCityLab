# Rail stations for the Empty Lot Planner transit check

`rail_stations.csv`: one row per station (city, name, lat, lon, source).

- Los Angeles: parent stations (`location_type == 1`) from LA Metro's rail GTFS,
  https://gitlab.com/LACMTA/gtfs_rail (downloaded Nov 14, 2026).
- Houston: METRORail and other rail stations from OpenStreetMap via Overpass
  (© OpenStreetMap contributors, ODbL), freight (UP) points removed.

Only rail stations are included. AB 2097's "major transit stop" also covers some
bus intersections with frequent service, so a building far from rail may still
qualify; the planner reports "likely eligible", never a legal answer.
