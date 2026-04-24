# Geospatial Feature Notes

- `lat_bin` and `long_bin` provide coarse deterministic spatial grouping without fitting a global learned map.
- `geo_cell` captures a higher-resolution latitude-longitude grid and remains leakage-safe because it is derived only from observed coordinates.
- `distance_to_seattle_core` and `distance_to_bellevue_core` summarize proximity to major employment and price centers using fixed anchors.
- `grade_living_interaction` and `location_grade_interaction` expose simple structural-location interactions without introducing price-derived information.
- The enhanced feature branch is optional and separate from the frozen baseline branch so official results remain reproducible.