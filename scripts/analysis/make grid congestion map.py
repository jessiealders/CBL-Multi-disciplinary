import csv
import json
from pathlib import Path
from collections import Counter


# Purpose
# This script makes the Eindhoven congestion map.
# It does not calculate congestion. The calculation is done first in:
# "Calculate grid congestion by neighbourhood.py".
#
# Data used and why
# 1. eindhoven_grid_congestion_by_neighbourhood.csv
#    This has the calculated risk class and congestion summary for each
#    neighbourhood.
#
# 2. eindhoven_buurten_cbs_2024.geojson
#    This has the official CBS neighbourhood shapes. We need these shapes to
#    draw the map.
#
# Map approach
# 1. Read the calculated CSV.
# 2. Read the neighbourhood shapes.
# 3. Match each shape to one CSV row by buurtcode.
# 4. Add a colour based on risk_class:
#    red = high, orange = medium, yellow = congested grid area,
#    green = low, grey = unknown.
# 5. Save an interactive HTML map.
#
# Important note:
# This script does not calculate congestion. It only makes the map.
# The map is a postcode-based neighbourhood approximation, not an exact PC6 or
# transformer map.

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[2]
OTHER_DATA_DIR = ROOT / "processed data"
SPATIAL_DATA_DIR = OTHER_DATA_DIR / "spatial"

BUURTEN_FILE = SPATIAL_DATA_DIR / "eindhoven_buurten_cbs_2024.geojson"
INPUT_CSV = TASK_DIR / "processed data" / "grid" / "eindhoven_grid_congestion_by_neighbourhood.csv"
OUTPUT_MAP = OTHER_DATA_DIR / "eindhoven_grid_congestion_map.html"


def read_csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def full_buurtcode(short_code):
    if str(short_code).startswith("BU"):
        return str(short_code)
    return f"BU0772{int(short_code):03d}0"


def color_for_risk(risk):
    return {
        "high": "#b71c1c",
        "medium": "#ef6c00",
        "congested grid area": "#fdd835",
        "low": "#43a047",
        "unknown": "#bdbdbd",
    }.get(risk, "#bdbdbd")


def build_map_geojson():
    rows = read_csv_rows(INPUT_CSV)
    row_by_buurtcode = {row["buurtcode"]: row for row in rows}
    buurten = json.loads(BUURTEN_FILE.read_text(encoding="utf-8"))

    for feature in buurten["features"]:
        original_props = feature["properties"]
        buurtcode = full_buurtcode(original_props["buurtcode"])
        row = row_by_buurtcode.get(buurtcode)

        if row is None:
            row = {
                "buurtcode": buurtcode,
                "buurtnaam": original_props["buurtnaam"],
                "wijknaam": original_props.get("wijknaam") or original_props.get("wijkcode", ""),
                "postcode4": "",
                "pc6_count": "0",
                "max_afname_code": "",
                "avg_afname_code": "",
                "share_afname_code_gt0": "",
                "voedingsgebied_id": "",
                "main_feeding_area": "",
                "other_feeding_areas": "",
                "remaining_afname_mw": "",
                "utilization_afname_percent": "",
                "queue_afname_mw": "",
                "risk_class": "unknown",
            }

        feature["properties"] = {
            **row,
            "fill_color": color_for_risk(row["risk_class"]),
        }

    return buurten, rows


def write_map_html(buurten, rows):
    counts = Counter(row["risk_class"] for row in rows)
    geojson_text = json.dumps(buurten, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Eindhoven Risk Classification Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Arial, sans-serif; }}
    #map {{ height: 100%; width: 100%; }}
    .panel {{
      position: absolute; z-index: 1000; top: 14px; left: 14px; max-width: 470px;
      background: white; border: 1px solid #cfd8dc; border-radius: 8px;
      padding: 12px 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.16);
      font-size: 14px; line-height: 1.35;
    }}
    .panel h1 {{ font-size: 17px; margin: 0 0 8px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin-top: 6px; }}
    .swatch {{ width: 16px; height: 12px; border: 1px solid #777; display: inline-block; }}
    .note {{ color: #455a64; margin-top: 8px; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Eindhoven Risk Classification Map</h1>
    <div>Neighbourhoods mapped: <strong>{len(rows)}</strong></div>
    <div>
      High: <strong>{counts.get("high", 0)}</strong>,
      Medium: <strong>{counts.get("medium", 0)}</strong>,
      Congested grid area: <strong>{counts.get("congested grid area", 0)}</strong>,
      Low: <strong>{counts.get("low", 0)}</strong>,
      Unknown: <strong>{counts.get("unknown", 0)}</strong>
    </div>
    <div class="legend-row"><span class="swatch" style="background:#b71c1c"></span> High: average afname code is 2 or higher</div>
    <div class="legend-row"><span class="swatch" style="background:#ef6c00"></span> Medium: average afname code is 1 or higher</div>
    <div class="legend-row"><span class="swatch" style="background:#fdd835"></span> Congested grid area: average afname is below 1, but queue exists or grid use is 90%+</div>
    <div class="legend-row"><span class="swatch" style="background:#43a047"></span> Low: no clear congestion signal</div>
    <div class="legend-row"><span class="swatch" style="background:#bdbdbd"></span> Unknown: no postcode match</div>
    <div class="note">
      Approach: the calculation links each neighbourhood to its most common
      4-digit postcode. Then all PC6 rows inside that postcode group are
      summarized. Feeding-area capacity and queue data are added as extra
      context. This is a postcode-based neighbourhood approximation. It is
      useful for an Eindhoven overview, but it is not a precise PC6 or
      transformer map.
    </div>
  </div>
  <div id="map"></div>
  <script>
    const neighbourhoods = {geojson_text};
    const map = L.map("map").setView([51.4416, 5.4697], 12);

    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    function style(feature) {{
      return {{
        color: "#263238",
        weight: 0.7,
        fillColor: feature.properties.fill_color,
        fillOpacity: 0.68
      }};
    }}

    function valueOrNa(value) {{
      return value === undefined || value === null || value === "" ? "n/a" : value;
    }}

    function popup(feature, layer) {{
      const p = feature.properties;
      layer.bindPopup(`
        <strong>${{p.buurtnaam}}</strong><br>
        ${{p.wijknaam}}<br>
        Postcode-4: ${{valueOrNa(p.postcode4)}}<br>
        Risk class: <strong>${{p.risk_class}}</strong><br>
        PC6 rows used: ${{valueOrNa(p.pc6_count)}}<br>
        Max afname code: ${{valueOrNa(p.max_afname_code)}}<br>
        Avg afname code: ${{valueOrNa(p.avg_afname_code)}}<br>
        Share afname code > 0: ${{valueOrNa(p.share_afname_code_gt0)}}%<br>
        Main feeding area: ${{valueOrNa(p.main_feeding_area || p.voedingsgebied_id)}}<br>
        Other feeding areas: ${{valueOrNa(p.other_feeding_areas)}}<br>
        Remaining capacity: ${{valueOrNa(p.remaining_afname_mw)}} MW<br>
        Grid use: ${{valueOrNa(p.utilization_afname_percent)}}%<br>
        Queue: ${{valueOrNa(p.queue_afname_mw)}} MW
      `);
    }}

    const layer = L.geoJSON(neighbourhoods, {{ style, onEachFeature: popup }}).addTo(map);
    map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});
  </script>
</body>
</html>
"""
    OUTPUT_MAP.write_text(html, encoding="utf-8")


def main():
    buurten, rows = build_map_geojson()
    write_map_html(buurten, rows)
    print(f"Saved map: {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
