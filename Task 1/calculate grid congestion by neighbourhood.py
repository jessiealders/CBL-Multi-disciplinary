import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


# Purpose
# This script calculates a postcode-based neighbourhood approximation of grid
# congestion risk for Eindhoven.
# It saves a CSV table. The map is made in a separate file:
# "Make grid congestion map.py".
#
# Data used and why
# 1. eindhoven_buurten_cbs_2024.geojson
#    This gives the official Eindhoven neighbourhood shapes and names.
#    It also already has the most common postcode for each neighbourhood.
#    The congestion data has no map shapes, so we need this file for the
#    neighbourhood layer.
#
# 2. lili_populationdesnity_districts.csv
#    This file is no longer needed for the main postcode join, because the CBS
#    GeoJSON already has "meest_voorkomende_postcode". It can still be useful
#    as a backup population/statistics file.
#
# 3. congestie_pc6.csv
#    This is the main congestion dataset. It has PC6 postcode rows, for example
#    5617AA, 5617AB, 5617AC. The values "afname" and "opwek" show congestion
#    codes for electricity demand and generation.
#
# 4. voedingsgebieden.csv
#    This adds grid-area context, such as available capacity, needed capacity,
#    remaining capacity, grid utilization, and queue size.
#
# 5. projecten.csv
#    This adds planned grid projects for each feeding area.
#
# Approach
# 1. Read the neighbourhoods.
# 2. Find the most common 4-digit postcode for each neighbourhood.
# 3. Group all PC6 rows by their first 4 postcode numbers.
#    Example: 5617AA, 5617AB, and 5617AC are grouped as 5617.
# 4. Join each neighbourhood to the matching postcode-4 group.
# 5. Calculate summary values for that postcode group:
#    max_afname_code, avg_afname_code, share_afname_code_gt0,
#    max_opwek_code, and avg_opwek_code.
# 6. Find the main feeding area for that postcode group.
# 7. Add capacity data and planned projects for that feeding area.
# 8. Classify each neighbourhood into a simple risk class.
# 9. Save the result as a CSV file.
#
# Important note:
# This is an estimate by neighbourhood. The grid data is at PC6 postcode level,
# but the output is at neighbourhood level. So we use the most common postcode
# in each neighbourhood as the link between the two datasets.
# This is useful for an Eindhoven overview. It should be described as a
# postcode-based neighbourhood approximation, not as a precise transformer or
# exact PC6 map.
#
# Risk classification
# high:
#     avg_afname_code >= 2
# medium:
#     avg_afname_code >= 1
# congested grid area:
#     avg_afname_code < 1, but queue_afname_mw > 0 or grid utilization >= 90%
# low:
#     no local afname congestion and no strong grid-area warning
# unknown:
#     no postcode or grid match

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = Path(__file__).resolve().parent
OTHER_DATA_DIR = ROOT / "other data"
SPATIAL_DATA_DIR = OTHER_DATA_DIR / "spatial"
DATASET_DIR = ROOT / "Data_Set"


def find_dataset_5_dir():
    for path in DATASET_DIR.iterdir():
        if path.is_dir() and path.name.startswith("Dataset 5"):
            return path
    raise FileNotFoundError("Could not find Dataset 5 folder.")


GRID_DATA_DIR = find_dataset_5_dir()

BUURTEN_FILE = SPATIAL_DATA_DIR / "eindhoven_buurten_cbs_2024.geojson"
POSTCODE_BY_BUURT_FILE = OTHER_DATA_DIR / "lili_populationdesnity_districts.csv"
CONGESTION_PC6_FILE = GRID_DATA_DIR / "congestie_pc6.csv"
FEEDING_AREAS_FILE = GRID_DATA_DIR / "voedingsgebieden.csv"
PROJECTS_FILE = GRID_DATA_DIR / "projecten.csv"
OUTPUT_CSV = TASK_DIR / "eindhoven_grid_congestion_by_neighbourhood.csv"
OUTPUT_RISK_CLASSIFICATION_CSV = TASK_DIR / "eindhoven_neighbourhood_risk_classification.csv"


def parse_num(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def read_csv_rows(path, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def round_or_blank(value, digits=2):
    if value is None:
        return ""
    return round(value, digits)


def full_buurtcode(short_code):
    if str(short_code).startswith("BU"):
        return str(short_code)
    return f"BU0772{int(short_code):03d}0"


def load_postcode_by_buurt():
    rows = read_csv_rows(POSTCODE_BY_BUURT_FILE)
    postcode_by_buurt = {}

    for row in rows:
        buurtcode = row.get("buurtcode", "")
        postcode = str(row.get("meestVoorkomendePostcode", "")).strip().strip('"')

        if buurtcode and postcode and postcode != "-99997":
            postcode_by_buurt[buurtcode] = postcode[:4]

    return postcode_by_buurt


def load_feeding_lookup():
    rows = read_csv_rows(FEEDING_AREAS_FILE, delimiter=";")
    feeding_lookup = {}

    for row in rows:
        area_id = row["voedingsgebied_id"]
        available = parse_num(row["aanwezige_transportcapaciteit_afname"])
        needed = parse_num(row["benodigde_transportcapaciteit_afname"])
        queue = parse_num(row["wachtrij_afname"])
        requests = parse_num(row["unieke_verzoeken_afname"])

        if available is not None and needed is not None:
            remaining = available - needed
            utilization = needed / available * 100 if available else None
        else:
            remaining = None
            utilization = None

        feeding_lookup[area_id] = {
            "jaar": row["jaar"],
            "available_afname_mw": available,
            "needed_afname_mw": needed,
            "remaining_afname_mw": remaining,
            "utilization_afname_percent": utilization,
            "queue_afname_mw": queue,
            "requests_afname": requests,
            "resolved_year_afname": row.get("jaartal_opgelost_afname") or "",
            "rnb": row.get("RNB") or "",
        }

    return feeding_lookup


def load_projects_by_area():
    rows = read_csv_rows(PROJECTS_FILE, delimiter=";")
    projects_by_area = defaultdict(list)

    for row in rows:
        area_id = row.get("gebied_id")
        if area_id:
            projects_by_area[area_id].append(
                f"{row.get('projectnaam', '')} ({row.get('jaar', '')})"
            )

    return projects_by_area


def load_pc4_lookup():
    rows = read_csv_rows(CONGESTION_PC6_FILE, delimiter=";")
    pc4_lookup = defaultdict(list)

    for row in rows:
        postcode = row["postcode"]
        if len(postcode) >= 4:
            pc4_lookup[postcode[:4]].append(row)

    return pc4_lookup


def summarize_pc4(pc4, pc4_lookup):
    rows = pc4_lookup.get(pc4, [])

    if not rows:
        return None

    afname_values = [
        parse_num(row["afname"])
        for row in rows
        if parse_num(row["afname"]) is not None
    ]
    opwek_values = [
        parse_num(row["opwek"])
        for row in rows
        if parse_num(row["opwek"]) is not None
    ]

    area_counts = Counter(
        row["voedingsgebied_id"]
        for row in rows
        if row.get("voedingsgebied_id")
    )
    main_area = area_counts.most_common(1)[0][0] if area_counts else ""
    other_areas = [
        f"{area_id} ({count})"
        for area_id, count in area_counts.most_common()
        if area_id != main_area
    ]

    main_area_name = next(
        (
            row["voedingsgebied_naam"]
            for row in rows
            if row.get("voedingsgebied_id") == main_area
        ),
        "",
    )
    tennet_station = next(
        (
            row["tennet_id"]
            for row in rows
            if row.get("voedingsgebied_id") == main_area
        ),
        "",
    )
    grid_operator = next(
        (
            row["RNB_postcode"]
            for row in rows
            if row.get("voedingsgebied_id") == main_area
        ),
        "",
    )

    return {
        "pc6_count": len(rows),
        "max_afname_code": max(afname_values) if afname_values else None,
        "avg_afname_code": sum(afname_values) / len(afname_values)
        if afname_values
        else None,
        "share_afname_code_gt0": (
            sum(value > 0 for value in afname_values) / len(afname_values) * 100
            if afname_values
            else None
        ),
        "max_opwek_code": max(opwek_values) if opwek_values else None,
        "avg_opwek_code": sum(opwek_values) / len(opwek_values)
        if opwek_values
        else None,
        "voedingsgebied_id": main_area,
        "main_feeding_area": main_area,
        "other_feeding_areas": "; ".join(other_areas),
        "voedingsgebied_naam": main_area_name,
        "tennet_station": tennet_station,
        "grid_operator": grid_operator,
    }


def classify_risk(avg_afname_code, utilization_percent, queue_afname_mw):
    if avg_afname_code is None:
        return "unknown"
    if avg_afname_code >= 2:
        return "high"
    if avg_afname_code >= 1:
        return "medium"
    if queue_afname_mw is not None and queue_afname_mw > 0:
        return "congested grid area"
    if utilization_percent is not None and utilization_percent >= 90:
        return "congested grid area"
    return "low"


def build_output_rows():
    feeding_lookup = load_feeding_lookup()
    projects_by_area = load_projects_by_area()
    pc4_lookup = load_pc4_lookup()
    buurten = json.loads(BUURTEN_FILE.read_text(encoding="utf-8"))
    output_rows = []

    for feature in buurten["features"]:
        props = feature["properties"]
        buurtcode = full_buurtcode(props["buurtcode"])
        buurtnaam = props["buurtnaam"]
        wijknaam = props.get("wijknaam") or props.get("wijkcode", "")
        postcode4 = str(props.get("meest_voorkomende_postcode") or "").strip()
        pc4_summary = summarize_pc4(postcode4, pc4_lookup)

        if pc4_summary:
            area_id = pc4_summary["voedingsgebied_id"]
            capacity = feeding_lookup.get(area_id, {})
            utilization = capacity.get("utilization_afname_percent")
            queue = capacity.get("queue_afname_mw")
            risk = classify_risk(pc4_summary["avg_afname_code"], utilization, queue)

            output_row = {
                "buurtcode": buurtcode,
                "buurtnaam": buurtnaam,
                "wijknaam": wijknaam,
                "postcode4": postcode4,
                "pc6_count": pc4_summary["pc6_count"],
                "max_afname_code": round_or_blank(pc4_summary["max_afname_code"]),
                "avg_afname_code": round_or_blank(pc4_summary["avg_afname_code"]),
                "share_afname_code_gt0": round_or_blank(
                    pc4_summary["share_afname_code_gt0"],
                ),
                "max_opwek_code": round_or_blank(pc4_summary["max_opwek_code"]),
                "avg_opwek_code": round_or_blank(pc4_summary["avg_opwek_code"]),
                "voedingsgebied_id": area_id,
                "main_feeding_area": pc4_summary["main_feeding_area"],
                "other_feeding_areas": pc4_summary["other_feeding_areas"],
                "voedingsgebied_naam": pc4_summary["voedingsgebied_naam"],
                "tennet_station": pc4_summary["tennet_station"],
                "grid_operator": pc4_summary["grid_operator"],
                "available_afname_mw": round_or_blank(
                    capacity.get("available_afname_mw"),
                ),
                "needed_afname_mw": round_or_blank(capacity.get("needed_afname_mw")),
                "remaining_afname_mw": round_or_blank(
                    capacity.get("remaining_afname_mw"),
                ),
                "utilization_afname_percent": round_or_blank(utilization),
                "queue_afname_mw": round_or_blank(queue, 3),
                "requests_afname": round_or_blank(capacity.get("requests_afname")),
                "resolved_year_afname": capacity.get("resolved_year_afname", ""),
                "risk_class": risk,
                "projects": "; ".join(projects_by_area.get(area_id, [])),
            }
        else:
            output_row = {
                "buurtcode": buurtcode,
                "buurtnaam": buurtnaam,
                "wijknaam": wijknaam,
                "postcode4": postcode4,
                "pc6_count": 0,
                "max_afname_code": "",
                "avg_afname_code": "",
                "share_afname_code_gt0": "",
                "max_opwek_code": "",
                "avg_opwek_code": "",
                "voedingsgebied_id": "",
                "main_feeding_area": "",
                "other_feeding_areas": "",
                "voedingsgebied_naam": "",
                "tennet_station": "",
                "grid_operator": "",
                "available_afname_mw": "",
                "needed_afname_mw": "",
                "remaining_afname_mw": "",
                "utilization_afname_percent": "",
                "queue_afname_mw": "",
                "requests_afname": "",
                "resolved_year_afname": "",
                "risk_class": "unknown",
                "projects": "",
            }

        output_rows.append(output_row)

    return output_rows


def write_output_csv(output_rows):
    fieldnames = list(output_rows[0].keys())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_risk_classification_csv(output_rows):
    fieldnames = [
        "buurtcode",
        "buurtnaam",
        "wijknaam",
        "postcode4",
        "risk_class",
        "pc6_count",
        "max_afname_code",
        "avg_afname_code",
        "share_afname_code_gt0",
        "max_opwek_code",
        "avg_opwek_code",
        "voedingsgebied_id",
        "main_feeding_area",
        "other_feeding_areas",
        "voedingsgebied_naam",
        "remaining_afname_mw",
        "utilization_afname_percent",
        "queue_afname_mw",
        "projects",
    ]

    with OUTPUT_RISK_CLASSIFICATION_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main():
    output_rows = build_output_rows()
    write_output_csv(output_rows)
    write_risk_classification_csv(output_rows)

    counts = Counter(row["risk_class"] for row in output_rows)
    print("Eindhoven postcode-based neighbourhood approximation")
    print("-" * 48)
    print(f"Rows: {len(output_rows)}")
    for risk in ["high", "medium", "congested grid area", "low", "unknown"]:
        print(f"{risk}: {counts.get(risk, 0)}")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved risk classification CSV: {OUTPUT_RISK_CLASSIFICATION_CSV}")
    print("Map is made by: Make grid congestion map.py")


if __name__ == "__main__":
    main()
