import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRID_DATA_DIR = ROOT / "Data_Set" / "Dataset 5 – Grid Congestion & Constraints"
OTHER_DATA_DIR = ROOT / "other data"

CONGESTION_PC6_FILE = GRID_DATA_DIR / "congestie_pc6.csv"
FEEDING_AREAS_FILE = GRID_DATA_DIR / "voedingsgebieden.csv"
PROJECTS_FILE = GRID_DATA_DIR / "projecten.csv"
CHARGING_POINTS_FILE = OTHER_DATA_DIR / "charging_points_strijp_s.csv"

POSTCODE_PREFIX = "5617"
POWER_PER_CONNECTOR_KW = 22
CONNECTORS_PER_NEW_CHARGER = 2
NEW_CHARGER_SCENARIOS = [0, 5, 10, 20, 30, 50]


def parse_decimal(value):
    if value is None or value == "":
        return None
    return float(str(value).replace(",", "."))


def read_semicolon_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file, delimiter=";"))


def read_comma_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def find_feeding_area_for_postcode(postcode_prefix):
    rows = read_semicolon_csv(CONGESTION_PC6_FILE)
    postcode_rows = [
        row for row in rows
        if row["postcode"].startswith(postcode_prefix)
    ]

    if not postcode_rows:
        raise ValueError(f"No postcode rows found for postcode prefix {postcode_prefix}")

    area_counts = {}
    for row in postcode_rows:
        area_id = row["voedingsgebied_id"]
        area_counts[area_id] = area_counts.get(area_id, 0) + 1

    feeding_area_id = max(area_counts, key=area_counts.get)
    feeding_area_row = next(
        row for row in postcode_rows
        if row["voedingsgebied_id"] == feeding_area_id
    )

    return {
        "postcode_prefix": postcode_prefix,
        "postcode_rows": len(postcode_rows),
        "feeding_area_id": feeding_area_id,
        "feeding_area_name": feeding_area_row["voedingsgebied_naam"],
        "tennet_station": feeding_area_row["tennet_id"],
        "grid_operator": feeding_area_row["RNB_postcode"],
    }


def load_feeding_area_capacity(feeding_area_id):
    rows = read_semicolon_csv(FEEDING_AREAS_FILE)
    row = next(
        item for item in rows
        if item["voedingsgebied_id"] == feeding_area_id
    )

    available_capacity_mw = parse_decimal(row["aanwezige_transportcapaciteit_afname"])
    needed_capacity_mw = parse_decimal(row["benodigde_transportcapaciteit_afname"])
    remaining_capacity_mw = available_capacity_mw - needed_capacity_mw

    return {
        "year": row["jaar"],
        "available_capacity_mw": available_capacity_mw,
        "needed_capacity_mw": needed_capacity_mw,
        "remaining_capacity_mw": remaining_capacity_mw,
        "remaining_capacity_kw": remaining_capacity_mw * 1000,
        "existing_grid_utilization_percent": (
            needed_capacity_mw / available_capacity_mw * 100
        ),
    }


def load_projects_for_feeding_area(feeding_area_id):
    rows = read_semicolon_csv(PROJECTS_FILE)
    return [
        row for row in rows
        if row["gebied_id"] == feeding_area_id
    ]


def count_existing_charger_connectors():
    rows = read_comma_csv(CHARGING_POINTS_FILE)
    connector_count = sum(parse_decimal(row["connectors"]) or 0 for row in rows)

    return {
        "charging_points": len(rows),
        "connectors": connector_count,
    }


def calculate_projected_ev_demand_kw(existing_connectors, new_chargers):
    new_connectors = new_chargers * CONNECTORS_PER_NEW_CHARGER
    total_connectors = existing_connectors + new_connectors
    return total_connectors * POWER_PER_CONNECTOR_KW


def calculate_gfur(projected_ev_demand_kw, remaining_capacity_kw):
    return projected_ev_demand_kw / remaining_capacity_kw


def calculate_capacity_cover_ratio(remaining_capacity_kw, projected_ev_demand_kw):
    return remaining_capacity_kw / projected_ev_demand_kw


def classify_feasibility(gfur):
    if gfur <= 1:
        return "feasible under remaining feeding-area capacity"
    return "not feasible without extra grid capacity"


def calculate_scenario(new_chargers, existing_connectors, remaining_capacity_kw):
    projected_ev_demand_kw = calculate_projected_ev_demand_kw(
        existing_connectors,
        new_chargers,
    )
    gfur = calculate_gfur(projected_ev_demand_kw, remaining_capacity_kw)
    capacity_cover = calculate_capacity_cover_ratio(
        remaining_capacity_kw,
        projected_ev_demand_kw,
    )

    return {
        "new_chargers": new_chargers,
        "new_connectors": new_chargers * CONNECTORS_PER_NEW_CHARGER,
        "total_connectors": existing_connectors
        + new_chargers * CONNECTORS_PER_NEW_CHARGER,
        "projected_ev_demand_kw": round(projected_ev_demand_kw, 1),
        "remaining_grid_capacity_kw": round(remaining_capacity_kw, 1),
        "gfur_ratio": round(gfur, 4),
        "gfur_percent": round(gfur * 100, 2),
        "capacity_cover_ratio": round(capacity_cover, 2),
        "interpretation": classify_feasibility(gfur),
    }


def calculate_gfur_kpi():
    feeding_area = find_feeding_area_for_postcode(POSTCODE_PREFIX)
    capacity = load_feeding_area_capacity(feeding_area["feeding_area_id"])
    projects = load_projects_for_feeding_area(feeding_area["feeding_area_id"])
    charger_inventory = count_existing_charger_connectors()

    scenarios = [
        calculate_scenario(
            new_chargers,
            charger_inventory["connectors"],
            capacity["remaining_capacity_kw"],
        )
        for new_chargers in NEW_CHARGER_SCENARIOS
    ]

    summary = {
        **feeding_area,
        **capacity,
        "existing_public_chargers_inside_strijp_s": charger_inventory["charging_points"],
        "existing_public_charger_connectors": charger_inventory["connectors"],
        "assumed_power_per_connector_kw": POWER_PER_CONNECTOR_KW,
        "connectors_per_new_charger": CONNECTORS_PER_NEW_CHARGER,
        "projects_for_feeding_area": [
            f"{project['projectnaam']} ({project['jaar']})"
            for project in projects
        ],
    }

    return scenarios, summary


def print_gfur_results(scenarios, summary):
    print("Grid Feasibility Capacity Utilization Ratio (GFUR)")
    print("-" * 72)
    print(f"Postcode prefix: {summary['postcode_prefix']}")
    print(f"Feeding area: {summary['feeding_area_id']} - {summary['feeding_area_name']}")
    print(f"TenneT station: {summary['tennet_station']}")
    print(f"Grid operator: {summary['grid_operator']}")
    print(f"Year: {summary['year']}")
    print()
    print(f"Available capacity: {summary['available_capacity_mw']:.2f} MW")
    print(f"Needed capacity: {summary['needed_capacity_mw']:.2f} MW")
    print(f"Remaining capacity: {summary['remaining_capacity_kw']:.1f} kW")
    print(
        "Existing grid utilization before extra EV demand: "
        f"{summary['existing_grid_utilization_percent']:.2f}%"
    )
    print()
    print(
        "Existing public chargers inside Strijp S: "
        f"{summary['existing_public_chargers_inside_strijp_s']}"
    )
    print(
        "Existing public charger connectors: "
        f"{summary['existing_public_charger_connectors']:.0f}"
    )
    print(f"Assumed power per connector: {summary['assumed_power_per_connector_kw']} kW")
    print(f"Connectors per new charger: {summary['connectors_per_new_charger']}")
    print()

    print("Scenario results")
    print("-" * 110)
    print(
        f"{'New chargers':>12} {'Total connectors':>17} "
        f"{'EV demand (kW)':>16} {'GFUR (%)':>10} "
        f"{'Capacity cover':>16} {'Interpretation':>34}"
    )
    print("-" * 110)

    for row in scenarios:
        print(
            f"{row['new_chargers']:12.0f} "
            f"{row['total_connectors']:17.0f} "
            f"{row['projected_ev_demand_kw']:16.1f} "
            f"{row['gfur_percent']:10.2f} "
            f"{row['capacity_cover_ratio']:16.2f} "
            f"{row['interpretation']:>34}"
        )


if __name__ == "__main__":
    gfur_scenarios, gfur_summary = calculate_gfur_kpi()
    print_gfur_results(gfur_scenarios, gfur_summary)
