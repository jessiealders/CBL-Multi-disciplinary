from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import defaultdict
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
import os
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared.walking_network import WalkingNetwork

from simulation import (
    BASE_ARRIVALS_PER_HOUR,
    BASELINE_EV_ARRIVALS,
    CANDIDATE_LOCATIONS_PATH,
    EV_DEMAND_HEATMAP_PATH,
    EXISTING_CHARGERS_PATH,
    HEATMAP_DENSITY_PATH,
    OUTPUT_DIR,
    WALKING_NETWORK_PATH,
    build_simulation_config,
    generate_arrival_times,
    load_blended_heatmap_weights,
    load_candidate_locations,
    load_existing_charger_locations,
    run_simulation,
    scale_profile_to_total,
)


USE_WALKING_NETWORK_DISTANCE = True
EV_ADOPTION_FORECAST_PATH = (
    ROOT / "processed data" / "ev adoption" / "ev_adoption_forecast.csv"
)
CHARGER_LEVELS = list(range(5, 36))
SCENARIO_YEARS = [2026, 2028, 2030, 2035]
MAX_GAVE_UP_PCT = 10.0


######################
# One-run simulation
#####################

@dataclass(frozen=True)
class RunConfig:
    scenario: str
    seed: int
    num_chargers: int
    arrivals_per_hour: list[int]
    year: int | None = None
    charger_strategy: str = "random"
    fixed_charger_fids: str | None = None
    min_charge_time: int = 45
    max_charge_time: int = 120
    walking_threshold_m: float = 300.0
    walking_speed_m_per_min: float = 83.3  # 5 km/h
    distance_mode: str = "euclidean"
    write_events: bool = False


def run_one_day(
    candidate_locations,
    destination_weights,
    existing_charger_locations,
    walking_network,
    cfg: RunConfig,
) -> dict[str, Any]:
    rng = random.Random()
    rng.seed(cfg.seed)
    arrival_times = generate_arrival_times(rng, cfg.arrivals_per_hour)

    events_dir = ROOT / "output" /"scenarios"/ "scenario_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    config = build_simulation_config(
        description=cfg.scenario,
        charger_strategy=cfg.charger_strategy,
        num_chargers=cfg.num_chargers,
        arrival_times=arrival_times,
        seed=cfg.seed,
        min_charge_time=cfg.min_charge_time,
        max_charge_time=cfg.max_charge_time,
        walking_threshold_m=cfg.walking_threshold_m,
        walking_speed_m_per_min=cfg.walking_speed_m_per_min,
        fixed_charger_fids=cfg.fixed_charger_fids,
        distance_mode=cfg.distance_mode,
        write_events=cfg.write_events,
    )
    result = run_simulation(
        f"{cfg.scenario}_seed{cfg.seed}",
        config,
        candidate_locations,
        destination_weights,
        existing_charger_locations,
        walking_network,
        events_dir,
        rng=rng,
    )

    total = len(arrival_times)

    return {
        "scenario": cfg.scenario,
        "year": cfg.year or "",
        "seed": cfg.seed,
        "distance_mode": cfg.distance_mode,
        "num_chargers": cfg.num_chargers,
        "total_arrivals": total,
        "charged": result["completed_charging"],
        "gave_up": result["gave_up"],
        "pct_gave_up": result["gave_up_pct"],
        "avg_waiting_time": result["avg_waiting_time"],
        "avg_walk_m": result["avg_walking_dist_m"],
        "util_mean": result["avg_charger_utilization"],
        "util_max": result["avg_charger_utilization"],
        "chosen_fids": result["charger_fids"],
    }


###################
# # Scenario series
###################

_shared = {}

def _init(candidates, weights, existing, network):
    _shared["candidates"] = candidates
    _shared["weights"] = weights
    _shared["existing"] = existing
    _shared["network"] = network

def _run(cfg):
    return run_one_day(
        _shared["candidates"], _shared["weights"],
        _shared["existing"], _shared["network"], cfg
    )


def selected_distance_mode() -> str:
    """Return the distance mode used by scenario simulations."""
    if USE_WALKING_NETWORK_DISTANCE:
        return "network"
    return "euclidean"


def load_walking_network_for_mode(distance_mode: str) -> WalkingNetwork | None:
    """Load the walking network only when scenario runs use network distance."""
    if distance_mode != "network":
        return None

    walking_network = WalkingNetwork.from_geojson(WALKING_NETWORK_PATH)
    print(
        f"Loaded walking network: {walking_network.trace_count} traces, "
        f"{len(walking_network.network_nodes)} connected nodes"
    )
    return walking_network


def load_ev_adoption_arrivals(path: Path) -> dict[int, int]:
    """Load yearly EV arrivals from the EV adoption forecast CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"EV adoption forecast not found: {path}. "
            "Run ev_adoption_forecast.py first."
        )

    arrivals_by_year: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "year" not in reader.fieldnames:
            raise ValueError("EV adoption forecast must contain a 'year' column.")

        ev_columns = [
            "combined_ev_cars_each_year",
            "logistic_ev_cars_each_year",
            "forecast_ev_cars",
            "ev_cars",
        ]
        ev_column = next(
            (column for column in ev_columns if column in reader.fieldnames),
            None,
        )
        if ev_column is None:
            raise ValueError(
                "EV adoption forecast is missing an EV car column. "
                f"Expected one of: {ev_columns}"
            )

        for row in reader:
            year = int(float(row["year"]))
            arrivals_by_year[year] = int(round(float(row[ev_column])))

    if not arrivals_by_year:
        raise ValueError(f"EV adoption forecast has no rows: {path}")
    return arrivals_by_year


def location_fids(locations) -> str:
    """Convert locations to the fixed charger FID string used by simulation."""
    return ";".join(str(location.fid) for location in locations)


def build_existing_plus_candidate_fids(
    existing_charger_locations,
    candidate_locations,
    total_chargers: int,
    seed: int,
) -> str:
    """Start from existing chargers and add random allowed candidate locations."""
    if total_chargers < len(existing_charger_locations):
        raise ValueError(
            "total_chargers must be at least the number of existing chargers."
        )

    additional_needed = total_chargers - len(existing_charger_locations)
    allowed_candidates = list(candidate_locations)
    if additional_needed > len(allowed_candidates):
        raise ValueError(
            f"Need {additional_needed} new chargers but only "
            f"{len(allowed_candidates)} allowed candidate locations are available."
        )

    rng = random.Random(seed)
    added_locations = rng.sample(allowed_candidates, k=additional_needed)
    return location_fids(list(existing_charger_locations) + added_locations)


def location_lookup(candidate_locations, existing_charger_locations) -> dict[int, Any]:
    """Create a lookup table for candidate and existing charger locations."""
    lookup = {location.fid: location for location in candidate_locations}
    lookup.update({location.fid: location for location in existing_charger_locations})
    return lookup


def parse_fids(raw_fids: str) -> list[int]:
    """Split the semicolon FID string from simulation output."""
    return [int(float(fid)) for fid in str(raw_fids).split(";") if fid.strip()]


def excel_column_name(index: int) -> str:
    """Return the Excel column name for a 1-based column number."""
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def safe_sheet_name(name: str) -> str:
    """Make a short sheet name that Excel can accept."""
    for char in "[]:*?/\\":
        name = name.replace(char, "_")
    return name[:31]


def worksheet_xml(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Build a simple Excel worksheet from rows and columns."""
    table_rows = [dict(zip(columns, columns))]
    table_rows.extend(rows)

    row_xml_parts = []
    for row_index, row in enumerate(table_rows, start=1):
        cell_xml_parts = []
        for col_index, column in enumerate(columns, start=1):
            value = row.get(column, "")
            cell_ref = f"{excel_column_name(col_index)}{row_index}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cell_xml_parts.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = xml_escape(str(value))
                cell_xml_parts.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        row_xml_parts.append(f'<row r="{row_index}">{"".join(cell_xml_parts)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet '
        'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(
    path: Path,
    sheets: dict[str, list[dict[str, Any]]],
    columns: list[str],
) -> None:
    """Write a small XLSX workbook with one sheet per selected scenario."""
    if not sheets:
        raise ValueError(f"No sheets to write to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.sheet.main+xml"/>',
    ]
    for sheet_index in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = "".join(
        f'<sheet name="{xml_escape(name)}" sheetId="{sheet_index}" '
        f'r:id="rId{sheet_index}"/>'
        for sheet_index, name in enumerate(sheet_names, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets>"
        "</workbook>"
    )

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for sheet_index in range(1, len(sheet_names) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{sheet_index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_index}.xml"/>'
        )
    workbook_rels.append("</Relationships>")

    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", "".join(content_types))
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for sheet_index, sheet_name in enumerate(sheet_names, start=1):
            workbook.writestr(
                f"xl/worksheets/sheet{sheet_index}.xml",
                worksheet_xml(sheets[sheet_name], columns),
            )


def best_result_for_locations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick one run for the location output, using the best passing KPI row."""
    passing_rows = [row for row in rows if kpis_pass(row)]
    choices = passing_rows or rows
    return min(
        choices,
        key=lambda row: (
            float(row["pct_gave_up"]),
            float(row["avg_waiting_time"]),
            float(row["avg_walk_m"]),
            int(row["seed"]),
        ),
    )


def write_charger_location_csv(
    output_dir: Path,
    results: list[dict[str, Any]],
    candidate_locations,
    existing_charger_locations,
    selected_scenarios: list[str],
) -> tuple[Path, Path]:
    """Write one best charger location set for each KPI-fit scenario."""
    csv_path = output_dir / "scenario_charger_locations.csv"
    xlsx_path = output_dir / "scenario_charger_locations.xlsx"
    lookup = location_lookup(candidate_locations, existing_charger_locations)
    existing_fids = {location.fid for location in existing_charger_locations}
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped_results[result["scenario"]].append(result)

    rows: list[dict[str, Any]] = []
    sheets: dict[str, list[dict[str, Any]]] = {}

    for scenario in selected_scenarios:
        scenario_rows = grouped_results.get(scenario, [])
        if not scenario_rows:
            continue
        result = best_result_for_locations(scenario_rows)
        scenario_location_rows = []
        for order, fid in enumerate(parse_fids(result["chosen_fids"]), start=1):
            location = lookup.get(fid)
            if location is None:
                continue
            row = {
                "scenario": result["scenario"],
                "year": result["year"],
                "seed": result["seed"],
                "num_chargers": result["num_chargers"],
                "charger_order": order,
                "fid": fid,
                "charger_type": (
                    "existing" if fid in existing_fids else "free_placement"
                ),
                "x_coordinate": location.x,
                "y_coordinate": location.y,
            }
            rows.append(row)
            scenario_location_rows.append(row)
        sheets[safe_sheet_name(scenario)] = scenario_location_rows

    fieldnames = [
        "scenario",
        "year",
        "seed",
        "num_chargers",
        "charger_order",
        "fid",
        "charger_type",
        "x_coordinate",
        "y_coordinate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_xlsx(xlsx_path, sheets, fieldnames)

    return csv_path, xlsx_path


def kpis_pass(row: dict[str, Any]) -> bool:
    """Check whether an averaged scenario result satisfies the KPI limits."""
    return float(row["pct_gave_up"]) <= MAX_GAVE_UP_PCT


def write_kpi_fit_summary_csv(
    output_dir: Path,
    averaged_results: list[dict[str, Any]],
    ev_adoption_arrivals: dict[int, int],
    baseline_chargers: int,
) -> tuple[Path, list[dict[str, Any]]]:
    """Find the minimum added chargers needed for selected demand cases."""
    output_path = output_dir / "scenario_kpi_fit_summary.csv"
    demand_cases = [
        {
            "demand_case": "baseline",
            "year": "",
            "input_ev_arrivals": BASELINE_EV_ARRIVALS,
            "scenario_name": "baseline",
        },
    ]
    for year in SCENARIO_YEARS:
        demand_cases.append(
            {
                "demand_case": f"ev_adoption_{year}",
                "year": year,
                "input_ev_arrivals": ev_adoption_arrivals[year],
                "scenario_prefix": f"ev_adoption_{year}_chargers_",
            }
        )

    rows: list[dict[str, Any]] = []
    averaged_columns = list(averaged_results[0].keys()) if averaged_results else []
    for case in demand_cases:
        if "scenario_name" in case:
            candidates = [
                row for row in averaged_results if row["scenario"] == case["scenario_name"]
            ]
            passing = candidates[0] if candidates else None
        else:
            candidates = [
                row
                for row in averaged_results
                if row["scenario"].startswith(case["scenario_prefix"])
            ]
            candidates.sort(key=lambda row: float(row["num_chargers"]))
            passing = next((row for row in candidates if kpis_pass(row)), None)
        row = {
            "demand_case": case["demand_case"],
            "input_ev_arrivals": case["input_ev_arrivals"],
            "minimum_total_chargers": (
                passing["num_chargers"] if passing else ""
            ),
            "added_chargers_to_baseline": (
                float(passing["num_chargers"]) - baseline_chargers
                if passing
                else ""
            ),
            "selected_scenario": passing["scenario"] if passing else "",
            "kpi_passed": bool(passing and kpis_pass(passing)),
            "kpi_limits": f"pct_gave_up <= {MAX_GAVE_UP_PCT}",
        }
        for column in averaged_columns:
            row[column] = passing[column] if passing else ""
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path, rows


def main() -> None:
    candidates = load_candidate_locations(CANDIDATE_LOCATIONS_PATH)
    existing_charger_locations = load_existing_charger_locations(EXISTING_CHARGERS_PATH)
    destination_weights = load_blended_heatmap_weights(
        candidates, HEATMAP_DENSITY_PATH, EV_DEMAND_HEATMAP_PATH
    )
    OUTPUT_DIR.mkdir(exist_ok=True)

    seeds = list(range(10))  # bump to 20/50 for better statistics
    distance_mode = selected_distance_mode()
    print(f"Using distance mode: {distance_mode}")
    walking_network = load_walking_network_for_mode(distance_mode)
    ev_adoption_arrivals_all = load_ev_adoption_arrivals(EV_ADOPTION_FORECAST_PATH)
    ev_adoption_arrivals = {
        year: arrivals
        for year, arrivals in ev_adoption_arrivals_all.items()
        if year in SCENARIO_YEARS
    }
    baseline_arrivals_per_hour = scale_profile_to_total(
        BASE_ARRIVALS_PER_HOUR,
        BASELINE_EV_ARRIVALS,
    )
    baseline_chargers = len(existing_charger_locations)

    runs: list[RunConfig] = []

    # Scenario 1: current infrastructure
    for seed in seeds:
        runs.append(
            RunConfig(
                scenario="baseline",
                seed=seed,
                num_chargers=baseline_chargers,
                arrivals_per_hour=baseline_arrivals_per_hour,
                charger_strategy="existing",
                distance_mode=distance_mode,
                write_events=False,
            )
        )

    # Scenario 2: EV adoption demand growth with current charger count.
    for year in SCENARIO_YEARS:
        ev_arrivals = ev_adoption_arrivals[year]
        for seed in seeds:
            runs.append(
                RunConfig(
                    scenario=f"ev_adoption_{year}_current_chargers",
                    year=year,
                    seed=seed,
                    num_chargers=baseline_chargers,
                    arrivals_per_hour=scale_profile_to_total(
                        BASE_ARRIVALS_PER_HOUR,
                        ev_arrivals,
                    ),
                    charger_strategy="existing",
                    distance_mode=distance_mode,
                    write_events=False,
                )
            )

    # Scenario 3: linearly more chargers
    for chargers in CHARGER_LEVELS:
        for seed in seeds:
            fixed_charger_fids = build_existing_plus_candidate_fids(
                existing_charger_locations,
                candidates,
                chargers,
                seed=seed + chargers * 1000,
            )
            runs.append(
                RunConfig(
                    scenario=f"add_chargers_{chargers}",
                    seed=seed,
                    num_chargers=chargers,
                    arrivals_per_hour=baseline_arrivals_per_hour,
                    charger_strategy="fixed",
                    fixed_charger_fids=fixed_charger_fids,
                    distance_mode=distance_mode,
                    write_events=False,
                )
            )

    # Scenario 4: EV adoption demand growth and charger levels by year.
    for year in SCENARIO_YEARS:
        ev_arrivals = ev_adoption_arrivals[year]
        for chargers in CHARGER_LEVELS:
            for seed in seeds:
                fixed_charger_fids = build_existing_plus_candidate_fids(
                    existing_charger_locations,
                    candidates,
                    chargers,
                    seed=seed + year * 1000 + chargers,
                )
                runs.append(
                    RunConfig(
                        scenario=f"ev_adoption_{year}_chargers_{chargers}",
                        year=year,
                        seed=seed,
                        num_chargers=chargers,
                        arrivals_per_hour=scale_profile_to_total(
                            BASE_ARRIVALS_PER_HOUR,
                            ev_arrivals,
                        ),
                        charger_strategy="fixed",
                        fixed_charger_fids=fixed_charger_fids,
                        distance_mode=distance_mode,
                        write_events=False,
                    )
                )

    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=_init,
        initargs=(candidates, destination_weights, existing_charger_locations, walking_network),
    ) as pool:
        results = list(pool.map(_run, runs))

    out_path = ROOT  / "output" /"scenarios"/ "scenario_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} runs to: {out_path}")

    # -----------------------------------------
    # Average results across seeds
    # -----------------------------------------

    grouped = defaultdict(list)

    for row in results:
        grouped[row["scenario"]].append(row)

    averaged_results = []

    for scenario, rows in grouped.items():
        avg_total_arrivals = mean(r["total_arrivals"] for r in rows)
        avg_charged = mean(r["charged"] for r in rows)
        averaged_results.append(
            {
                "scenario": scenario,
                "year": rows[0]["year"],
                "num_runs": len(rows),
                "distance_mode": rows[0]["distance_mode"],
                "num_chargers": mean(r["num_chargers"] for r in rows),
                "total_arrivals": avg_total_arrivals,
                "charged": avg_charged,
                "served_rate": (
                    avg_charged / avg_total_arrivals * 100
                    if avg_total_arrivals
                    else 0
                ),
                "gave_up": mean(r["gave_up"] for r in rows),
                "pct_gave_up": mean(r["pct_gave_up"] for r in rows),
                "avg_waiting_time": mean(r["avg_waiting_time"] for r in rows),
                "avg_walk_m": mean(r["avg_walk_m"] for r in rows),
                "util_mean": mean(r["util_mean"] for r in rows),
                "util_max": mean(r["util_max"] for r in rows),
            }
        )

    avg_out_path = ROOT  / "output" /"scenarios"/"scenario_results_avg.csv"

    with avg_out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(averaged_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(averaged_results)

    print(
        f"Wrote {len(averaged_results)} averaged scenarios to: "
        f"{avg_out_path}"
    )
    kpi_fit_path, kpi_fit_rows = write_kpi_fit_summary_csv(
        avg_out_path.parent,
        averaged_results,
        ev_adoption_arrivals,
        baseline_chargers,
    )
    print(f"Wrote KPI fit summary to: {kpi_fit_path}")

    selected_scenarios = [
        row["selected_scenario"]
        for row in kpi_fit_rows
        if row["selected_scenario"]
    ]
    location_csv_path, location_xlsx_path = write_charger_location_csv(
        out_path.parent,
        results,
        candidates,
        existing_charger_locations,
        selected_scenarios,
    )
    print(f"Wrote selected scenario charger locations to: {location_csv_path}")
    print(f"Wrote selected scenario location workbook to: {location_xlsx_path}")


if __name__ == "__main__":
    main()
