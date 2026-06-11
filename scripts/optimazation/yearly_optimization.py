from __future__ import annotations

import csv
import importlib.util
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = ROOT / "Jessie python files"
for module_path in (ROOT, SCRIPT_DIR, SIMULATION_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from scripts.shared.walking_network import WalkingNetwork

from optimization import (
    OptimizationSettings,
    euclidean_distance_m,
    find_minimum_feasible_chargers,
    with_service_capacity,
)


def load_simulation_module():
    """Load the simulation file from its real folder path."""
    simulation_path = SIMULATION_DIR / "simulation.py"
    spec = importlib.util.spec_from_file_location("simulation", simulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load simulation module from {simulation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulation"] = module
    spec.loader.exec_module(module)
    return module
##-----------------------------------
USE_WALKING_NETWORK_DISTANCE = False
##-----------------------------------

simulation = load_simulation_module()
CANDIDATE_LOCATIONS_PATH = simulation.CANDIDATE_LOCATIONS_PATH
BASE_ARRIVALS_PER_HOUR = simulation.BASE_ARRIVALS_PER_HOUR
DEFAULT_CHARGER_CONNECTORS = simulation.DEFAULT_CHARGER_CONNECTORS
DEFAULT_MAX_CHARGE_TIME = simulation.DEFAULT_MAX_CHARGE_TIME
DEFAULT_MIN_CHARGE_TIME = simulation.DEFAULT_MIN_CHARGE_TIME
DEFAULT_SIMULATION_TIME = simulation.DEFAULT_SIMULATION_TIME
DEFAULT_WALKING_SPEED_M_PER_MIN = simulation.DEFAULT_WALKING_SPEED_M_PER_MIN
EV_DEMAND_HEATMAP_PATH = simulation.EV_DEMAND_HEATMAP_PATH
EXISTING_CHARGERS_PATH = simulation.EXISTING_CHARGERS_PATH
HEATMAP_DENSITY_PATH = simulation.HEATMAP_DENSITY_PATH
OUTPUT_DIR = simulation.OUTPUT_DIR
WALKING_NETWORK_PATH = simulation.WALKING_NETWORK_PATH
build_simulation_config = simulation.build_simulation_config
generate_arrival_times = simulation.generate_arrival_times
load_blended_heatmap_weights = simulation.load_blended_heatmap_weights
load_candidate_locations = simulation.load_candidate_locations
load_existing_charger_locations = simulation.load_existing_charger_locations
run_simulation = simulation.run_simulation
scale_profile_to_total = simulation.scale_profile_to_total
select_fixed_locations = simulation.select_fixed_locations


EV_ADOPTION_FORECAST_PATH = (
    ROOT / "processed data" / "ev adoption" / "ev_adoption_forecast.csv"
)
DAILY_EV_ARRIVAL_SHARE = 1.0
NUMBER_OF_RUNS = 10
LOCAL_SEARCH_ROUNDS = 3



@dataclass(frozen=True)
class EvAdoptionInput:
    year: int
    total_passenger_cars: int
    ev_adoption_percentage: float
    forecast_ev_cars: int
    input_ev_arrivals: int


@dataclass(frozen=True)
class OptimizationRunConfig:
    year: int
    seed: int
    input_ev_arrivals: int
    total_passenger_cars: int
    ev_adoption_percentage: float
    forecast_ev_cars: int
    arrivals_per_hour: list[int]
    min_charge_time: int = DEFAULT_MIN_CHARGE_TIME
    max_charge_time: int = DEFAULT_MAX_CHARGE_TIME
    walking_speed_m_per_min: float = DEFAULT_WALKING_SPEED_M_PER_MIN
    max_optimized_chargers: int | None = 50
    optimization_settings: OptimizationSettings = field(
        default_factory=lambda: OptimizationSettings(
            local_search_rounds=LOCAL_SEARCH_ROUNDS
        )
    )
    write_events: bool = False


def location_fids(locations) -> str:
    """Convert selected locations to a fixed-location FID string."""
    return ";".join(str(location.fid) for location in locations)


def walking_network_distance_fn(walking_network):
    """Create a distance function that uses the real walking network."""
    def distance_m(origin, destination) -> float:
        origin_lon, origin_lat = origin.lat_lon()
        destination_lon, destination_lat = destination.lat_lon()
        return walking_network.distance_m(
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon,
        )

    return distance_m


def cached_distance_fn(distance_fn):
    """Cache repeated distance checks between the same two locations."""
    distance_cache: dict[tuple[int, int], float] = {}

    def distance_m(origin, destination) -> float:
        cache_key = (origin.fid, destination.fid)
        if cache_key not in distance_cache:
            distance_cache[cache_key] = distance_fn(origin, destination)
        return distance_cache[cache_key]

    return distance_m


def choose_distance_fn(walking_network):
    """Choose straight-line distance or real walking-network distance."""
    if USE_WALKING_NETWORK_DISTANCE:
        return cached_distance_fn(walking_network_distance_fn(walking_network))
    return cached_distance_fn(euclidean_distance_m)


def selected_distance_mode() -> str:
    """Return the distance mode used by optimization."""
    if USE_WALKING_NETWORK_DISTANCE:
        return "network"
    return "euclidean"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write result rows to a CSV file."""
    if not rows:
        raise ValueError(f"No rows to write to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def round2(value: float) -> float:
    """Round a number to 2 decimal places."""
    return round(float(value), 2)


def percent_rate(part: float, total: float) -> float:
    """Return part divided by total as a percentage."""
    if total <= 0:
        return 0.0
    return part / total * 100


def simulation_kpis_feasible(
    unmet_rate: float,
    utilization: float,
    settings: OptimizationSettings,
) -> bool:
    """Check if simulation results pass the optimization KPI limits."""
    return (
        unmet_rate / 100 <= settings.max_unmet_demand_rate
        and utilization <= settings.max_charger_utilization
    )


def load_ev_adoption_inputs(path: Path) -> dict[int, EvAdoptionInput]:
    """Load yearly EV adoption inputs from the forecast CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"EV adoption forecast not found: {path}. "
            "Run ev_adoption_forecast.py first."
        )

    ev_adoption_inputs: dict[int, EvAdoptionInput] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "year",
            "total_passenger_cars",
            "logistic_percentage",
            "logistic_ev_cars_each_year",
        }
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"EV adoption forecast is missing columns: {sorted(missing)}"
            )

        for row in reader:
            year = int(row["year"])
            forecast_ev_cars = int(round(float(row["logistic_ev_cars_each_year"])))
            input_ev_arrivals = int(round(forecast_ev_cars * DAILY_EV_ARRIVAL_SHARE))
            ev_adoption_inputs[year] = EvAdoptionInput(
                year=year,
                total_passenger_cars=int(round(float(row["total_passenger_cars"]))),
                ev_adoption_percentage=float(row["logistic_percentage"]),
                forecast_ev_cars=forecast_ev_cars,
                input_ev_arrivals=input_ev_arrivals,
            )

    if not ev_adoption_inputs:
        raise ValueError(f"EV adoption forecast has no rows: {path}")
    return ev_adoption_inputs


def average_yearly_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated optimization runs per year."""
    grouped = defaultdict(list)
    for row in results:
        grouped[row["year"]].append(row)

    averaged_results = []
    for year, rows in grouped.items():
        selected_locations = Counter(
            row["selected_locations"] for row in rows
        ).most_common(1)[0][0]
        averaged_results.append(
            {
                "year": year,
                "average_input_ev_cars": round2(mean(r["input_ev_cars"] for r in rows)),
                "average_existing_chargers_before": round2(mean(
                    r["existing_chargers_before"] for r in rows
                )),
                "average_new_chargers_added": round2(mean(
                    r["new_chargers_added"] for r in rows
                )),
                "average_minimum_chargers": round2(
                    mean(r["minimum_chargers"] for r in rows)
                ),
                "selected_locations": selected_locations,
                "average_served_rate": round2(mean(r["served_rate"] for r in rows)),
                "average_unmet_rate": round2(mean(r["unmet_rate"] for r in rows)),
                "average_served_demand": round2(mean(r["served_demand"] for r in rows)),
                "average_unmet_demand": round2(mean(r["unmet_demand"] for r in rows)),
                "average_walking_distance": round2(mean(
                    r["average_walking_distance"] for r in rows
                )),
                "average_utilization": round2(
                    mean(r["average_utilization"] for r in rows)
                ),
                "simulation_kpis_feasible": all(
                    r["simulation_kpis_feasible"] for r in rows
                ),
                "optimization_feasible": all(
                    r["optimization_feasible"] for r in rows
                ),
            }
        )

    return averaged_results


def build_year_run(
    adoption: EvAdoptionInput,
    seed: int = 0,
) -> OptimizationRunConfig:
    """Create one run setup for one year."""
    return OptimizationRunConfig(
        year=adoption.year,
        seed=seed,
        input_ev_arrivals=adoption.input_ev_arrivals,
        total_passenger_cars=adoption.total_passenger_cars,
        ev_adoption_percentage=adoption.ev_adoption_percentage,
        forecast_ev_cars=adoption.forecast_ev_cars,
        arrivals_per_hour=scale_profile_to_total(
            BASE_ARRIVALS_PER_HOUR,
            adoption.input_ev_arrivals,
        ),
        write_events=False,
    )


def run_one_optimization_year(
    candidate_locations,
    destination_weights,
    existing_charger_locations,
    walking_network,
    cfg: OptimizationRunConfig,
) -> dict[str, Any]:
    """Run optimization and simulation for one year and seed."""
    arrival_rng = random.Random()
    arrival_rng.seed(cfg.seed)
    arrival_times = generate_arrival_times(arrival_rng, cfg.arrivals_per_hour)

    settings = with_service_capacity(
        cfg.optimization_settings,
        DEFAULT_SIMULATION_TIME,
        cfg.min_charge_time,
        cfg.max_charge_time,
    )
    distance_fn = choose_distance_fn(walking_network)
    min_new_chargers = 0

    while True:
        optimization_result = find_minimum_feasible_chargers(
            candidate_locations,
            destination_weights,
            total_demand=len(arrival_times),
            settings=settings,
            min_chargers=min_new_chargers,
            max_chargers=cfg.max_optimized_chargers,
            distance_fn=distance_fn,
            existing_locations=existing_charger_locations,
        )
        selected_locations = optimization_result.selected_locations
        selected_fids = location_fids(selected_locations)
        new_chargers_added = (
            len(optimization_result.new_locations)
            if optimization_result.new_locations is not None
            else 0
        )

        events_dir = ROOT / "output" / "optimization" / "optimization_events"
        events_dir.mkdir(parents=True, exist_ok=True)
        sim_config = build_simulation_config(
            description=f"optimized_{cfg.year}",
            charger_strategy="fixed",
            num_chargers=len(selected_locations),
            arrival_times=arrival_times,
            seed=cfg.seed,
            min_charge_time=cfg.min_charge_time,
            max_charge_time=cfg.max_charge_time,
            walking_threshold_m=settings.walking_threshold_m,
            walking_speed_m_per_min=cfg.walking_speed_m_per_min,
            fixed_charger_fids=selected_fids,
            charger_connectors=settings.connectors_per_new_charger,
            write_events=cfg.write_events,
        )
        sim_rng = random.Random()
        sim_rng.seed(cfg.seed)
        sim_result = run_simulation(
            f"optimized_{cfg.year}_seed{cfg.seed}_new{new_chargers_added}",
            sim_config,
            candidate_locations,
            destination_weights,
            existing_charger_locations,
            walking_network,
            events_dir,
            rng=sim_rng,
        )
        served_demand = sim_result["completed_charging"]
        unmet_demand = sim_result["gave_up"]
        unmet_rate = round2(percent_rate(unmet_demand, cfg.input_ev_arrivals))
        utilization = round2(sim_result["avg_charger_utilization"])
        sim_feasible = simulation_kpis_feasible(unmet_rate, utilization, settings)

        if sim_feasible:
            return {
                "year": cfg.year,
                "seed": cfg.seed,
                "input_ev_cars": cfg.input_ev_arrivals,
                "existing_chargers_before": len(existing_charger_locations),
                "new_chargers_added": new_chargers_added,
                "minimum_chargers": sim_result["num_chargers"],
                "selected_locations": selected_fids,
                "served_demand": served_demand,
                "served_rate": round2(
                    percent_rate(served_demand, cfg.input_ev_arrivals)
                ),
                "unmet_demand": unmet_demand,
                "unmet_rate": unmet_rate,
                "average_walking_distance": round2(sim_result["avg_walking_dist_m"]),
                "average_utilization": utilization,
                "optimization_coverage_rate": round2(
                    optimization_result.kpis.coverage_rate
                ),
                "optimization_unmet_demand_rate": round2(
                    optimization_result.kpis.unmet_demand_rate
                ),
                "simulation_kpis_feasible": sim_feasible,
                "optimization_feasible": optimization_result.feasible,
            }

        min_new_chargers = new_chargers_added + 1
        if (
            cfg.max_optimized_chargers is not None
            and min_new_chargers > cfg.max_optimized_chargers
        ):
            raise ValueError(
                f"Simulation KPIs failed for {cfg.year}, seed {cfg.seed}, "
                f"even with {new_chargers_added} new chargers: "
                f"unmet_rate={unmet_rate}% "
                f"(limit {settings.max_unmet_demand_rate * 100:.2f}%), "
                f"utilization={utilization} "
                f"(limit {settings.max_charger_utilization:.2f})."
            )
        print(
            f"Simulation KPIs failed for {cfg.year}, seed {cfg.seed} "
            f"with {new_chargers_added} new chargers "
            f"(unmet_rate={unmet_rate}%, utilization={utilization}); "
            f"trying {min_new_chargers} new chargers."
        )


def run_cumulative_yearly_optimization(
    ev_adoption_inputs: dict[int, EvAdoptionInput],
    candidates,
    destination_weights,
    existing_charger_locations,
    walking_network,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run years in order, carrying chargers forward each year."""
    results: list[dict[str, Any]] = []
    current_charger_locations = list(existing_charger_locations)
    all_locations = list(existing_charger_locations) + list(candidates)

    for year, adoption in sorted(ev_adoption_inputs.items()):
        cfg = build_year_run(adoption, seed=seed)
        result = run_one_optimization_year(
            candidates,
            destination_weights,
            current_charger_locations,
            walking_network,
            cfg,
        )
        results.append(result)
        current_charger_locations = select_fixed_locations(
            all_locations,
            result["selected_locations"],
        )

    return results


def run_cumulative_yearly_repetitions(
    ev_adoption_inputs: dict[int, EvAdoptionInput],
    candidates,
    destination_weights,
    existing_charger_locations,
    walking_network,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Run cumulative yearly optimization for several random seeds."""
    results: list[dict[str, Any]] = []
    for seed in seeds:
        results.extend(
            run_cumulative_yearly_optimization(
                ev_adoption_inputs,
                candidates,
                destination_weights,
                existing_charger_locations,
                walking_network,
                seed=seed,
            )
        )
    return results


def main() -> None:
    """Run the full yearly charger optimization workflow."""
    candidates = load_candidate_locations(CANDIDATE_LOCATIONS_PATH)
    existing_charger_locations = load_existing_charger_locations(EXISTING_CHARGERS_PATH)
    destination_weights = load_blended_heatmap_weights(
        candidates, HEATMAP_DENSITY_PATH, EV_DEMAND_HEATMAP_PATH
    )
    walking_network = WalkingNetwork.from_geojson(WALKING_NETWORK_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Using distance mode: {selected_distance_mode()}")

    ev_adoption_inputs = load_ev_adoption_inputs(
        EV_ADOPTION_FORECAST_PATH
    )
    seeds = list(range(NUMBER_OF_RUNS))
    if not seeds:
        raise ValueError("NUMBER_OF_RUNS must be at least 1.")

    results = run_cumulative_yearly_repetitions(
        ev_adoption_inputs,
        candidates,
        destination_weights,
        existing_charger_locations,
        walking_network,
        seeds,
    )
    
    out_path = ROOT / "output" / "optimization"/"optimization_results.csv"
    write_csv(out_path, results)
    print(f"Wrote {len(results)} optimization runs to: {out_path}")

    averaged_results = average_yearly_results(results)
    avg_out_path = ROOT / "output" / "optimization"/ "optimization_results_avg.csv"
    write_csv(avg_out_path, averaged_results)
    print(
        f"Wrote {len(averaged_results)} averaged optimization years to: "
        f"{avg_out_path}"
    )


if __name__ == "__main__":
    main()
