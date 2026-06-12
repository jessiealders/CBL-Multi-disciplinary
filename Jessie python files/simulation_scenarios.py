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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared.walking_network import WalkingNetwork

from simulation import (
    BASE_ARRIVALS_PER_HOUR,
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


USE_WALKING_NETWORK_DISTANCE = False


######################
# One-run simulation
#####################

@dataclass(frozen=True)
class RunConfig:
    scenario: str
    seed: int
    num_chargers: int
    arrivals_per_hour: list[int]
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
        charger_strategy="random",
        num_chargers=cfg.num_chargers,
        arrival_times=arrival_times,
        seed=cfg.seed,
        min_charge_time=cfg.min_charge_time,
        max_charge_time=cfg.max_charge_time,
        walking_threshold_m=cfg.walking_threshold_m,
        walking_speed_m_per_min=cfg.walking_speed_m_per_min,
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

    runs: list[RunConfig] = []

    # Scenario 1: current infrastructure
    for seed in seeds:
        runs.append(
            RunConfig(
                scenario="baseline",
                seed=seed,
                num_chargers=4,
                arrivals_per_hour=BASE_ARRIVALS_PER_HOUR,
                distance_mode=distance_mode,
                write_events=False,
            )
        )

    # Scenario 2: more cars arriving
    for seed in seeds:
        runs.append(
            RunConfig(
                scenario="more_cars_50pct",
                seed=seed,
                num_chargers=4,
                arrivals_per_hour=scale_profile_to_total(
                    BASE_ARRIVALS_PER_HOUR,
                    int(round(sum(BASE_ARRIVALS_PER_HOUR) * 1.5)),
                ),
                distance_mode=distance_mode,
                write_events=False,
            )
        )

    # Scenario 3: linearly more chargers
    for chargers in [4, 6, 8, 10, 14, 18, 22, 26, 30]:
        for seed in seeds:
            runs.append(
                RunConfig(
                    scenario=f"add_chargers_{chargers}",
                    seed=seed,
                    num_chargers=chargers,
                    arrivals_per_hour=BASE_ARRIVALS_PER_HOUR,
                    distance_mode=distance_mode,
                    write_events=False,
                )
            )

    # Scenario 4: increase demand and chargers at the same time
    # For each charger level, run multiple demand multipliers.
    # Demand estimate based on extensive research
    # 2026, 2028, 2030, 2032, 2034
    demand_multipliers = [1.0, 1.5, 2.2, 3.0, 4.0]
    charger_levels = [4, 6, 8, 10, 14, 18, 22, 26, 30]

    for chargers in charger_levels:
        for mult in demand_multipliers:
            for seed in seeds:
                mult_label = str(mult).replace(".", "p")
                runs.append(
                    RunConfig(
                        scenario=f"scale_both_c{chargers}_m{mult_label}",
                        seed=seed,
                        num_chargers=chargers,
                        arrivals_per_hour=scale_profile_to_total(
                            BASE_ARRIVALS_PER_HOUR,
                            int(round(sum(BASE_ARRIVALS_PER_HOUR) * mult)),
                        ),
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
        averaged_results.append(
            {
                "scenario": scenario,
                "num_runs": len(rows),
                "distance_mode": rows[0]["distance_mode"],
                "num_chargers": mean(r["num_chargers"] for r in rows),
                "total_arrivals": mean(r["total_arrivals"] for r in rows),
                "charged": mean(r["charged"] for r in rows),
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


if __name__ == "__main__":
    main()
