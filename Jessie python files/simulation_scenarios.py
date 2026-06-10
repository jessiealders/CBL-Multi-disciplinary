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
    CANDIDATE_LOCATIONS_PATH,
    EV_DEMAND_HEATMAP_PATH,
    EXISTING_CHARGERS_PATH,
    HEATMAP_DENSITY_PATH,
    OUTPUT_DIR,
    WALKING_NETWORK_PATH,
    load_blended_heatmap_weights,
    load_candidate_locations,
    load_existing_charger_locations,
    run_simulation,
)


# -------------------------
# Arrival generation
# -------------------------


def generate_arrival_times(rng: random.Random, arrivals_per_hour: list[int]) -> list[float]:
    """Return arrival times in minutes in [0, 1440).

    For each hour, generate N arrivals uniformly at random inside that hour.
    """
    if len(arrivals_per_hour) != 24:
        raise ValueError("arrivals_per_hour must have length 24")

    times: list[float] = []
    for h, n in enumerate(arrivals_per_hour):
        start = h * 60
        for _ in range(int(n)):
            times.append(start + rng.random() * 60)
    times.sort()
    return times

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

    events_dir = ROOT / "other data" / "scenario_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "description": cfg.scenario,
        "charger_strategy": "random",
        "num_cars": len(arrival_times),
        "num_chargers": cfg.num_chargers,
        "arrival_times": arrival_times,
        "simulation_time": 24 * 60,
        "min_charge_time": cfg.min_charge_time,
        "max_charge_time": cfg.max_charge_time,
        "walking_threshold_m": cfg.walking_threshold_m,
        "walking_speed_m_per_min": cfg.walking_speed_m_per_min,
        "seed": cfg.seed,
        "verbose": False,
        "write_events": cfg.write_events,
    }
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

def scale_profile(profile: list[int], factor: float) -> list[int]:
    return [int(round(x * factor)) for x in profile]

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

def main() -> None:
    candidates = load_candidate_locations(CANDIDATE_LOCATIONS_PATH)
    existing_charger_locations = load_existing_charger_locations(EXISTING_CHARGERS_PATH)
    destination_weights = load_blended_heatmap_weights(
        candidates, HEATMAP_DENSITY_PATH, EV_DEMAND_HEATMAP_PATH
    )
    walking_network = WalkingNetwork.from_geojson(WALKING_NETWORK_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # A simple default "realistic-ish" arrival curve.
    # We can tune these numbers later
    # Research concluded 500 cars in Strijp-s per avg
    base_profile = [
        5, 5, 5, 5, 5, 10,      # 00-05
        25, 39, 39, 29,         # 06-09
        20, 20,                 # 10-11
        29, 29,                 # 12-13
        20, 20, 25,             # 14-16
        44, 44, 34,             # 17-19
        20, 15, 10, 5,          # 20-23
    ]

    seeds = list(range(10))  # bump to 20/50 for better statistics

    runs: list[RunConfig] = []

    # Scenario 1: current infrastructure
    for seed in seeds:
        runs.append(
            RunConfig(
                scenario="baseline",
                seed=seed,
                num_chargers=4,
                arrivals_per_hour=base_profile,
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
                arrivals_per_hour=scale_profile(base_profile, 1.5),
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
                    arrivals_per_hour=base_profile,
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
                        arrivals_per_hour=scale_profile(base_profile, mult),
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

    out_path = ROOT / "other data" / "scenario_results.csv"
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

    avg_out_path = ROOT / "other data" / "scenario_results_avg.csv"

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
