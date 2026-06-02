from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation import load_candidate_locations, p, run_simulation

ROOT = Path(__file__).resolve().parents[1]


# -------------------------
# Arrival generation
# -------------------------


def generate_arrival_times(arrivals_per_hour: list[int]) -> list[float]:
    """Return arrival times in minutes in [0, 1440).

    For each hour, generate N arrivals uniformly at random inside that hour.
    """
    if len(arrivals_per_hour) != 24:
        raise ValueError("arrivals_per_hour must have length 24")

    times: list[float] = []
    for h, n in enumerate(arrivals_per_hour):
        start = h * 60
        for _ in range(int(n)):
            times.append(start + random.random() * 60)
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


def run_one_day(candidate_locations, cfg: RunConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    arrival_times = generate_arrival_times(cfg.arrivals_per_hour)

    src, chosen_locations = run_simulation(
        candidate_locations=candidate_locations,
        num_chargers=cfg.num_chargers,
        arrival_times=arrival_times,
        simulation_time=24 * 60,
        min_charge_time=cfg.min_charge_time,
        max_charge_time=cfg.max_charge_time,
        walking_threshold_m=cfg.walking_threshold_m,
        walking_speed_m_per_min=cfg.walking_speed_m_per_min,
        seed=cfg.seed,
        verbose=False,
    )

    total = len(arrival_times)
    charged = sum(1 for car in src.cars if getattr(car, "status", None) == "charged")
    gave_up = sum(1 for car in src.cars if getattr(car, "status", None) == "gave_up")

    wait_times = [car.waitingTime for car in src.cars if car.waitingTime is not None]
    walk_dists = [car.walkingDist for car in src.cars if car.walkingDist is not None]

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else None
    avg_walk = sum(walk_dists) / len(walk_dists) if walk_dists else None

    utilizations = [c.chargingTime / (24 * 60 * c.capacity) for c in src.chargers]
    util_mean = sum(utilizations) / len(utilizations) if utilizations else None
    util_max = max(utilizations) if utilizations else None

    chosen_fids = [c.fid for c in chosen_locations]

    if cfg.write_events:
        out_dir = ROOT / "other data" / "scenario_events"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg.scenario}_seed{cfg.seed}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            if src.events:
                writer = csv.DictWriter(f, fieldnames=sorted({k for e in src.events for k in e.keys()}))
                writer.writeheader()
                writer.writerows(src.events)

    return {
        "scenario": cfg.scenario,
        "seed": cfg.seed,
        "num_chargers": cfg.num_chargers,
        "total_arrivals": total,
        "charged": charged,
        "gave_up": gave_up,
        "pct_gave_up": (gave_up / total * 100) if total else 0.0,
        "avg_wait_min": avg_wait,
        "avg_walk_m": avg_walk,
        "util_mean": util_mean,
        "util_max": util_max,
        "chosen_fids": ";".join(map(str, chosen_fids)),
    }


###################
# # Scenario series
###################

def scale_profile(profile: list[int], factor: float) -> list[int]:
    return [int(round(x * factor)) for x in profile]


def main() -> None:
    candidates = load_candidate_locations(p(r"other data\freepacement_lessdata_strijp_lili.csv"))

    # A simple default "realistic-ish" arrival curve.
    # We can tune these numbers later
    base_profile = [
        1, 1, 1, 1, 1, 2,  # 00-05
        5, 8, 8, 6,        # 06-09
        4, 4,              # 10-11
        6, 6,              # 12-13
        4, 4, 5,           # 14-16
        9, 9, 7,           # 17-19
        4, 3, 2, 1,        # 20-23
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
    demand_multipliers = [1.0, 1.5, 2.0, 2.5, 3]
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
    for cfg in runs:
        results.append(run_one_day(candidates, cfg))

    out_path = ROOT / "other data" / "scenario_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} runs to: {out_path}")


if __name__ == "__main__":
    main()
