from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Protocol


class LocationLike(Protocol):
    fid: int
    x: float
    y: float
    postcode: str | None
    connectors: int | None


DistanceFn = Callable[[LocationLike, LocationLike], float]


@dataclass(frozen=True)
class OptimizationSettings:
    """KPI thresholds and planning assumptions for charger placement."""

    # Walking coverage note only, not an active feasibility KPI:
    # min_coverage_rate = 0.80 and walking_threshold_m = 300 m.
    walking_threshold_m: float = 300.0
    min_coverage_rate: float = 0.80
    max_unmet_demand_rate: float = 0.10
    max_charger_utilization: float = 0.85
    connectors_per_new_charger: int = 2
    service_capacity_per_connector: float | None = None
    future_demand_multiplier: float = 1.0
    unmet_demand_penalty_m: float = 10_000.0
    local_search_rounds: int = 3


@dataclass(frozen=True)
class OptimizationKpis:
    coverage_rate: float
    avg_walking_dist_m: float
    unmet_demand_rate: float
    charger_utilization: float
    objective_value: float


@dataclass(frozen=True)
class OptimizationResult:
    selected_locations: list[LocationLike]
    kpis: OptimizationKpis
    feasible: bool
    new_locations: list[LocationLike] | None = None


def euclidean_distance_m(origin: LocationLike, destination: LocationLike) -> float:
    """Calculate the straight-line distance between two locations."""
    dx = origin.x - destination.x
    dy = origin.y - destination.y
    return (dx * dx + dy * dy) ** 0.5


def normalize_weights(
    locations: list[LocationLike], destination_weights: Iterable[float] | None
) -> list[float]:
    """Prepare demand weights, or use equal weights if none are useful."""
    if destination_weights is None:
        return [1.0 for _ in locations]

    weights = [max(0.0, float(weight)) for weight in destination_weights]
    if len(weights) != len(locations):
        raise ValueError(
            "destination_weights must have the same length as candidate locations."
        )
    if sum(weights) <= 0:
        return [1.0 for _ in locations]
    return weights


def filter_grid_feasible_locations(
    locations: list[LocationLike], grid_feasibility: dict[str, bool] | None = None
) -> list[LocationLike]:
    """Keep locations that are allowed by the grid data."""
    if not grid_feasibility:
        return list(locations)

    feasible: list[LocationLike] = []
    for location in locations:
        postcode = (location.postcode or "").replace(" ", "").upper()
        if not postcode or grid_feasibility.get(postcode, True):
            feasible.append(location)
    return feasible


def estimate_service_capacity_per_connector(
    simulation_time: float, min_charge_time: float, max_charge_time: float
) -> float:
    """Estimate how many EVs one connector can serve."""
    average_charge_time = (min_charge_time + max_charge_time) / 2
    if average_charge_time <= 0:
        raise ValueError("Average charge time must be positive.")
    return simulation_time / average_charge_time


def with_service_capacity(
    settings: OptimizationSettings,
    simulation_time: float,
    min_charge_time: float,
    max_charge_time: float,
) -> OptimizationSettings:
    """Add service capacity to the optimization settings."""
    service_capacity = estimate_service_capacity_per_connector(
        simulation_time,
        min_charge_time,
        max_charge_time,
    )
    return replace(settings, service_capacity_per_connector=service_capacity)


def location_connector_count(
    location: LocationLike, settings: OptimizationSettings
) -> int:
    """Use known connectors, or use the default for a new charger."""
    connectors = getattr(location, "connectors", None)
    if connectors is None:
        return settings.connectors_per_new_charger
    return max(1, int(connectors))


def total_raw_service_capacity(
    selected_locations: list[LocationLike], settings: OptimizationSettings
) -> float:
    """Calculate total charging capacity before utilization limits."""
    if settings.service_capacity_per_connector is None:
        raise ValueError("service_capacity_per_connector must be set before evaluation.")
    total_connectors = sum(
        location_connector_count(location, settings)
        for location in selected_locations
    )
    return (
        total_connectors * settings.service_capacity_per_connector
    )


def evaluate_locations(
    selected_locations: list[LocationLike],
    destination_locations: list[LocationLike],
    destination_weights: list[float],
    total_demand: float,
    settings: OptimizationSettings,
    distance_fn: DistanceFn = euclidean_distance_m,
) -> OptimizationKpis:
    """Calculate KPIs for one selected charger layout."""
    if not selected_locations:
        unmet_rate = 1.0 if total_demand > 0 else 0.0
        return OptimizationKpis(
            coverage_rate=0.0,
            avg_walking_dist_m=0.0,
            unmet_demand_rate=unmet_rate,
            charger_utilization=0.0,
            objective_value=settings.unmet_demand_penalty_m * unmet_rate,
        )

    weighted_distance = 0.0
    covered_weight = 0.0
    weight_total = sum(destination_weights)

    for destination, weight in zip(destination_locations, destination_weights):
        nearest_distance = min(
            distance_fn(destination, charger) for charger in selected_locations
        )
        weighted_distance += nearest_distance * weight
        if nearest_distance <= settings.walking_threshold_m:
            covered_weight += weight

    raw_capacity = total_raw_service_capacity(selected_locations, settings)
    usable_capacity = raw_capacity * settings.max_charger_utilization
    served_demand = min(total_demand, usable_capacity)
    unmet_demand = max(0.0, total_demand - usable_capacity)
    unmet_rate = unmet_demand / total_demand if total_demand > 0 else 0.0
    utilization = served_demand / raw_capacity if raw_capacity > 0 else 0.0
    avg_walking_dist = weighted_distance / weight_total if weight_total > 0 else 0.0
    coverage_rate = covered_weight / weight_total if weight_total > 0 else 0.0

    return OptimizationKpis(
        coverage_rate=coverage_rate,
        avg_walking_dist_m=avg_walking_dist,
        unmet_demand_rate=unmet_rate,
        charger_utilization=utilization,
        objective_value=(
            weighted_distance
            + settings.unmet_demand_penalty_m * unmet_rate
        ),
    )


def is_feasible(kpis: OptimizationKpis, settings: OptimizationSettings) -> bool:
    """Check if a charger layout passes the KPI limits."""
    return (
        kpis.unmet_demand_rate <= settings.max_unmet_demand_rate
        and kpis.charger_utilization <= settings.max_charger_utilization
    )


def optimize_fixed_number(
    candidate_locations: list[LocationLike],
    number_chargers: int,
    destination_weights: Iterable[float] | None,
    total_demand: float,
    settings: OptimizationSettings,
    distance_fn: DistanceFn = euclidean_distance_m,
    grid_feasibility: dict[str, bool] | None = None,
    existing_locations: list[LocationLike] | None = None,
) -> OptimizationResult:
    """Choose the best locations for a fixed number of chargers."""
    if number_chargers < 0:
        raise ValueError("number_chargers must be at least 0.")

    existing_locations = existing_locations or []
    existing_fids = {location.fid for location in existing_locations}
    feasible_candidates = filter_grid_feasible_locations(
        candidate_locations, grid_feasibility
    )
    feasible_candidates = [
        location for location in feasible_candidates
        if location.fid not in existing_fids
    ]
    if number_chargers > len(feasible_candidates):
        raise ValueError(
            f"Requested {number_chargers} chargers but only "
            f"{len(feasible_candidates)} grid-feasible candidates are available."
        )

    weights = normalize_weights(candidate_locations, destination_weights)
    scaled_total_demand = total_demand * settings.future_demand_multiplier

    selected_new: list[LocationLike] = []
    remaining = list(feasible_candidates)

    while len(selected_new) < number_chargers:
        best_candidate = min(
            remaining,
            key=lambda candidate: evaluate_locations(
                existing_locations + selected_new + [candidate],
                candidate_locations,
                weights,
                scaled_total_demand,
                settings,
                distance_fn,
            ).objective_value,
        )
        selected_new.append(best_candidate)
        remaining.remove(best_candidate)

    selected_new = improve_by_swapping(
        selected_new,
        remaining,
        candidate_locations,
        weights,
        scaled_total_demand,
        settings,
        distance_fn,
        fixed_locations=existing_locations,
    )
    all_selected = existing_locations + selected_new

    kpis = evaluate_locations(
        all_selected,
        candidate_locations,
        weights,
        scaled_total_demand,
        settings,
        distance_fn,
    )
    return OptimizationResult(
        all_selected, kpis, is_feasible(kpis, settings), new_locations=selected_new
    )


def improve_by_swapping(
    selected: list[LocationLike],
    remaining: list[LocationLike],
    destination_locations: list[LocationLike],
    destination_weights: list[float],
    total_demand: float,
    settings: OptimizationSettings,
    distance_fn: DistanceFn,
    fixed_locations: list[LocationLike] | None = None,
) -> list[LocationLike]:
    """Improve selected chargers by swapping with unused candidates."""
    fixed_locations = fixed_locations or []
    current = list(selected)
    available = list(remaining)
    current_score = evaluate_locations(
        fixed_locations + current,
        destination_locations,
        destination_weights,
        total_demand,
        settings,
        distance_fn,
    ).objective_value

    for _ in range(settings.local_search_rounds):
        improved = False
        for selected_location in list(current):
            for candidate in list(available):
                trial = [
                    candidate if location == selected_location else location
                    for location in current
                ]
                trial_score = evaluate_locations(
                    fixed_locations + trial,
                    destination_locations,
                    destination_weights,
                    total_demand,
                    settings,
                    distance_fn,
                ).objective_value
                if trial_score + 1e-9 < current_score:
                    current.remove(selected_location)
                    available.append(selected_location)
                    available.remove(candidate)
                    current.append(candidate)
                    current_score = trial_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current


def find_minimum_feasible_chargers(
    candidate_locations: list[LocationLike],
    destination_weights: Iterable[float] | None,
    total_demand: float,
    settings: OptimizationSettings,
    min_chargers: int = 1,
    max_chargers: int | None = None,
    distance_fn: DistanceFn = euclidean_distance_m,
    grid_feasibility: dict[str, bool] | None = None,
    existing_locations: list[LocationLike] | None = None,
) -> OptimizationResult:
    """Find the smallest charger count that passes the KPI limits."""
    feasible_candidates = filter_grid_feasible_locations(
        candidate_locations, grid_feasibility
    )
    upper_bound = max_chargers or len(feasible_candidates)

    best_result: OptimizationResult | None = None
    for number_chargers in range(min_chargers, upper_bound + 1):
        result = optimize_fixed_number(
            candidate_locations,
            number_chargers,
            destination_weights,
            total_demand,
            settings,
            distance_fn,
            grid_feasibility,
            existing_locations,
        )
        best_result = result
        if result.feasible:
            return result

    if best_result is None:
        raise ValueError("No candidate locations available for optimization.")
    raise ValueError(
        f"No feasible charger layout found between {min_chargers} and "
        f"{upper_bound} new chargers. Last KPI values: "
        f"unmet_demand_rate={best_result.kpis.unmet_demand_rate * 100:.2f}% "
        f"(limit {settings.max_unmet_demand_rate * 100:.2f}%), "
        f"charger_utilization={best_result.kpis.charger_utilization:.2f} "
        f"(limit {settings.max_charger_utilization:.2f})."
    )


def select_optimized_locations(
    strategy: str,
    number_chargers: int,
    candidate_locations: list[LocationLike],
    destination_weights: Iterable[float] | None,
    config: dict,
    distance_fn: DistanceFn = euclidean_distance_m,
    grid_feasibility: dict[str, bool] | None = None,
    existing_locations: list[LocationLike] | None = None,
) -> OptimizationResult:
    """Build settings from a simulation config and select charger locations."""
    settings = with_service_capacity(
        OptimizationSettings(
            walking_threshold_m=config.get("walking_threshold_m", 300.0),
            min_coverage_rate=config.get("min_coverage_rate", 0.80),
            max_unmet_demand_rate=config.get("max_unmet_demand_rate", 0.10),
            max_charger_utilization=config.get("max_charger_utilization", 0.85),
            connectors_per_new_charger=config.get("connectors_per_new_charger", 2),
            future_demand_multiplier=config.get("future_demand_multiplier", 1.0),
            unmet_demand_penalty_m=config.get("unmet_demand_penalty_m", 10_000.0),
        ),
        config["simulation_time"],
        config["min_charge_time"],
        config["max_charge_time"],
    )
    if config.get("service_capacity_per_connector") is not None:
        settings = replace(
            settings,
            service_capacity_per_connector=config["service_capacity_per_connector"],
        )

    if strategy == "optimized_minimum":
        return find_minimum_feasible_chargers(
            candidate_locations,
            destination_weights,
            total_demand=config["num_cars"],
            settings=settings,
            min_chargers=1,
            max_chargers=config.get("max_optimized_chargers", number_chargers),
            distance_fn=distance_fn,
            grid_feasibility=grid_feasibility,
            existing_locations=existing_locations,
        )

    return optimize_fixed_number(
        candidate_locations,
        number_chargers,
        destination_weights,
        total_demand=config["num_cars"],
        settings=settings,
        distance_fn=distance_fn,
        grid_feasibility=grid_feasibility,
        existing_locations=existing_locations,
    )
