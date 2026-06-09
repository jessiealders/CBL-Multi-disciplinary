import simpy
import random
import csv
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shared.walking_network import WalkingNetwork

# -----------------------------
# Paths and scenario settings
# -----------------------------

CANDIDATE_LOCATIONS_PATH = (
    ROOT / "processed data" / "freepacement_lessdata_strijp_lili.csv"
)
EXISTING_CHARGERS_PATH = (
    ROOT
    / "processed data"
    / "charging points"
    / "existing_charging_points_strijp_s.csv"
)
HEATMAP_DENSITY_PATH = ROOT / "processed data" / "gpx_heatmap_density.npz"
EV_DEMAND_HEATMAP_PATH = ROOT / "processed data" / "heatmap4_density.npz"
WALKING_NETWORK_PATH = ROOT / "processed data" / "spatial" / "walking_traces.geojson"
OUTPUT_DIR = ROOT / "processed data" / "simulation"

RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
RD_TO_WEB_MERCATOR = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)

FIXED_EXISTING_FIDS = {1, 7, 17, 19}


SCENARIOS = {
    "current_situation": {
        "description": "Current charging points in Strijp-S.",
        "charger_strategy": "existing",
        "num_cars": 40,
        "num_chargers": 5,
        "simulation_time": 200,
        "min_charge_time": 1,
        "max_charge_time": 30,
        "max_wait_time": 5,
        "walking_threshold_m": 300,
        "seed": 10,
    },
}


# -----------------------------
# Coordinate helpers
# -----------------------------


def rd_to_wgs84(x, y):
    """Convert Dutch RD coordinates (EPSG:28992) to latitude/longitude."""
    lon, lat = RD_TO_WGS84.transform(x, y)
    return lat, lon

def p(rel_windows_path: str) -> Path:
    """Windows to POSIX path conversion."""
    return ROOT.joinpath(*rel_windows_path.split("\\"))

# -----------------------------
# Data models and loading
# -----------------------------


@dataclass(frozen=True)
class CandidateLocation:
    fid: int
    identificatie: str
    x: float
    y: float
    max_area: float
    postcode: str | None = None

    def lat_lon(self):
        return rd_to_wgs84(self.x, self.y)


def load_candidate_locations(path: Path) -> list[CandidateLocation]:
    """Load candidate charger locations (centroids) from the CSV."""
    locations: list[CandidateLocation] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Prefer X_coordinate/Y_coordinate; fall back to X/Y.
            x = float(row.get("X_coordinate") or row.get("X") or row["X"])
            y = float(row.get("Y_coordinate") or row.get("Y") or row["Y"])
            fid = int(float(row.get("fid") or 0))
            identificatie = (row.get("identificatie") or "").strip().strip('"')
            max_area = float(row.get("Max_area") or 0.0)
            postcode = (row.get("addr:postcode") or "").strip() or None
            locations.append(
                CandidateLocation(
                    fid=fid,
                    identificatie=identificatie,
                    x=x,
                    y=y,
                    max_area=max_area,
                    postcode=postcode,
                )
            )
    return locations


def load_existing_charger_locations(path: Path) -> list[CandidateLocation]:
    """Load existing Strijp-S charging points as charger locations."""
    locations: list[CandidateLocation] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            x = float(row["X_coordinate"])
            y = float(row["Y_coordinate"])
            charger_id = row.get("charger_id") or index
            locations.append(
                CandidateLocation(
                    fid=int(float(charger_id)),
                    identificatie=(row.get("geovisia_id") or "").strip(),
                    x=x,
                    y=y,
                    max_area=0.0,
                    postcode=(row.get("most_common_postcode") or "").strip() or None,
                )
            )
    return locations


def _sample_density(
    locations: list["CandidateLocation"], density_path: Path
) -> np.ndarray | None:
    if not density_path.exists():
        print(f"Heatmap density file not found: {density_path}.")
        return None
    data = np.load(density_path)
    counts = data["counts"]  # shape (bins_y, bins_x), EPSG:3857
    xmin, xmax = float(data["xmin"]), float(data["xmax"])
    ymin, ymax = float(data["ymin"]), float(data["ymax"])
    bins_y, bins_x = counts.shape

    samples = np.empty(len(locations))
    for i, loc in enumerate(locations):
        x3857, y3857 = RD_TO_WEB_MERCATOR.transform(loc.x, loc.y)
        ix = int((x3857 - xmin) / (xmax - xmin) * bins_x)
        iy = int((y3857 - ymin) / (ymax - ymin) * bins_y)
        ix = max(0, min(ix, bins_x - 1))
        iy = max(0, min(iy, bins_y - 1))
        samples[i] = counts[iy, ix]
    return samples


def load_blended_heatmap_weights(
    locations: list["CandidateLocation"],
    gpx_path: Path,
    ev_path: Path,
    gpx_share: float = 0.20,
    ev_share: float = 0.80,
) -> list[float] | None:
    gpx_samples = _sample_density(locations, gpx_path)
    ev_samples = _sample_density(locations, ev_path)

    if gpx_samples is None and ev_samples is None:
        print("Both heatmap files missing. Using uniform destination weights.")
        return None

    def _normalise(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return arr / hi if hi > 0 else arr

    if gpx_samples is None:
        blended = _normalise(ev_samples)
    elif ev_samples is None:
        blended = _normalise(gpx_samples)
    else:
        blended = gpx_share * _normalise(gpx_samples) + ev_share * _normalise(
            ev_samples
        )

    # +1 baseline so no location has zero probability
    return [float(v) + 1.0 for v in blended]


# -----------------------------
# Simulation entities
# -----------------------------


class Source:
    """
    Source: works as simulation generator.
    Generates a given number of cars and chargers.
    Stores cars and chargers in list so we can easily access them throughout the simulation.
    Parameters: environment, number of cars, number of chargers to generate
    """

    def __init__(
        self,
        env,
        config,
        number_cars,
        number_chargers,
        candidate_locations=None,
        charger_locations=None,
        destination_weights=None,
        walking_network=None,
        verbose=False,
    ):
        self.env = env
        self.config = config
        self.number_cars = number_cars
        self.number_chargers = number_chargers
        self.chargers = []
        self.cars = []
        self.verbose = verbose
        self.events: list[dict] = []
        self.candidate_locations = candidate_locations or []
        self.fixed_charger_locations = charger_locations
        self.destination_weights = destination_weights
        self.walking_network = walking_network
        self.chosen_charger_locations: list[CandidateLocation] = []
        self.action = env.process(self.generate())

    def log(self, kind: str, car_name: str, msg: str, **payload):
        """Store a structured event; optionally print it."""
        row = {
            "t": float(self.env.now),
            "kind": kind,
            "car": car_name,
            "msg": msg,
            **payload,
        }
        self.events.append(row)
        if self.verbose:
            print(f"{self.env.now:.2f} {car_name} {kind}: {msg}")

    def generate(self):
        """
        Generates number of chargers and cars based on the given numbers,
        And stores these in lists self.chargers and self.cars so we can access them.
        Returns: None
        """
        # Generate chargers and add them to the list of chargers.
        # If we have candidate locations, pick unique locations at random (no repeats).
        if self.fixed_charger_locations is not None:
            self.chosen_charger_locations = list(self.fixed_charger_locations)
            self.number_chargers = len(self.chosen_charger_locations)
        elif self.candidate_locations:
            if self.number_chargers > len(self.candidate_locations):
                raise ValueError(
                    f"Requested {self.number_chargers} chargers but only {len(self.candidate_locations)} candidate locations exist."
                )
            self.chosen_charger_locations = random.sample(
                self.candidate_locations, k=self.number_chargers
            )
        else:
            self.chosen_charger_locations = []

        for charger_id in range(self.number_chargers):
            loc = None
            if self.chosen_charger_locations:
                loc = self.chosen_charger_locations[charger_id]
            self.chargers.append(Charger(self.env, charger_id, location=loc))

        # Generate cars, start the charging process and add them to the list of cars
        for car_id in range(self.number_cars):
            car = Car(self)
            self.env.process(car.charge(self.env, f"Car {car_id}"))
            self.cars.append(car)

        # The generate function needs to yield a timeout, otherwise it's not valid
        # This line basically does nothing
        yield self.env.timeout(0)


class Charger(simpy.Resource):
    """
    Charger: Simpy Resource: provides a service (charging), can be occupied by cars
    Parameters: environment, charger id, capacity (= 5 because only 1 car can charge at each charger)
    """

    # Initialize the charger
    def __init__(
        self, env, charger_id, capacity=5, location: CandidateLocation | None = None
    ):
        super().__init__(env, capacity)
        self.charger_id = charger_id
        self.location = location
        # Initialize chargingTime: total time a car charged at this charger
        self.chargingTime = 0

    def __str__(self):
        """
        Change the string represenation of charger so we can easily print chargers.
        """
        if self.location:
            return f"Charger {self.charger_id} (fid={self.location.fid})"
        return f"Charger {self.charger_id}"


class Car:
    """
    Car: object that arrives, looks for the best available charger, charges, and then leaves.
    Uses external variables: minimal charging time, maximal charging time, simulation time, number of chargers
    Parameters: source object (for accessing the list of chargers)
    """

    def __init__(self, src):
        # Change the generation of arrival times and destinations to distributions based on real data
        self.src = src
        # Randomly generate how long it takes to charge
        self.chargeTime = random.randint(
            src.config["min_charge_time"], src.config["max_charge_time"]
        )
        # Randomly choose an arrival time
        self.arrivalTime = random.randint(0, src.config["simulation_time"])
        # Destination is now a real centroid point (x,y). We sample it from the candidate locations.
        # (Assumption for now: trips start/end within the same candidate set.)
        if not src.candidate_locations:
            raise ValueError(
                "No candidate locations loaded. Cannot pick a geographic destination."
            )
        if src.destination_weights:
            self.destination = random.choices(
                src.candidate_locations, weights=src.destination_weights, k=1
            )[0]
        else:
            self.destination = random.choice(src.candidate_locations)
        self.waitingTime = None
        self.totalWaitingTime = 0.0
        self.walkingDist = None
        self.status = "created"
        # Create a dictionary of the closest chargers, charger as keys and walking distance as values
        # Sorted, so the closest charger is the first item
        self.closestChargers = self.find_closest_chargers(src)
        # Get the first item from closestChargers
        self.chosenCharger = list(self.closestChargers.keys())[0]

    def find_closest_chargers(self, src):
        """
        Creates a dictionary of the closest chargers to the chosen destination.
        Keys are chargers, values are walking distances from destination to charger
        Parameters: source object to access the list of chargers
        Returns: dictionary of sorted closest chargers with walking distances
        """
        charger_dict = {}
        # Save walking distances for each charger in dictionary
        for charger in src.chargers:
            charger_dict[charger] = self.calculate_walk_dist(charger)
        # Sort the dictionary by walking distances
        sorted_charger_dict = {
            k: v for k, v in sorted(charger_dict.items(), key=lambda item: item[1])
        }
        return sorted_charger_dict

    def charge(self, env, name):
        """
        Arrive, then check if best charger is available. If not, loop to find next best charger and try that one.
        Parameters: environment, source object and name of the car (for printing)
        Returns: None
        """
        # Arrive at self.arrivaltime
        yield env.timeout(self.arrivalTime)
        self.src.log("arrived", name, "arrived", destination_fid=self.destination.fid)

        # Loop: keep looking for an available charger.
        while True:
            # Make request for charger
            charger = self.chosenCharger
            req = charger.request()
            # Wait up to max_wait_time for this charger before trying the next one.
            wait_start = env.now
            results = yield req | env.timeout(self.src.config.get("max_wait_time", 5))
            # Check if request went through
            if req in results:
                self.totalWaitingTime += env.now - wait_start
                self.waitingTime = self.totalWaitingTime
                self.finalCharger = charger
                self.src.log(
                    "start_charge",
                    name,
                    f"starting to charge at {charger}",
                    charger_id=charger.charger_id,
                    charger_fid=(charger.location.fid if charger.location else None),
                    waited=float(self.waitingTime),
                )
                # Find walkingDist to chosen charger
                self.walkingDist = self.closestChargers[self.chosenCharger]

                # Charge and add charging time to charger's total charging time
                yield env.timeout(self.chargeTime)
                charger.chargingTime += self.chargeTime
                # Release the request because it is done, and end the charge function
                charger.release(req)
                self.status = "charged"
                self.src.log(
                    "done",
                    name,
                    "done charging",
                    charger_id=charger.charger_id,
                    charger_fid=(charger.location.fid if charger.location else None),
                    charge_time=float(self.chargeTime),
                    walking_dist_m=float(self.walkingDist),
                )
                return

            else:
                # Cancel the request for the current (unavailable) charger
                req.cancel()
                self.totalWaitingTime += env.now - wait_start
                # Save the unavailable charger
                lastCharger = self.chosenCharger
                # Find the next best charger
                nextCharger = self.find_next_best_charger(self.chosenCharger)
                if nextCharger is None:
                    self.status = "gave_up"
                    self.waitingTime = None
                    self.walkingDist = None
                    self.src.log(
                        "gave_up",
                        name,
                        "no alternate charger available, gave up",
                        last_charger_id=lastCharger.charger_id,
                    )
                    return
                # Save the next charger as chosen charger
                self.chosenCharger = nextCharger
                self.src.log(
                    "switch",
                    name,
                    f"{lastCharger} not available, trying {self.chosenCharger}",
                    from_charger_id=lastCharger.charger_id,
                    to_charger_id=self.chosenCharger.charger_id,
                )
                # Travel to next charger using calculated travel time
                yield env.timeout(
                    self.charger_travel_time(lastCharger, self.chosenCharger)
                )

    def find_next_best_charger(self, last_charger):
        """
        Finds the next closest charger based on the list of closest chargers (sorted by distance from destination)
        Parameters: source object, last chosen charger
        (temporary) returns: index of the next charger in the list
        """
        # Create a list of the closest chargers sorted by distance
        chargers_list = list(self.closestChargers.keys())
        # Find and return the next charger in the list
        last_idx = chargers_list.index(last_charger)
        next_idx = last_idx + 1
        if next_idx >= len(chargers_list):
            return None
        return chargers_list[next_idx]

    def calculate_walk_dist(self, charger):
        """
        Calculates the walking distance from the destination to the charger.
        Parameters: charger to calculate distance to
        (temporary) returns: absolute difference between charger's index and destination
        """
        # Walking distance in meters
        if charger.location is None:
            # Fallback to old behavior if no locations
            charger_idx = self.src.chargers.index(charger)
            return abs(charger_idx - 0)
        # if (
        #     self.src.config.get("distance_mode", "network") == "network"
        #     and self.src.walking_network

        # ):
        #     destination_lat, destination_lon = self.destination.lat_lon()
        #     charger_lat, charger_lon = charger.location.lat_lon()
        #     return self.src.walking_network.distance_m(
        #         destination_lat,
        #         destination_lon,
        #         charger_lat,
        #         charger_lon,
        #     )
        dx = charger.location.x - self.destination.x
        dy = charger.location.y - self.destination.y
        return (dx * dx + dy * dy) ** 0.5

    def charger_travel_time(self, charger1, charger2):
        """
        Calculates travel time from one charger to another
        Parameters: charger1's index, charger2's index
        (temporary) returns: difference between charger indexes
        """
        # Travel time between chargers is approximated from euclidean distance.
        # Units: 1 time unit == 1 minute, walking_speed_m_per_min controls conversion.
        if charger1.location is None or charger2.location is None:
            charger1_idx = self.src.chargers.index(charger1)
            charger2_idx = self.src.chargers.index(charger2)
            return abs(charger1_idx - charger2_idx)
        # if (
        #     self.src.config.get("distance_mode", "network") == "network"
        #     and self.src.walking_network
        # ):
        #     charger1_lat, charger1_lon = charger1.location.lat_lon()
        #     charger2_lat, charger2_lon = charger2.location.lat_lon()
        #     dist_m = self.src.walking_network.distance_m(
        #         charger1_lat,
        #         charger1_lon,
        #         charger2_lat,
        #         charger2_lon,
        #     )
        #     return dist_m / self.src.config.get("walking_speed_m_per_min", 83.3)
        dx = charger2.location.x - charger1.location.x
        dy = charger2.location.y - charger1.location.y
        dist_m = (dx * dx + dy * dy) ** 0.5
        return dist_m / self.src.config.get("walking_speed_m_per_min", 83.3)


# -----------------------------
# Charger-location strategies
# -----------------------------


def select_spread_out_locations(
    locations: list[CandidateLocation], number_chargers: int
) -> list[CandidateLocation]:
    """Greedy spread: start near the center, then repeatedly pick the farthest candidate."""
    if number_chargers >= len(locations):
        return list(locations)

    center_x = sum(loc.x for loc in locations) / len(locations)
    center_y = sum(loc.y for loc in locations) / len(locations)
    chosen = [
        min(
            locations,
            key=lambda loc: (loc.x - center_x) ** 2 + (loc.y - center_y) ** 2,
        )
    ]

    while len(chosen) < number_chargers:
        remaining = [loc for loc in locations if loc not in chosen]
        next_location = max(
            remaining,
            key=lambda loc: min(
                (loc.x - selected.x) ** 2 + (loc.y - selected.y) ** 2
                for selected in chosen
            ),
        )
        chosen.append(next_location)

    return chosen


def parse_location_fids(raw_fids) -> list[int]:
    if raw_fids is None:
        return []
    if isinstance(raw_fids, str):
        parts = raw_fids.replace(",", ";").split(";")
        return [int(float(part.strip())) for part in parts if part.strip()]
    return [int(float(fid)) for fid in raw_fids]


def select_fixed_locations(
    locations: list[CandidateLocation], fixed_fids
) -> list[CandidateLocation]:
    fids = parse_location_fids(fixed_fids)
    by_fid = {location.fid: location for location in locations}
    missing = [fid for fid in fids if fid not in by_fid]
    if missing:
        raise ValueError(f"Fixed charger FIDs not found in candidate locations: {missing}")
    return [by_fid[fid] for fid in fids]

#decides where the chargers will be placed before the simulation starts.
def select_charger_locations(
    strategy: str,
    number_chargers: int,
    candidate_locations: list[CandidateLocation],
    destination_weights: list[float] | None,
    existing_charger_locations: list[CandidateLocation],
    config: dict | None = None,
    walking_network: WalkingNetwork | None = None,
) -> list[CandidateLocation]:
    config = config or {}
    fixed_charger_fids = config.get("fixed_charger_fids")
    if fixed_charger_fids is not None:
        return select_fixed_locations(
            existing_charger_locations + candidate_locations, fixed_charger_fids
        )
    if strategy == "fixed":
        raise ValueError("Use fixed_charger_fids when charger_strategy is 'fixed'.")
    if strategy == "existing":
        return existing_charger_locations[:number_chargers]

    # choose locations with highest demand weight ?
    if strategy == "demand_hotspots" and destination_weights:
        weighted = sorted(
            zip(candidate_locations, destination_weights),
            key=lambda item: item[1],
            reverse=True,
        )
        return [location for location, _ in weighted[:number_chargers]]
    # Ramdom location
    if strategy == "spread_out":
        return select_spread_out_locations(candidate_locations, number_chargers)

    if strategy in {"optimized", "optimized_minimum"}:
        raise ValueError(
            "Run optimization outside simulation.py, then pass the result with "
            "fixed_charger_fids."
        )

    # Fixed locations for the existing infrastructure
    fixed_locations = [
        loc for loc in candidate_locations
        if loc.fid in FIXED_EXISTING_FIDS
    ]

    if number_chargers <= len(fixed_locations):
        return fixed_locations[:number_chargers]

    remaining_locations = [
        loc for loc in candidate_locations
        if loc.fid not in FIXED_EXISTING_FIDS
    ]

    additional_needed = number_chargers - len(fixed_locations)

    random_locations = random.sample(
        remaining_locations,
        k=additional_needed,
    )

    return fixed_locations + random_locations


# -----------------------------
# Output helpers
# -----------------------------


def write_events(events_dir: Path, scenario_name: str, events: list[dict]) -> Path:
    events_path = events_dir / f"simulation_events_{scenario_name}.csv"
    if not events:
        events_path.write_text("", encoding="utf-8")
        return events_path

    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({k for event in events for k in event.keys()})
        )
        writer.writeheader()
        writer.writerows(events)
    return events_path

def write_summary(events_dir: Path, results: list[dict]) -> Path:
    summary_path = events_dir / "simulation_scenario_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return summary_path

# -----------------------------
# Simulation runner
# -----------------------------


def run_simulation(
    scenario_name: str,
    config: dict,
    candidate_locations: list[CandidateLocation],
    destination_weights: list[float] | None,
    existing_charger_locations: list[CandidateLocation],
    walking_network: WalkingNetwork,
    events_dir: Path,
) -> dict:
    simulation_time = config["simulation_time"]
    num_cars = config["num_cars"]
    num_chargers = config["num_chargers"]
    random.seed(config["seed"])

    charger_locations = select_charger_locations(
        config["charger_strategy"],
        num_chargers,
        candidate_locations,
        destination_weights,
        existing_charger_locations,
        config,
        walking_network,
    )

    env = simpy.Environment()
    src = Source(
        env,
        config,
        num_cars,
        len(charger_locations),
        candidate_locations=candidate_locations,
        charger_locations=charger_locations,
        destination_weights=destination_weights,
        walking_network=walking_network,
        verbose=config.get("verbose", False),
    )
    env.run(until=simulation_time)

    completed_cars = [
        car for car in src.cars if getattr(car, "status", None) == "charged"
    ]
    gave_up_cars = [
        car for car in src.cars if getattr(car, "status", None) == "gave_up"
    ]
    total_walkdist = sum(car.walkingDist for car in completed_cars)
    total_waiting = sum(car.waitingTime for car in completed_cars)
    total_utilization = sum(
        charger.chargingTime / (simulation_time * charger.capacity)
        for charger in src.chargers
    )

    events_path = write_events(events_dir, scenario_name, src.events)
    charger_fids = ";".join(str(location.fid) for location in charger_locations)

    return {
        "scenario": scenario_name,
        "description": config["description"],
        "charger_strategy": config["charger_strategy"],
        "num_cars": num_cars,
        "num_chargers": len(charger_locations),
        "max_wait_time": config.get("max_wait_time", 5),
        "walking_threshold_m": config["walking_threshold_m"],
        "completed_charging": len(completed_cars),
        "gave_up": len(gave_up_cars),
        "completed_pct": len(completed_cars) / num_cars * 100,
        "gave_up_pct": len(gave_up_cars) / num_cars * 100,
        "avg_waiting_time": (
            total_waiting / len(completed_cars) if completed_cars else 0
        ),
        "avg_walking_dist_m": (
            total_walkdist / len(completed_cars) if completed_cars else 0
        ),
        "avg_charger_utilization": (
            total_utilization / len(src.chargers) if src.chargers else 0
        ),
        "charger_fids": charger_fids
    }


def print_summary(results: list[dict]) -> None:
    print("\nSimulation result")
    print(
        f"{'scenario':<24} {'completed %':>11} {'gave up %':>9} "
        f"{'avg wait':>9} {'avg walk m':>11} {'utilization':>11}"
    )
    for result in results:
        print(
            f"{result['scenario']:<24} "
            f"{result['completed_pct']:>11.1f} "
            f"{result['gave_up_pct']:>9.1f} "
            f"{result['avg_waiting_time']:>9.2f} "
            f"{result['avg_walking_dist_m']:>11.1f} "
            f"{result['avg_charger_utilization']:>11.2f}"
        )


def main() -> None:
    candidate_locations = load_candidate_locations(CANDIDATE_LOCATIONS_PATH)
    existing_charger_locations = load_existing_charger_locations(EXISTING_CHARGERS_PATH)
    destination_weights = load_blended_heatmap_weights(
        candidate_locations, HEATMAP_DENSITY_PATH, EV_DEMAND_HEATMAP_PATH
    )
    walking_network = WalkingNetwork.from_geojson(WALKING_NETWORK_PATH)
    print(
        f"Loaded walking network: {walking_network.trace_count} traces, "
        f"{len(walking_network.network_nodes)} connected nodes"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []
    for scenario_name, config in SCENARIOS.items():
        print(f"Running scenario: {scenario_name}")
        results.append(
            run_simulation(
                scenario_name,
                config,
                candidate_locations,
                destination_weights,
                existing_charger_locations,
                walking_network,
                OUTPUT_DIR,
            )
        )

    summary_path = write_summary(OUTPUT_DIR, results)
    print_summary(results)
    print(f"\nWrote scenario summary: {summary_path}")


if __name__ == "__main__":
    main()
