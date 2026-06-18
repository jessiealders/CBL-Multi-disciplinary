import simpy
import random
import csv
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
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
OUTPUT_DIR = ROOT / "output" / "baseline"

RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
RD_TO_WEB_MERCATOR = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)

FIXED_EXISTING_FIDS = {308440, 308702, 307904, 307905, 308108}
BASELINE_EV_ARRIVALS = 159
DEFAULT_SIMULATION_TIME = 24 * 60
DEFAULT_MIN_CHARGE_TIME = 45
DEFAULT_MAX_CHARGE_TIME = 120
DEFAULT_MAX_WAIT_TIME = 5
DEFAULT_WALKING_THRESHOLD_M = 300.0
DEFAULT_WALKING_SPEED_M_PER_MIN = 83.3
DEFAULT_CHARGER_CONNECTORS = 2
USE_WALKING_NETWORK_DISTANCE = True

BASE_ARRIVALS_PER_HOUR = [
    5, 5, 5, 5, 5, 10,      # 00-05
    25, 39, 39, 29,         # 06-09
    20, 20,                 # 10-11
    29, 29,                 # 12-13
    20, 20, 25,             # 14-16
    44, 44, 34,             # 17-19
    20, 15, 10, 5,          # 20-23
]


def selected_distance_mode() -> str:
    """Return the distance mode used by this simulation run."""
    if USE_WALKING_NETWORK_DISTANCE:
        return "network"
    return "euclidean"


def generate_arrival_times(rng: random.Random, arrivals_per_hour: list[int]) -> list[float]:
    """Return arrival times in minutes in [0, 1440)."""
    if len(arrivals_per_hour) != 24:
        raise ValueError("arrivals_per_hour must have length 24")

    times: list[float] = []
    for h, n in enumerate(arrivals_per_hour):
        start = h * 60
        for _ in range(int(n)):
            times.append(start + rng.random() * 60)
    times.sort()
    return times


def scale_profile_to_total(profile: list[int], total_arrivals: int) -> list[int]:
    """Scale a 24-hour arrival profile to one total number of EVs."""
    if len(profile) != 24:
        raise ValueError("profile must have length 24")
    if total_arrivals < 0:
        raise ValueError("total_arrivals must be non-negative")

    profile_total = sum(profile)
    if profile_total <= 0:
        return [0 for _ in profile]

    raw = [value * total_arrivals / profile_total for value in profile]
    scaled = [int(value) for value in raw]
    remainder = total_arrivals - sum(scaled)
    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - scaled[index],
        reverse=True,
    )
    for index in order[:remainder]:
        scaled[index] += 1
    return scaled


def build_simulation_config(
    description: str,
    charger_strategy: str,
    num_chargers: int,
    arrival_times: list[float],
    seed: int,
    min_charge_time: int = DEFAULT_MIN_CHARGE_TIME,
    max_charge_time: int = DEFAULT_MAX_CHARGE_TIME,
    walking_threshold_m: float = DEFAULT_WALKING_THRESHOLD_M,
    walking_speed_m_per_min: float = DEFAULT_WALKING_SPEED_M_PER_MIN,
    fixed_charger_fids: str | None = None,
    charger_connectors: int = DEFAULT_CHARGER_CONNECTORS,
    simulation_time: int = DEFAULT_SIMULATION_TIME,
    max_wait_time: int = DEFAULT_MAX_WAIT_TIME,
    distance_mode: str | None = None,
    verbose: bool = False,
    write_events: bool = False,
) -> dict:
    config = {
        "description": description,
        "charger_strategy": charger_strategy,
        "num_cars": len(arrival_times),
        "num_chargers": num_chargers,
        "arrival_times": arrival_times,
        "simulation_time": simulation_time,
        "min_charge_time": min_charge_time,
        "max_charge_time": max_charge_time,
        "max_wait_time": max_wait_time,
        "walking_threshold_m": walking_threshold_m,
        "walking_speed_m_per_min": walking_speed_m_per_min,
        "charger_connectors": charger_connectors,
        "distance_mode": distance_mode or selected_distance_mode(),
        "seed": seed,
        "verbose": verbose,
        "write_events": write_events,
    }
    if fixed_charger_fids is not None:
        config["fixed_charger_fids"] = fixed_charger_fids
    return config

# Quick run as a small example
SCENARIOS = {
    "current_situation": {
        "description": "Current charging points in Strijp-S.",
        "charger_strategy": "existing",
        "num_cars": BASELINE_EV_ARRIVALS,
        "num_chargers": 5,
        "simulation_time": DEFAULT_SIMULATION_TIME,
        "min_charge_time": DEFAULT_MIN_CHARGE_TIME,
        "max_charge_time": DEFAULT_MAX_CHARGE_TIME,
        "max_wait_time": DEFAULT_MAX_WAIT_TIME,
        "walking_threshold_m": DEFAULT_WALKING_THRESHOLD_M,
        "walking_speed_m_per_min": DEFAULT_WALKING_SPEED_M_PER_MIN,
        "charger_connectors": DEFAULT_CHARGER_CONNECTORS,
        "distance_mode": selected_distance_mode(),
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
    connectors: int | None = None

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
            connectors = max(1, int(round(float(row.get("connectors") or 1))))
            locations.append(
                CandidateLocation(
                    fid=int(float(charger_id)),
                    identificatie=(row.get("geovisia_id") or "").strip(),
                    x=x,
                    y=y,
                    max_area=0.0,
                    postcode=(row.get("most_common_postcode") or "").strip() or None,
                    connectors=connectors,
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
    low_weight_floor: float = 0.07,
    low_weight_exponent: float = 2,
) -> list[float] | None:
    gpx_samples = _sample_density(locations, gpx_path)
    ev_samples = _sample_density(locations, ev_path)

    if gpx_samples is None and ev_samples is None:
        print("Both heatmap files missing. Using uniform destination weights.")
        return None

    def _normalise(arr: np.ndarray) -> np.ndarray:
        hi = arr.max()
        return arr / hi if hi > 0 else arr

    if gpx_samples is None:
        blended = _normalise(ev_samples)
    elif ev_samples is None:
        blended = _normalise(gpx_samples)
    else:
        blended = gpx_share * _normalise(gpx_samples) + ev_share * _normalise(
            ev_samples
        )

    penalized = low_weight_floor + np.power(blended, low_weight_exponent)
    return [float(v) for v in penalized]


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
        rng: random.Random,
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
        self.rng = rng
        self.chargers = []
        self.cars = []
        self.verbose = verbose
        self.events: list[dict] = []
        self.candidate_locations = candidate_locations or []
        self.fixed_charger_locations = charger_locations
        self.destination_weights = destination_weights
        self.walking_network = walking_network
        self.walking_distance_cache: dict[tuple[int, int], float] = {}
        self.chosen_charger_locations: list[CandidateLocation] = []
        self.action = env.process(self.generate())

    def walking_distance_m(
        self, start_location: CandidateLocation, end_location: CandidateLocation
    ) -> float:
        """Return cached real-network walking distance between two locations."""
        cache_key = tuple(sorted((start_location.fid, end_location.fid)))
        if cache_key not in self.walking_distance_cache:
            start_lat, start_lon = start_location.lat_lon()
            end_lat, end_lon = end_location.lat_lon()
            self.walking_distance_cache[cache_key] = self.walking_network.distance_m(
                start_lat,
                start_lon,
                end_lat,
                end_lon,
            )
        return self.walking_distance_cache[cache_key]

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
            self.chosen_charger_locations = self.rng.sample(
                self.candidate_locations, k=self.number_chargers
            )
        else:
            self.chosen_charger_locations = []

        for charger_id in range(self.number_chargers):
            loc = None
            if self.chosen_charger_locations:
                loc = self.chosen_charger_locations[charger_id]
            charger_connectors = int(
                getattr(loc, "connectors", None)
                or self.config.get("charger_connectors", 2)
            )
            self.chargers.append(
                Charger(
                    self.env,
                    charger_id,
                    connectors=charger_connectors,
                    location=loc,
                )
            )

        # Generate cars, start the charging process and add them to the list of cars
        for car_id in range(self.number_cars):
            car = Car(self, car_id)
            self.env.process(car.charge(self.env, f"Car {car_id}"))
            self.cars.append(car)

        # The generate function needs to yield a timeout, otherwise it's not valid
        # This line basically does nothing
        yield self.env.timeout(0)


class Charger(simpy.Resource):
    """
    Charger: Simpy Resource: provides a service (charging), can be occupied by cars
    Parameters: environment, charger id, and number of connectors.
    """

    # Initialize the charger
    def __init__(
        self, env, charger_id, connectors=2, location: CandidateLocation | None = None
    ):
        super().__init__(env, capacity=connectors)
        self.charger_id = charger_id
        self.location = location
        self.connectors = connectors
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

    def __init__(self, src: Source, car_id: int | None = None):
        # Change the generation of arrival times and destinations to distributions based on real data
        self.src = src
        # Randomly generate how long it takes to charge
        self.chargeTime = self.src.rng.randint(
            src.config["min_charge_time"], src.config["max_charge_time"]
        )
        arrival_times = src.config.get("arrival_times")
        if arrival_times is not None and car_id is not None:
            self.arrivalTime = arrival_times[car_id]
        else:
            self.arrivalTime = self.src.rng.randint(0, src.config["simulation_time"])
        # Destination is now a real centroid point (x,y). We sample it from the candidate locations.
        # (Assumption for now: trips start/end within the same candidate set.)
        if not src.candidate_locations:
            raise ValueError(
                "No candidate locations loaded. Cannot pick a geographic destination."
            )
        if src.destination_weights:
            self.destination = self.src.rng.choices(
                src.candidate_locations, weights=src.destination_weights, k=1
            )[0]
        else:
            self.destination = self.src.rng.choice(src.candidate_locations)
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
        if (
            self.src.config.get("distance_mode", "euclidean") == "network"
            and self.src.walking_network
        ):
            return self.src.walking_distance_m(self.destination, charger.location)
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
        if (
            self.src.config.get("distance_mode", "euclidean") == "network"
            and self.src.walking_network
        ):
            dist_m = self.src.walking_distance_m(
                charger1.location,
                charger2.location,
            )
            return dist_m / self.src.config.get(
                "walking_speed_m_per_min", DEFAULT_WALKING_SPEED_M_PER_MIN
            )
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
    rng: random.Random,
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

    random_locations = rng.sample(
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
    rng: random.Random,
) -> dict:
    simulation_time = config["simulation_time"]
    num_cars = config["num_cars"]
    num_chargers = config["num_chargers"]

    charger_locations = select_charger_locations(
        config["charger_strategy"],
        num_chargers,
        candidate_locations,
        destination_weights,
        existing_charger_locations,
        rng,
        config,
        walking_network,
    )

    env = simpy.Environment()
    src = Source(
        env,
        config,
        num_cars,
        len(charger_locations),
        rng,
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
    charger_utilizations = [
        charger.chargingTime / (simulation_time * charger.connectors)
        for charger in src.chargers
    ]
    total_utilization = sum(charger_utilizations)
    successfully_charged = len(completed_cars)
    abandoned_attempts = len(gave_up_cars)
    successfully_charged_pct = successfully_charged / num_cars * 100 if num_cars else 0
    abandoned_attempts_pct = abandoned_attempts / num_cars * 100 if num_cars else 0
    average_waiting_time = (
        total_waiting / successfully_charged if successfully_charged else 0
    )
    average_walking_distance = (
        total_walkdist / successfully_charged if successfully_charged else 0
    )
    average_charger_utilization = (
        total_utilization / len(src.chargers) if src.chargers else 0
    )
    maximum_charger_utilization = max(charger_utilizations) if charger_utilizations else 0

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
        "completed_charging": successfully_charged,
        "gave_up": abandoned_attempts,
        "completed_pct": successfully_charged_pct,
        "gave_up_pct": abandoned_attempts_pct,
        "avg_waiting_time": average_waiting_time,
        "avg_walking_dist_m": average_walking_distance,
        "avg_charger_utilization": average_charger_utilization,
        "max_charger_utilization": maximum_charger_utilization,
        "charger_fids": charger_fids
    }


def print_summary(results: list[dict]) -> None:
    print("\nSimulation result")
    print(
        f"{'scenario':<24} {'completed %':>11} {'gave up %':>9} "
        f"{'avg wait':>9} {'avg walk m':>11} {'avg util':>9} {'max util':>9}"
    )
    for result in results:
        print(
            f"{result['scenario']:<24} "
            f"{result['completed_pct']:>11.1f} "
            f"{result['gave_up_pct']:>9.1f} "
            f"{result['avg_waiting_time']:>9.2f} "
            f"{result['avg_walking_dist_m']:>11.1f} "
            f"{result['avg_charger_utilization']:>9.2f} "
            f"{result['max_charger_utilization']:>9.2f}"
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

    baseline_output_dir = ROOT / "output" / "baseline"
    baseline_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scenario_name, config in SCENARIOS.items():
        print(f"Using distance mode: {config['distance_mode']}")
        print(f"Running scenario: {scenario_name}")
        rng = random.Random()
        rng.seed(config["seed"])
        arrivals_per_hour = scale_profile_to_total(
            BASE_ARRIVALS_PER_HOUR,
            config["num_cars"],
        )
        arrival_times = generate_arrival_times(rng, arrivals_per_hour)
        run_config = build_simulation_config(
            description=config["description"],
            charger_strategy=config["charger_strategy"],
            num_chargers=config["num_chargers"],
            arrival_times=arrival_times,
            seed=config["seed"],
            min_charge_time=config["min_charge_time"],
            max_charge_time=config["max_charge_time"],
            walking_threshold_m=config["walking_threshold_m"],
            walking_speed_m_per_min=config["walking_speed_m_per_min"],
            charger_connectors=config["charger_connectors"],
            simulation_time=config["simulation_time"],
            max_wait_time=config["max_wait_time"],
            distance_mode=config["distance_mode"],
            verbose=config.get("verbose", False),
            write_events=config.get("write_events", False),
        )
        results.append(
            run_simulation(
                scenario_name,
                run_config,
                candidate_locations,
                destination_weights,
                existing_charger_locations,
                walking_network,
                baseline_output_dir,
                rng=rng,
            )
        )

    summary_path = write_summary(baseline_output_dir, results)
    print_summary(results)
    print(f"\nWrote scenario summary: {summary_path}")


if __name__ == "__main__":
    main()
