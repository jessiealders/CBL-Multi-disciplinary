import simpy
import random
import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CandidateLocation:
    fid: int
    identificatie: str
    x: float
    y: float
    max_area: float
    postcode: str | None = None


# Load the candidate locations from a CSV file.
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


def load_heatmap_weights(
    locations: list["CandidateLocation"], density_path: Path
) -> list[float] | None:
    if not density_path.exists():
        print(
            f"Heatmap density file not found: {density_path}. Using uniform destination weights."
        )
        return None
    data = np.load(density_path)
    counts = data["counts"]  # shape (bins_y, bins_x), EPSG:3857
    xmin, xmax = float(data["xmin"]), float(data["xmax"])
    ymin, ymax = float(data["ymin"]), float(data["ymax"])
    bins_y, bins_x = counts.shape

    to_3857 = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)

    weights: list[float] = []
    for loc in locations:
        x3857, y3857 = to_3857.transform(loc.x, loc.y)
        ix = int((x3857 - xmin) / (xmax - xmin) * bins_x)
        iy = int((y3857 - ymin) / (ymax - ymin) * bins_y)
        ix = max(0, min(ix, bins_x - 1))
        iy = max(0, min(iy, bins_y - 1))
        # +1 so every location retains at least a baseline probability
        weights.append(float(counts[iy, ix]) + 1.0)
    return weights


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
        number_cars,
        number_chargers,
        candidate_locations=None,
        destination_weights=None,
        verbose=False,
    ):
        self.env = env
        self.number_cars = number_cars
        self.number_chargers = number_chargers
        self.chargers = []
        self.cars = []
        self.verbose = verbose
        self.events: list[dict] = []
        self.candidate_locations = candidate_locations or []
        self.destination_weights = destination_weights
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
        if self.candidate_locations:
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
            env.process(car.charge(self.env, f"Car {car_id}"))
            self.cars.append(car)

        # The generate function needs to yield a timeout, otherwise it's not valid
        # This line basically does nothing
        yield self.env.timeout(0)


class Charger(simpy.Resource):
    """
    Charger: Simpy Resource: provides a service (charging), can be occupied by cars
    Parameters: environment, charger id, capacity (= 1 because only 1 car can charge at each charger)
    """

    # Initialize the charger
    def __init__(
        self, env, charger_id, capacity=1, location: CandidateLocation | None = None
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
        self.chargeTime = random.randint(min_charge_time, max_charge_time)
        # Randomly choose an arrival time
        self.arrivalTime = random.randint(0, simulation_time)
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
        self.walkingDist = None
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

        # Loop: keep looking for a charger that's available within
        while True:
            # Make request for charger
            charger = self.chosenCharger
            req = charger.request()
            # results = request if it went through, otherwise wait 1 [time unit]
            results = yield req | env.timeout(1)
            # Check if request went through
            if req in results:
                # Calculate waiting time
                self.waitingTime = env.now - self.arrivalTime
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
                # Save the unavailable charger
                lastCharger = self.chosenCharger
                # Find the next best charger
                nextCharger = self.find_next_best_charger(self.chosenCharger)
                # Check if the next best charger is within the walking threshold to the destination
                # Remember self.closestChargers is a dict with charger as key and walkingDist as value
                if self.closestChargers[nextCharger] < walking_threshold_m:
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
                else:
                    self.status = "gave_up"
                    self.src.log(
                        "gave_up",
                        name,
                        f"{nextCharger} too far, gave up",
                        last_charger_id=lastCharger.charger_id,
                        next_charger_id=nextCharger.charger_id,
                        next_dist_m=float(self.closestChargers[nextCharger]),
                    )
                    return

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
        next_charger = chargers_list[last_idx + 1]
        return next_charger

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
        dx = charger2.location.x - charger1.location.x
        dy = charger2.location.y - charger1.location.y
        dist_m = (dx * dx + dy * dy) ** 0.5
        return dist_m / walking_speed_m_per_min


# Initialize the constants of the simulation
simulation_time = 200
min_charge_time = 1
max_charge_time = 30
num_cars = 40
num_chargers = 7
walking_threshold_m = 300  # max walking distance (meters) from destination
walking_speed_m_per_min = 83.3  # ~5 km/h
verbose = False  # set True for per-car event logs
# random.seed(0) # Leave it commented for randomness.
env = simpy.Environment()
# Load candidate locations (centroids) for Strijp-S free placement
candidate_locations = load_candidate_locations(
    ROOT / "other data" / "reepacement_lessdata_strijp_lili.csv"
)
# Weight destinations by GPX movement density.
destination_weights = load_heatmap_weights(
    candidate_locations, ROOT / "other data" / "gpx_heatmap_density.npz"
)
# Create the source object to generate the cars and chargers
src = Source(
    env,
    num_cars,
    num_chargers,
    candidate_locations=candidate_locations,
    destination_weights=destination_weights,
    verbose=verbose,
)

# Choosing random charging ports
if src.chosen_charger_locations:
    chosen = src.chosen_charger_locations
    chosen_str = ", ".join(
        f"fid={c.fid} ({c.identificatie or 'n/a'}{', ' + c.postcode if c.postcode else ''})"
        for c in chosen
    )
    print(f"Chosen charger candidate locations (n={len(chosen)}): {chosen_str}")
# Run the simulation until the given time
env.run(until=simulation_time)

# Calculate the average waiting times, average walking distance, and number of cars that did and didn't get to charge
total_waiting = 0
didnt_charge = 0
total_walkdist = 0
for car in src.cars:
    # If the waiting time is None, the car didn't charge
    if car.waitingTime == None:
        didnt_charge += 1
    else:
        total_waiting += car.waitingTime
        total_walkdist += car.walkingDist

cars_charged = num_cars - didnt_charge
avg_waiting = total_waiting / (cars_charged)
avg_walkdist = total_walkdist / (cars_charged)
perc_didnt_charge = didnt_charge / num_cars * 100
perc_charged = cars_charged / num_cars * 100


# Print car metrics
print(f"""Metrics:
% of cars that didnt charge: {perc_didnt_charge}%,
% of cars that charged: {perc_charged}%,
Average waiting time of cars that charged: {avg_waiting},
Average walking dist of cars that charged: {avg_walkdist}""")

# Write structured event log to csv
events_path = ROOT / "other data" / "simulation_events.csv"
with events_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=sorted({k for e in src.events for k in e.keys()})
    )
    writer.writeheader()
    writer.writerows(src.events)
print(f"Wrote events log: {events_path}")

# Calculate and print charger utilizations
for charger in src.chargers:
    utilization = charger.chargingTime / (simulation_time * charger.capacity)
    print(f"{charger} utilization: {utilization}")
