import simpy
import random
import csv
from pathlib import Path
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]


def p(rel_windows_path: str) -> Path:
    """Windows to POSIX path conversion."""
    return ROOT.joinpath(*rel_windows_path.split("\\"))


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


class Source:
    '''
    Source: works as simulation generator.
    Generates a given number of cars and chargers.
    Stores cars and chargers in list so we can easily access them throughout the simulation.
    Parameters: environment, number of cars, number of chargers to generate
    '''

    def __init__(self, env, number_cars, number_chargers, candidate_locations=None, verbose=False):
        self.env = env
        self.number_cars = number_cars
        self.number_chargers = number_chargers
        self.chargers = []
        self.cars = []
        self.verbose = verbose
        self.events: list[dict] = []
        self.candidate_locations = candidate_locations or []
        self.chosen_charger_locations: list[CandidateLocation] = []
        self.action = env.process(self.generate())

    def log(self, kind: str, car_name: str, msg: str, **payload):
        """Store a structured event; optionally print it."""
        row = {"t": float(self.env.now), "kind": kind, "car": car_name, "msg": msg, **payload}
        self.events.append(row)
        if self.verbose:
            print(f"{self.env.now:.2f} {car_name} {kind}: {msg}")

    def generate(self):
        '''
        Generates number of chargers and cars based on the given numbers,
        And stores these in lists self.chargers and self.cars so we can access them.
        Returns: None
        '''
        # Generate chargers and add them to the list of chargers.
        # If we have candidate locations, pick unique locations at random (no repeats).
        if self.candidate_locations:
            if self.number_chargers > len(self.candidate_locations):
                raise ValueError(
                    f"Requested {self.number_chargers} chargers but only {len(self.candidate_locations)} candidate locations exist."
                )
            self.chosen_charger_locations = random.sample(self.candidate_locations, k=self.number_chargers)
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
            self.env.process(car.charge(self.env, f'Car {car_id}'))
            self.cars.append(car)

        # The generate function needs to yield a timeout, otherwise it's not valid
        # This line basically does nothing
        yield self.env.timeout(0)

class Charger(simpy.Resource):
    '''
    Charger: Simpy Resource: provides a service (charging), can be occupied by cars
    Parameters: environment, charger id, capacity (= 1 because only 1 car can charge at each charger)
    '''

    # Initialize the charger
    def __init__(self, env, charger_id, capacity=1, location: CandidateLocation | None = None):
        super().__init__(env, capacity)
        self.charger_id = charger_id
        self.location = location
        # Initialize chargingTime: total time a car charged at this charger
        self.chargingTime = 0

    def __str__(self):
        '''
        Change the string represenation of charger so we can easily print chargers.
        '''
        if self.location:
            return f"Charger {self.charger_id} (fid={self.location.fid})"
        return f"Charger {self.charger_id}"


class Car:
    '''
    Car: object that arrives, looks for the best available charger, charges, and then leaves.
    Uses external variables: minimal charging time, maximal charging time, simulation time, number of chargers
    Parameters: source object (for accessing the list of chargers)
    '''

    def __init__(self,src):
        # Change the generation of arrival times and destinations to distributions based on real data
        self.src = src
        # Randomly generate how long it takes to charge
        self.chargeTime = random.randint(min_charge_time,max_charge_time)
        # Randomly choose an arrival time
        self.arrivalTime = random.randint(0,simulation_time)
        # Destination is now a real centroid point (x,y). We sample it from the candidate locations.
        # (Assumption for now: trips start/end within the same candidate set.)
        if not src.candidate_locations:
            raise ValueError("No candidate locations loaded. Cannot pick a geographic destination.")
        self.destination = random.choice(src.candidate_locations)
        self.waitingTime = None
        self.walkingDist = None
        self.status = "created"
        # Create a dictionary of the closest chargers, charger as keys and walking distance as values
        # Sorted, so the closest charger is the first item
        self.closestChargers = self.find_closest_chargers(src)
        # Get the first item from closestChargers
        self.chosenCharger = list(self.closestChargers.keys())[0]

    def find_closest_chargers(self, src):
        '''
        Creates a dictionary of the closest chargers to the chosen destination.
        Keys are chargers, values are walking distances from destination to charger
        Parameters: source object to access the list of chargers
        Returns: dictionary of sorted closest chargers with walking distances
        '''
        charger_dict = {}
        # Save walking distances for each charger in dictionary
        for charger in src.chargers:
            charger_dict[charger] = self.calculate_walk_dist(charger)
        # Sort the dictionary by walking distances
        sorted_charger_dict = {k: v for k, v in sorted(charger_dict.items(), key=lambda item: item[1])}
        return sorted_charger_dict

    def charge(self, env, name):
        '''
        Arrive, then check if best charger is available. If not, loop to find next best charger and try that one.
        Parameters: environment, source object and name of the car (for printing)
        Returns: None
        '''
        # Arrive at self.arrivaltime
        yield env.timeout(self.arrivalTime)
        self.src.log("arrived", name, "arrived", destination_fid=self.destination.fid)

        # Perfect-information policy:
        # - choose the closest currently-free charger within walking threshold
        # - if none are free, wait and retry
        # - give up after max_wait_min
        retry_every_min = 1
        max_wait_min = 5
        deadline = self.arrivalTime + max_wait_min

        while True:
            available = [
                c
                for c, dist in self.closestChargers.items()
                if dist < walking_threshold_m and c.count < c.capacity
            ]

            if not available:
                if env.now >= deadline:
                    self.status = "gave_up"
                    self.waitingTime = None
                    self.walkingDist = None
                    self.src.log(
                        "gave_up",
                        name,
                        f"gave up after waiting {max_wait_min} minutes for a free charger",
                        max_wait_min=float(max_wait_min),
                    )
                    return

                if self.status != "waiting_for_free":
                    self.status = "waiting_for_free"
                    self.src.log(
                        "wait_no_free",
                        name,
                        "no charger free within walking threshold; waiting",
                    )
                yield env.timeout(retry_every_min)
                continue

            chosen = min(available, key=lambda c: self.closestChargers[c])
            self.chosenCharger = chosen

            req = chosen.request()
            yield req

            self.waitingTime = env.now - self.arrivalTime
            self.finalCharger = chosen
            self.walkingDist = self.closestChargers[chosen]
            self.status = "charging"
            self.src.log(
                "start_charge",
                name,
                f"starting to charge at {chosen}",
                charger_id=chosen.charger_id,
                charger_fid=(chosen.location.fid if chosen.location else None),
                waited=float(self.waitingTime),
                walking_dist_m=float(self.walkingDist),
            )

            yield env.timeout(self.chargeTime)
            chosen.chargingTime += self.chargeTime
            chosen.release(req)
            self.status = "charged"
            self.src.log(
                "done",
                name,
                "done charging",
                charger_id=chosen.charger_id,
                charger_fid=(chosen.location.fid if chosen.location else None),
                charge_time=float(self.chargeTime),
                walking_dist_m=float(self.walkingDist),
            )
            return

    def calculate_walk_dist(self, charger):
        '''
        Calculates the walking distance from the destination to the charger.
        Parameters: charger to calculate distance to
        (temporary) returns: absolute difference between charger's index and destination
        '''
        # Walking distance in meters
        if charger.location is None:
            # Fallback to old behavior if no locations
            charger_idx = self.src.chargers.index(charger)
            return abs(charger_idx - 0)
        dx = charger.location.x - self.destination.x
        dy = charger.location.y - self.destination.y
        return (dx * dx + dy * dy) ** 0.5

    def charger_travel_time(self, charger1, charger2):
        '''
        Calculates travel time from one charger to another
        Parameters: charger1's index, charger2's index
        (temporary) returns: difference between charger indexes
        '''
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


def run_simulation(
    *,
    candidate_locations: list[CandidateLocation],
    num_chargers: int,
    arrival_times: list[float],
    simulation_time: int,
    min_charge_time: int,
    max_charge_time: int,
    walking_threshold_m: float,
    walking_speed_m_per_min: float,
    seed: int | None = None,
    verbose: bool = False,
):
    """Run one simulation with explicit arrival times.

    Returns:
      (src, chosen_locations)
    where src is the Source instance (contains events/cars/chargers), and chosen_locations
    is the list of CandidateLocation objects used for charger placement.
    """
    # Set globals expected by Car.__init__/Car.charger_travel_time.
    globals()["simulation_time"] = simulation_time
    globals()["min_charge_time"] = min_charge_time
    globals()["max_charge_time"] = max_charge_time
    globals()["walking_threshold_m"] = walking_threshold_m
    globals()["walking_speed_m_per_min"] = walking_speed_m_per_min

    if seed is not None:
        random.seed(seed)

    env = simpy.Environment()
    src = Source(env, len(arrival_times), num_chargers, candidate_locations=candidate_locations, verbose=verbose)

    # Override the randomly generated arrivalTime per Car with provided arrival_times.
    for car, at in zip(src.cars, arrival_times):
        car.arrivalTime = float(at)

    env.run(until=simulation_time)
    return src, src.chosen_charger_locations

if __name__ == "__main__":
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

    # Load candidate locations (centroids) for Strijp-S free placement
    candidate_locations = load_candidate_locations(
        p(r"other data\freepacement_lessdata_strijp_lili.csv")
    )

    # Use the original behavior (random arrival times) for interactive single runs.
    env = simpy.Environment()
    src = Source(env, num_cars, num_chargers, candidate_locations=candidate_locations, verbose=verbose)

    if src.chosen_charger_locations:
        chosen = src.chosen_charger_locations
        chosen_str = ", ".join(
            f"fid={c.fid} ({c.identificatie or 'n/a'}{', ' + c.postcode if c.postcode else ''})" for c in chosen
        )
        print(f"Chosen charger candidate locations (n={len(chosen)}): {chosen_str}")

    env.run(until=simulation_time)

    total_waiting = 0
    didnt_charge = 0
    total_walkdist = 0
    for car in src.cars:
        if car.waitingTime == None:
            didnt_charge += 1
        else:
            total_waiting += car.waitingTime
            total_walkdist += car.walkingDist

    cars_charged = num_cars - didnt_charge
    avg_waiting = total_waiting / (cars_charged) if cars_charged else 0
    avg_walkdist = total_walkdist / (cars_charged) if cars_charged else 0
    perc_didnt_charge = didnt_charge / num_cars * 100 if num_cars else 0
    perc_charged = cars_charged / num_cars * 100 if num_cars else 0

    print(
        f"""Metrics:
% of cars that didnt charge: {perc_didnt_charge}%,
% of cars that charged: {perc_charged}%,
Average waiting time of cars that charged: {avg_waiting},
Average walking dist of cars that charged: {avg_walkdist}"""
    )

    events_path = ROOT / "other data" / "simulation_events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for e in src.events for k in e.keys()}))
        writer.writeheader()
        writer.writerows(src.events)
    print(f"Wrote events log: {events_path}")

    for charger in src.chargers:
        utilization = charger.chargingTime / (simulation_time * charger.capacity)
        print(f"{charger} utilization: {utilization}")
