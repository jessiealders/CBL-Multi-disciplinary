import simpy
import random


class Source:
    '''
    Source: works as simulation generator.
    Generates a given number of cars and chargers.
    Stores cars and chargers in list so we can easily access them throughout the simulation.
    Parameters: environment, number of cars, number of chargers to generate
    '''
    
    def __init__(self, env, number_cars, number_chargers):
        self.env = env
        self.number_cars = number_cars
        self.number_chargers = number_chargers
        self.chargers = []
        self.cars = []
        self.action = env.process(self.generate())
        
    def generate(self):
        '''
        Generates number of chargers and cars based on the given numbers,
        And stores these in lists self.chargers and self.cars so we can access them.
        Returns: None
        '''
        # Generate chargers and add them to the list of chargers
        for charger_id in range(self.number_chargers):
            self.chargers.append(Charger(self.env, charger_id))
        
        # Generate cars, start the charging process and add them to the list of cars
        for car_id in range(self.number_cars):
            car = Car(self)
            env.process(car.charge(self.env, f'Car {car_id}'))
            self.cars.append(car)
        
        # The generate function needs to yield a timeout, otherwise it's not valid
        # This line basically does nothing
        yield self.env.timeout(0)

class Charger(simpy.Resource):
    '''
    Charger: Simpy Resource: provides a service (charging), can be occupied by cars
    Parameters: environment, charger id, capacity (= 1 because only 1 car can charge at each charger)
    '''
    
    def __init__(self, env, charger_id, capacity=1):
        super().__init__(env, capacity)
        self.charger_id = charger_id
        # Initialize chargingTime: total time a car charged at this charger 
        self.chargingTime = 0

    def __str__(self):
        '''
        Change the string represenation of charger so we can easily print chargers.
        '''
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
        # Randomly generate a destination (currently just an integer)
        self.destination = random.randint(0,num_chargers-1)
        self.waitingTime = None
        self.walkingDist = None
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
        print(f'{env.now} {name} arrived')
        
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
                print(
                    f'{env.now} {name} starting to charge at {charger}, waited {self.waitingTime}'
                )
                # Find walkingDist to chosen charger
                self.walkingDist = self.closestChargers[self.chosenCharger]
                
                # Charge and add charging time to charger's total charging time
                yield env.timeout(self.chargeTime)
                charger.chargingTime += self.chargeTime
                # Release the request because it is done, and end the charge function
                charger.release(req)
                print(f'{env.now} {name} is done charging')
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
                if self.closestChargers[nextCharger] < walking_threshold:
                    # Save the next charger as chosen charger
                    self.chosenCharger = nextCharger
                    print(f'{env.now} {name}: {lastCharger} not available, trying {self.chosenCharger}')
                    # Travel to next charger using calculated travel time
                    yield env.timeout(self.charger_travel_time(lastCharger, self.chosenCharger))
                else:
                    print(f'{env.now} {name}: {nextCharger} too far, gave up')
                    return
            
    
    def find_next_best_charger(self, last_charger):
        '''
        Finds the next closest charger based on the list of closest chargers (sorted by distance from destination)
        Parameters: source object, last chosen charger
        (temporary) returns: index of the next charger in the list
        '''
        # Create a list of the closest chargers sorted by distance
        chargers_list = list(self.closestChargers.keys())
        # Find and return the next charger in the list
        last_idx = chargers_list.index(last_charger)
        next_charger = chargers_list[last_idx + 1]
        return next_charger
    
    def calculate_walk_dist(self, charger):
        '''
        Calculates the walking distance from the destination to the charger.
        Parameters: charger to calculate distance to
        (temporary) returns: absolute difference between charger's index and destination
        '''
        charger_idx = self.src.chargers.index(charger)
        return abs(charger_idx - self.destination)
    
    def charger_travel_time(self, charger1, charger2):
        '''
        Calculates travel time from one charger to another
        Parameters: charger1's index, charger2's index
        (temporary) returns: difference between charger indexes
        '''
        charger1_idx = self.src.chargers.index(charger1)
        charger2_idx = self.src.chargers.index(charger2)
        return abs(charger1_idx-charger2_idx)

# Initialize the constants of the simulation
simulation_time = 200
min_charge_time = 1
max_charge_time = 30
num_cars = 40
num_chargers = 7
walking_threshold = 3
random.seed(0)
env = simpy.Environment()
# Create the source object to generate the cars and chargers
src = Source(env, num_cars, num_chargers)
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
avg_waiting = total_waiting/(cars_charged)
avg_walkdist = total_walkdist / (cars_charged)
perc_didnt_charge = didnt_charge / num_cars * 100
perc_charged = cars_charged / num_cars * 100


# Print car metrics
print(f'''Metrics:
% of cars that didnt charge: {perc_didnt_charge}%, 
% of cars that charged: {perc_charged}%, 
Average waiting time of cars that charged: {avg_waiting}, 
Average walking dist of cars that charged: {avg_walkdist}''')

# Calculate and print charger utilizations
for charger in src.chargers:
    utilization = charger.chargingTime / (simulation_time * charger.capacity)
    print(f'{charger} utilization: {utilization}')
