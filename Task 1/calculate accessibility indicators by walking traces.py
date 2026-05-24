import csv
import json
import math
import heapq
from pathlib import Path
from collections import defaultdict, deque


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "other data"
WALKING_TRACES_FILE = DATA_DIR / "walking_traces.geojson"
CHARGING_POINTS_FILE = DATA_DIR / "charging_points_strijp_s.csv"


DEMAND_POINTS = [
    {
        "name": "Klokgebouw",
        "type": "Event visitors",
        "lat": 51.448596,
        "lon": 5.456953,
    },
    {
        "name": "Microlab",
        "type": "Office workers",
        "lat": 51.444940859080994,
        "lon": 5.459615464599499,
    },
    {
        "name": "The Cohesion Lighthouse",
        "type": "Residential users",
        "lat": 51.44896391483022,
        "lon": 5.456209723741031,
    },
    {
        "name": "Urban Shopper Shopping Centre",
        "type": "Shopping / leisure visitors",
        "lat": 51.44762740232191,
        "lon": 5.455449308219853,
    },
    {
        "name": "SintLucas",
        "type": "Students, staff, and visitors",
        "lat": 51.44671018163512,
        "lon": 5.454531534582024,
    },
]


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def accessibility_category(distance_m):
    if distance_m <= 300:
        return "good"
    if distance_m <= 500:
        return "acceptable"
    return "poor / possible gap"


def coord_key(lon, lat):
    return f"{lon:.7f},{lat:.7f}"


def load_charging_points(path):
    chargers = []

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            chargers.append(
                {
                    "id": row["charger_id"],
                    "address": row["address"],
                    "provider": row["provider"],
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "connectors": float(row["connectors"]),
                }
            )

    return chargers


def build_walking_graph(path):
    geojson = json.loads(path.read_text(encoding="utf-8"))
    nodes = {}
    graph = defaultdict(list)

    for feature in geojson["features"]:
        coordinates = feature["geometry"]["coordinates"]

        for start, end in zip(coordinates, coordinates[1:]):
            start_key = coord_key(start[0], start[1])
            end_key = coord_key(end[0], end[1])

            nodes[start_key] = (start[1], start[0])
            nodes[end_key] = (end[1], end[0])

            length_m = haversine_m(start[1], start[0], end[1], end[0])
            graph[start_key].append((end_key, length_m))
            graph[end_key].append((start_key, length_m))

    return nodes, graph, len(geojson["features"])


def largest_connected_component(nodes, graph):
    seen = set()
    components = []

    for node in nodes:
        if node in seen:
            continue

        queue = deque([node])
        seen.add(node)
        component = {node}

        while queue:
            current = queue.popleft()
            for neighbor, _ in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)
    return components[0]


def nearest_network_node(lat, lon, nodes, network_nodes):
    nearest_node = None
    nearest_distance = float("inf")

    for node_id in network_nodes:
        node_lat, node_lon = nodes[node_id]
        distance = haversine_m(lat, lon, node_lat, node_lon)

        if distance < nearest_distance:
            nearest_node = node_id
            nearest_distance = distance

    return nearest_node, nearest_distance


def shortest_path_distances(start_node, graph, allowed_nodes):
    distances = {start_node: 0.0}
    heap = [(0.0, start_node)]

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_distance != distances[current_node]:
            continue

        for neighbor, edge_length in graph[current_node]:
            if neighbor not in allowed_nodes:
                continue

            new_distance = current_distance + edge_length
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

    return distances


def attach_chargers_to_network(chargers, nodes, network_nodes):
    attached_chargers = []

    for charger in chargers:
        network_node, snap_distance = nearest_network_node(
            charger["lat"],
            charger["lon"],
            nodes,
            network_nodes,
        )

        attached_chargers.append(
            {
                **charger,
                "network_node": network_node,
                "snap_to_network_m": snap_distance,
            }
        )

    return attached_chargers


def calculate_walking_kpi():
    nodes, graph, trace_count = build_walking_graph(WALKING_TRACES_FILE)
    allowed_nodes = largest_connected_component(nodes, graph)
    network_nodes = list(allowed_nodes)

    chargers = load_charging_points(CHARGING_POINTS_FILE)
    chargers = attach_chargers_to_network(chargers, nodes, network_nodes)

    results = []

    for demand in DEMAND_POINTS:
        demand_node, demand_snap = nearest_network_node(
            demand["lat"],
            demand["lon"],
            nodes,
            network_nodes,
        )

        distances = shortest_path_distances(demand_node, graph, allowed_nodes)
        nearest_result = None

        for charger in chargers:
            charger_node = charger["network_node"]
            if charger_node not in distances:
                continue

            walking_distance_m = (
                demand_snap
                + distances[charger_node]
                + charger["snap_to_network_m"]
            )

            if nearest_result is None or walking_distance_m < nearest_result["distance_m"]:
                nearest_result = {
                    "demand_point": demand["name"],
                    "demand_type": demand["type"],
                    "demand_lat": demand["lat"],
                    "demand_lon": demand["lon"],
                    "nearest_charger": charger["address"],
                    "charger_id": charger["id"],
                    "provider": charger["provider"],
                    "charger_lat": charger["lat"],
                    "charger_lon": charger["lon"],
                    "distance_m": round(walking_distance_m, 1),
                    "accessibility_category": accessibility_category(walking_distance_m),
                }

        if nearest_result is None:
            raise RuntimeError(f"No reachable charger found for {demand['name']}")

        results.append(nearest_result)

    summary = {
        "cleaned_walking_traces_used": trace_count,
        "network_nodes_used": len(network_nodes),
        "chargers_inside_strijp_s": len(chargers),
        "average_walking_distance_m": average_walking_distance(results),
        "coverage_within_300m_percent": coverage_within_threshold(results, 300),
        "coverage_within_500m_percent": coverage_within_threshold(results, 500),
    }

    return results, summary


def average_walking_distance(results):
    total_distance = sum(row["distance_m"] for row in results)
    return round(total_distance / len(results), 1)


def coverage_within_threshold(results, threshold_m):
    covered_points = sum(row["distance_m"] <= threshold_m for row in results)
    coverage_percent = covered_points / len(results) * 100
    return round(coverage_percent, 1)


def print_results(results, summary):
    print("Walking-distance KPI table")
    print("-" * 100)
    print(
        f"{'Demand point':35} {'Demand type':30} "
        f"{'Nearest charger':25} {'Distance (m)':>12} {'Category':>20}"
    )
    print("-" * 100)

    for row in results:
        print(
            f"{row['demand_point'][:35]:35} "
            f"{row['demand_type'][:30]:30} "
            f"{row['nearest_charger'][:25]:25} "
            f"{row['distance_m']:12.1f} "
            f"{row['accessibility_category']:>20}"
        )

    print()
    print("KPI summary")
    print("-" * 40)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    kpi_results, kpi_summary = calculate_walking_kpi()
    print_results(kpi_results, kpi_summary)
