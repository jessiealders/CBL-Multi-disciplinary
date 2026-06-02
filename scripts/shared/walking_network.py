import heapq
import json
import math
from collections import defaultdict, deque
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    """Straight-line distance between two latitude/longitude points."""
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


def coord_key(lon, lat):
    return f"{lon:.7f},{lat:.7f}"


def _iter_line_coordinates(geometry):
    if geometry["type"] == "LineString":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiLineString":
        yield from geometry["coordinates"]


class WalkingNetwork:
    """Reusable real walking-network distance calculator."""

    def __init__(self, nodes, graph, trace_count):
        self.nodes = nodes
        self.graph = graph
        self.trace_count = trace_count
        self.allowed_nodes = self._largest_connected_component()
        self.network_nodes = list(self.allowed_nodes)
        self._distance_cache = {}

    @classmethod
    def from_geojson(cls, path):
        path = Path(path)
        geojson = json.loads(path.read_text(encoding="utf-8"))
        nodes = {}
        graph = defaultdict(list)

        for feature in geojson["features"]:
            for coordinates in _iter_line_coordinates(feature["geometry"]):
                for start, end in zip(coordinates, coordinates[1:]):
                    start_key = coord_key(start[0], start[1])
                    end_key = coord_key(end[0], end[1])

                    nodes[start_key] = (start[1], start[0])
                    nodes[end_key] = (end[1], end[0])

                    length_m = haversine_m(start[1], start[0], end[1], end[0])
                    graph[start_key].append((end_key, length_m))
                    graph[end_key].append((start_key, length_m))

        return cls(nodes, graph, len(geojson["features"]))

    def _largest_connected_component(self):
        seen = set()
        components = []

        for node in self.nodes:
            if node in seen:
                continue

            queue = deque([node])
            seen.add(node)
            component = {node}

            while queue:
                current = queue.popleft()
                for neighbor, _ in self.graph[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        components.sort(key=len, reverse=True)
        return components[0] if components else set()

    def nearest_node(self, lat, lon):
        nearest_node = None
        nearest_distance = float("inf")

        for node_id in self.network_nodes:
            node_lat, node_lon = self.nodes[node_id]
            distance = haversine_m(lat, lon, node_lat, node_lon)

            if distance < nearest_distance:
                nearest_node = node_id
                nearest_distance = distance

        return nearest_node, nearest_distance

    def _shortest_path_distances(self, start_node):
        if start_node in self._distance_cache:
            return self._distance_cache[start_node]

        distances = {start_node: 0.0}
        heap = [(0.0, start_node)]

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            if current_distance != distances[current_node]:
                continue

            for neighbor, edge_length in self.graph[current_node]:
                if neighbor not in self.allowed_nodes:
                    continue

                new_distance = current_distance + edge_length
                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))

        self._distance_cache[start_node] = distances
        return distances

    def distance_m(self, start_lat, start_lon, end_lat, end_lon):
        """Walking distance over the network, including snap distance at both ends."""
        start_node, start_snap = self.nearest_node(start_lat, start_lon)
        end_node, end_snap = self.nearest_node(end_lat, end_lon)

        if start_node is None or end_node is None:
            return haversine_m(start_lat, start_lon, end_lat, end_lon)

        distances = self._shortest_path_distances(start_node)
        network_distance = distances.get(end_node)

        if network_distance is None:
            return haversine_m(start_lat, start_lon, end_lat, end_lon)

        return start_snap + network_distance + end_snap
