
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from math import hypot

NET = Path("network/modern_district.net.xml")
VEH = Path("routes/vehicles_long.rou.xml")
PED = Path("scenarios/slow_pedestrians.rou.xml")

ZONES = {
    "School": (20, 180, 420, 580),
    "Hospital": (420, 580, 420, 580),
    "Commercial": (425, 580, 230, 290),
    "Residential_A": (25, 155, 230, 370),
    "Residential_B": (25, 165, 25, 165),
    "Residential_C": (425, 555, 25, 165),
}


def parse_shape(shape):
    points = []
    if not shape:
        return points

    for item in shape.split():
        try:
            x, y = item.split(",")[:2]
            points.append((float(x), float(y)))
        except ValueError:
            pass

    return points


def point_rect_distance(x, y, rect):
    xmin, xmax, ymin, ymax = rect
    dx = max(xmin - x, 0, x - xmax)
    dy = max(ymin - y, 0, y - ymax)
    return hypot(dx, dy)


# Load network
print("Loading network...")
root = ET.parse(NET).getroot()

edges = {}

for edge in root.findall("edge"):
    if edge.get("function") == "internal":
        continue

    points = []
    for lane in edge.findall("lane"):
        points.extend(parse_shape(lane.get("shape")))

    if points:
        edges[edge.get("id")] = points

print("Network edges:", len(edges))


# Extract vehicle route edge usage
vehicle_usage = Counter()
vehicle_root = ET.parse(VEH).getroot()

for route in vehicle_root.findall(".//route"):
    vehicle_usage.update(route.get("edges", "").split())


# Extract pedestrian walk edge usage
pedestrian_usage = Counter()
ped_root = ET.parse(PED).getroot()

for walk in ped_root.findall(".//walk"):
    pedestrian_usage.update(walk.get("edges", "").split())

# Also inspect route references, if present
for route in ped_root.findall(".//route"):
    pedestrian_usage.update(route.get("edges", "").split())

print("Vehicle route edges:", len(vehicle_usage))
print("Pedestrian walk edges:", len(pedestrian_usage))


# Analyze each zone
for zone_name, rect in ZONES.items():
    print("\n" + "=" * 65)
    print("ZONE:", zone_name)
    print("=" * 65)

    results = []

    for edge_id, points in edges.items():
        distance = min(
            point_rect_distance(x, y, rect)
            for x, y in points
        )

        if distance <= 40:
            results.append((
                distance,
                edge_id,
                vehicle_usage[edge_id],
                pedestrian_usage[edge_id],
                points[0],
                points[-1]
            ))

    results.sort(key=lambda item: item[0])

    if not results:
        print("No nearby edges found.")
        continue

    print("Nearby edges (maximum 40m):")

    for distance, edge_id, veh, ped, start, end in results:
        print(
            f"{edge_id:15} "
            f"distance={distance:6.2f}m | "
            f"vehicles={veh:4} | "
            f"pedestrians={ped:4} | "
            f"start={start} | end={end}"
        )

    print("\nClosest 5 edges:")
    for item in results[:5]:
        print(f"{item[1]} -> {item[0]:.2f}m")
