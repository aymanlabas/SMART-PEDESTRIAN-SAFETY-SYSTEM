
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

NET = ROOT / "network" / "modern_district.net.xml"
VEHICLES = ROOT / "routes" / "vehicles_long.rou.xml"
PEDESTRIANS = ROOT / "scenarios" / "slow_pedestrians.rou.xml"

# Approximate zone boundaries based on urban_objects.add.xml
ZONES = {
    "School": (20, 180, 420, 580),
    "Hospital": (420, 580, 420, 580),
    "Commercial": (415, 590, 210, 310),
    "Residential_A": (15, 175, 215, 385),
    "Residential_B": (15, 175, 10, 185),
}

def parse_shape(shape):
    points = []
    for item in shape.split():
        try:
            x, y = item.split(",")[:2]
            points.append((float(x), float(y)))
        except (ValueError, IndexError):
            continue
    return points


def load_route_usage(path, entity_type):
    root = ET.parse(path).getroot()
    usage = Counter()

    if entity_type == "vehicle":
        for vehicle in root.findall("vehicle"):
            route = vehicle.find("route")
            if route is not None:
                usage.update(route.get("edges", "").split())

    else:
        for person in root.findall("person"):
            for walk in person.findall("walk"):
                usage.update(walk.get("edges", "").split())

    return usage


def main():
    print("Loading SUMO network...")

    root = ET.parse(NET).getroot()

    vehicle_usage = load_route_usage(VEHICLES, "vehicle")
    pedestrian_usage = load_route_usage(PEDESTRIANS, "pedestrian")

    print(f"Vehicle route edges: {len(vehicle_usage)}")
    print(f"Pedestrian route edges: {len(pedestrian_usage)}")

    edges = {}

    for edge in root.findall("edge"):
        edge_id = edge.get("id")

        if edge.get("function") == "internal":
            pass

        points = []

        for lane in edge.findall("lane"):
            points.extend(parse_shape(lane.get("shape", "")))

        if points:
            edges[edge_id] = {
                "points": points,
                "from": edge.get("from", ""),
                "to": edge.get("to", ""),
            }

    print(f"Network edges loaded: {len(edges)}")

    for zone, (xmin, xmax, ymin, ymax) in ZONES.items():

        print("\n" + "=" * 65)
        print(f"ZONE: {zone}")
        print("=" * 65)

        found = []

        for edge_id, info in edges.items():

            points = info["points"]

            inside = any(
                xmin <= x <= xmax and ymin <= y <= ymax
                for x, y in points
            )

            if inside:
                found.append((
                    edge_id,
                    info["from"],
                    info["to"],
                    vehicle_usage[edge_id],
                    pedestrian_usage[edge_id],
                ))

        if not found:
            print("No edges found in this area.")
            continue

        print(
            f"{'EDGE':<18} {'FROM':<10} {'TO':<10} "
            f"{'VEH':<6} {'PED':<6}"
        )

        for edge_id, source, target, veh, ped in found:
            print(
                f"{edge_id:<18} {source:<10} {target:<10} "
                f"{veh:<6} {ped:<6}"
            )


if __name__ == "__main__":
    main()
