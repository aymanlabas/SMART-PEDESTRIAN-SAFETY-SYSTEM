
import xml.etree.ElementTree as ET
from pathlib import Path

VEH_FILE = Path("routes/vehicles_long.rou.xml")
PED_FILE = Path("scenarios/slow_pedestrians.rou.xml")

ZONES = {
    "School": ["A2", "B2", "A3", "B3"],
    "Hospital": ["C2", "D2", "C3", "D3"],
    "Commercial": ["C1", "D1"],
    "Residential_A": ["A1", "B1", "A2", "B2"],
    "Residential_B": ["A0", "B0", "A1", "B1"],
    "Residential_C": ["C0", "D0", "C1", "D1"],
}

vehicle_root = ET.parse(VEH_FILE).getroot()
pedestrian_root = ET.parse(PED_FILE).getroot()

vehicle_routes = []
for vehicle in vehicle_root.findall(".//vehicle"):
    route = vehicle.find("route")
    if route is not None:
        vehicle_routes.append((
            vehicle.get("id"),
            route.get("edges", "").split()
        ))

pedestrian_routes = []
for person in pedestrian_root.findall(".//person"):
    for walk in person.findall(".//walk"):
        pedestrian_routes.append((
            person.get("id"),
            walk.get("edges", "").split()
        ))

for zone, junctions in ZONES.items():
    print("\n" + "=" * 65)
    print("ZONE:", zone, "| Junctions:", ", ".join(junctions))
    print("=" * 65)

    # Vehicle routes passing through candidate junctions
    print("\n--- VEHICLE ROUTES ---")
    count = 0

    for vehicle_id, edges in vehicle_routes:
        matching = [
            edge for edge in edges
            if any(
                edge.startswith(j) or edge.endswith(j)
                for j in junctions
            )
        ]

        if matching:
            print(vehicle_id, ":", " ".join(edges))
            count += 1

        if count >= 5:
            break

    if count == 0:
        print("No matching vehicle routes.")

    # Pedestrian routes using crossing/walking edges
    print("\n--- PEDESTRIAN ROUTES ---")
    count = 0

    for person_id, edges in pedestrian_routes:
        matching = [
            edge for edge in edges
            if any(edge.startswith(":" + j + "_") for j in junctions)
        ]

        if matching:
            print(person_id, ":", " ".join(edges))
            count += 1

        if count >= 5:
            break

    if count == 0:
        print("No matching pedestrian routes.")
