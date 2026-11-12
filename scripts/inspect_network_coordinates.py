
import xml.etree.ElementTree as ET
from pathlib import Path

NET_FILE = Path("network/modern_district.net.xml")
VEH_FILE = Path("routes/vehicles_long.rou.xml")
PED_FILE = Path("scenarios/slow_pedestrians.rou.xml")
ADD_FILE = Path("additional/urban_objects.add.xml")


def parse_points(shape):
    if not shape:
        return []
    points = []
    for item in shape.split():
        try:
            x, y = item.split(",")[:2]
            points.append((float(x), float(y)))
        except ValueError:
            continue
    return points


# 1. Network coordinate system
net = ET.parse(NET_FILE).getroot()
location = net.find("location")

print("\n========== NETWORK LOCATION ==========")
if location is not None:
    print(location.attrib)
else:
    print("No location element found.")

# 2. Calculate network bounding box
all_points = []

for edge in net.findall("edge"):
    if edge.get("function") == "internal":
        continue

    for lane in edge.findall("lane"):
        all_points.extend(parse_points(lane.get("shape")))

print("\n========== NETWORK BOUNDARIES ==========")
if all_points:
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    print(f"X: {min(xs):.2f} -> {max(xs):.2f}")
    print(f"Y: {min(ys):.2f} -> {max(ys):.2f}")

# 3. Inspect actual vehicle and pedestrian route edges
def inspect_routes(file_path, tag):
    root = ET.parse(file_path).getroot()
    print(f"\n========== {tag} ROUTE COORDINATES ==========")

    shown = 0
    for route in root.findall(".//route"):
        edges = route.get("edges", "").split()

        for edge_id in edges:
            edge = net.find(f"./edge[@id='{edge_id}']")
            if edge is None:
                continue

            lane = edge.find("lane")
            if lane is None:
                continue

            points = parse_points(lane.get("shape"))
            if points:
                print(
                    f"{edge_id}: "
                    f"start={points[0]}, end={points[-1]}"
                )
                shown += 1

            if shown >= 12:
                return

inspect_routes(VEH_FILE, "VEHICLES")
inspect_routes(PED_FILE, "PEDESTRIANS")

# 4. Inspect zone polygon coordinates
print("\n========== ZONE POLYGONS ==========")

root = ET.parse(ADD_FILE).getroot()

for poly in root.findall(".//poly"):
    print(
        f"{poly.get('id')}: "
        f"{poly.get('shape')}"
    )
