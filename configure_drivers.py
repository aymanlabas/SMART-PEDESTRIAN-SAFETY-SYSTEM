import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

network_file = Path("network/modern_district.net.xml")
routes_file = Path("routes/vehicles_long.rou.xml")

# Backup
for file in [network_file, routes_file]:
    backup = Path(str(file) + ".original.bak")
    if not backup.exists():
        shutil.copy2(file, backup)

# 1. Increase speed only on vehicle lanes
parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
tree = ET.parse(network_file, parser=parser)
root = tree.getroot()

updated_lanes = 0

for edge in root.findall("edge"):
    if edge.get("function"):
        continue

    for lane in edge.findall("lane"):
        disallowed = (lane.get("disallow") or "").split()

        if "pedestrian" in disallowed:
            lane.set("speed", "25.00")
            updated_lanes += 1

tree.write(network_file, encoding="UTF-8", xml_declaration=True)

# 2. Define driver profiles
parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
tree = ET.parse(routes_file, parser=parser)
root = tree.getroot()

profiles = [
    ("normalDriver", "13.89"),
    ("fastDriver", "19.44"),
    ("veryFastDriver", "25.00")
]

existing = {v.get("id") for v in root.findall("vType")}

for type_id, speed in profiles:
    if type_id not in existing:
        vtype = ET.Element("vType", {
            "id": type_id,
            "vClass": "passenger",
            "maxSpeed": speed
        })
        root.insert(0, vtype)

# 3. Assign profiles to existing vehicles
vehicles = root.findall("vehicle")

counts = {
    "normalDriver": 0,
    "fastDriver": 0,
    "veryFastDriver": 0
}

for i, vehicle in enumerate(vehicles):
    if i % 10 <= 6:
        driver = "normalDriver"
    elif i % 10 <= 8:
        driver = "fastDriver"
    else:
        driver = "veryFastDriver"

    vehicle.set("type", driver)
    counts[driver] += 1

ET.indent(tree, space="    ")
tree.write(routes_file, encoding="UTF-8", xml_declaration=True)

print("\n===== MODIFICATION TERMINEE =====")
print("Vehicle lanes updated:", updated_lanes)
print("Total vehicles:", len(vehicles))
print("Normal drivers:", counts["normalDriver"])
print("Fast drivers:", counts["fastDriver"])
print("Very fast drivers:", counts["veryFastDriver"])
print("\nVehicle lane speed limit: 90 km/h")
