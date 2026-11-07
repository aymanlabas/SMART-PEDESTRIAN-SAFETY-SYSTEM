import xml.etree.ElementTree as ET

file = "network/modern_district.net.xml"
root = ET.parse(file).getroot()

junctions = []

for j in root.findall("junction"):
    if j.get("type") != "internal":
        junctions.append({
            "id": j.get("id"),
            "x": float(j.get("x")),
            "y": float(j.get("y"))
        })

xs = [j["x"] for j in junctions]
ys = [j["y"] for j in junctions]

print("===== URBAN NETWORK BOUNDS =====")
print("X:", min(xs), "to", max(xs))
print("Y:", min(ys), "to", max(ys))
print("Total junctions:", len(junctions))

print("\n===== JUNCTION COORDINATES =====")
for j in junctions:
    print(f'{j["id"]:8} X={j["x"]:10.2f} Y={j["y"]:10.2f}')
