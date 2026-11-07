import xml.etree.ElementTree as ET

tree = ET.parse("network/modern_district.net.xml")
root = tree.getroot()

count = 0

for edge in root.findall("edge"):
    if edge.get("function"):
        continue

    lanes = edge.findall("lane")

    for lane in lanes:
        print(
            "Edge:", edge.get("id"),
            "| Lane:", lane.get("id"),
            "| Speed:", lane.get("speed"),
            "| Allow:", lane.get("allow"),
            "| Disallow:", lane.get("disallow")
        )

        count += 1

        if count >= 20:
            break

    if count >= 20:
        break
