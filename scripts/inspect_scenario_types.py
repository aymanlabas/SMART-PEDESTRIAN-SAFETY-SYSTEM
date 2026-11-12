
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

FILES = {
    "vehicles": PROJECT / "routes" / "vehicles_long.rou.xml",
    "pedestrians": PROJECT / "scenarios" / "slow_pedestrians.rou.xml"
}

for name, path in FILES.items():
    print(f"\n========== {name.upper()} ==========")

    root = ET.parse(path).getroot()

    print("\n--- vType definitions ---")
    for element in root.findall("vType"):
        print(ET.tostring(element, encoding="unicode"))

    tag = "vehicle" if name == "vehicles" else "person"

    print(f"\n--- First 3 {tag} definitions ---")
    for element in root.findall(tag)[:3]:
        print(ET.tostring(element, encoding="unicode"))
