
import argparse
import copy
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

PROJECT = Path(__file__).resolve().parent.parent

VEHICLES_FILE = PROJECT / "routes" / "vehicles_long.rou.xml"
PEDESTRIANS_FILE = PROJECT / "scenarios" / "slow_pedestrians.rou.xml"

OUTPUT_ROOT = PROJECT / "scenarios" / "generated_v2"

# ==========================================
# ZONES
# ==========================================

ZONES = {
    "school": ["A2", "B2", "A3", "B3"],
    "hospital": ["C2", "D2", "C3", "D3"],
    "commercial": ["C1", "D1"],
    "residential_a": ["A1", "B1", "A2", "B2"],
    "residential_b": ["A0", "B0", "A1", "B1"],
    "residential_c": ["C0", "D0", "C1", "D1"],
}

# ==========================================
# AVAILABLE TYPES
# ==========================================

VEHICLE_TYPES = {
    "normal": ["normalDriver"],
    "mixed": ["normalDriver", "fastDriver"],
    "fast": ["fastDriver", "veryFastDriver"],
}

PEDESTRIAN_TYPES = {
    "normal": ["normalPed"],
    "mixed": ["normalPed", "slowPed", "childBehaviorPed"],
    "vulnerable": ["slowPed", "childBehaviorPed"],
}

# ==========================================
# SCENARIO CONFIGURATIONS
# ==========================================

SCENARIOS = {
    "A": {
        "name": "low_density",
        "vehicle_time_factor": 1.50,
        "pedestrian_time_factor": 1.50,
        "vehicle_profile": "normal",
        "pedestrian_profile": "normal",
        "jitter": 0.3,
        "seed": 101,
    },
    "B": {
        "name": "medium_density",
        "vehicle_time_factor": 1.00,
        "pedestrian_time_factor": 1.00,
        "vehicle_profile": "normal",
        "pedestrian_profile": "mixed",
        "jitter": 0.5,
        "seed": 202,
    },
    "C": {
        "name": "high_density",
        "vehicle_time_factor": 0.70,
        "pedestrian_time_factor": 0.70,
        "vehicle_profile": "mixed",
        "pedestrian_profile": "mixed",
        "jitter": 0.7,
        "seed": 303,
    },
    "D": {
        "name": "fast_vehicles",
        "vehicle_time_factor": 1.00,
        "pedestrian_time_factor": 0.80,
        "vehicle_profile": "fast",
        "pedestrian_profile": "vulnerable",
        "jitter": 0.8,
        "seed": 404,
    },
    "E": {
        "name": "mixed_traffic",
        "vehicle_time_factor": 0.75,
        "pedestrian_time_factor": 1.00,
        "vehicle_profile": "mixed",
        "pedestrian_profile": "normal",
        "jitter": 1.0,
        "seed": 505,
    },
}

# ==========================================
# ROUTE SELECTION
# ==========================================

def touches_junction(edge, junctions):

    if edge.startswith(":"):
        return any(
            edge.startswith(f":{junction}_")
            for junction in junctions
        )

    return any(
        edge.startswith(junction) or edge.endswith(junction)
        for junction in junctions
    )


def vehicle_matches(vehicle, junctions):

    route = vehicle.find("route")

    if route is None:
        return False

    edges = route.get("edges", "").split()

    return any(
        touches_junction(edge, junctions)
        for edge in edges
    )


def pedestrian_matches(person, junctions):

    walk = person.find("walk")

    if walk is None:
        return False

    edges = walk.get("edges", "").split()

    return any(
        touches_junction(edge, junctions)
        for edge in edges
    )


# ==========================================
# TYPE VALIDATION
# ==========================================

def validate_types(root, required_types, filename):

    available = {
        item.get("id")
        for item in root.findall("vType")
    }

    missing = set(required_types) - available

    if missing:
        raise ValueError(
            f"Missing vTypes in {filename}: {missing}"
        )


# ==========================================
# DEPARTURE TIME
# ==========================================

def adjust_depart(original, factor, jitter, rng):

    original_depart = float(original.get("depart", "0"))

    new_depart = original_depart / factor

    new_depart += rng.uniform(-jitter, jitter)

    return max(0.0, new_depart)


# ==========================================
# SORT ELEMENTS BY DEPARTURE TIME
# ==========================================

def sort_routes_by_depart(root, tag):

    # Get all elements of the requested type
    elements = root.findall(tag)

    # Sort by departure time
    elements.sort(
        key=lambda element: float(
            element.get("depart", "0")
        )
    )

    # Remove all elements of this type
    for element in root.findall(tag):
        root.remove(element)

    # Reinsert them in sorted order
    for element in elements:
        root.append(element)
    elements = root.findall(tag)

    # Sort using numeric departure time
    elements.sort(
        key=lambda element: float(
            element.get("depart", "0")
        )
    )

    # Remove old elements
    for element in elements:
        root.remove(element)

    # Add them again in sorted order
    for element in elements:
        root.append(element)


# ==========================================
# GENERATE ONE SCENARIO
# ==========================================

def generate_scenario(zone, scenario_id):

    config = SCENARIOS[scenario_id]
    rng = random.Random(config["seed"])

    junctions = ZONES[zone]

    vehicle_tree = ET.parse(VEHICLES_FILE)
    pedestrian_tree = ET.parse(PEDESTRIANS_FILE)

    vehicle_root = vehicle_tree.getroot()
    pedestrian_root = pedestrian_tree.getroot()

    selected_vehicles = [
        element for element in vehicle_root.findall("vehicle")
        if vehicle_matches(element, junctions)
    ]

    selected_pedestrians = [
        element for element in pedestrian_root.findall("person")
        if pedestrian_matches(element, junctions)
    ]

    # Validate vehicle and pedestrian types
    validate_types(
        vehicle_root,
        VEHICLE_TYPES[config["vehicle_profile"]],
        VEHICLES_FILE
    )

    validate_types(
        pedestrian_root,
        PEDESTRIAN_TYPES[config["pedestrian_profile"]],
        PEDESTRIANS_FILE
    )

    # ======================================
    # OUTPUT DIRECTORY
    # ======================================

    output_dir = (
        OUTPUT_ROOT /
        zone /
        f"scenario_{scenario_id}_{config['name']}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================
    # VEHICLES
    # ======================================

    vehicle_output = ET.Element("routes")

    for vtype in vehicle_root.findall("vType"):
        vehicle_output.append(copy.deepcopy(vtype))

    vehicle_types = VEHICLE_TYPES[
        config["vehicle_profile"]
    ]

    for index, original in enumerate(selected_vehicles):

        vehicle = copy.deepcopy(original)

        vehicle.set(
            "id",
            f"{zone}_{scenario_id}_veh_{index}"
        )

        vehicle.set(
            "type",
            rng.choice(vehicle_types)
        )

        depart = adjust_depart(
            original,
            config["vehicle_time_factor"],
            config["jitter"],
            rng
        )

        vehicle.set("depart", f"{depart:.2f}")

        vehicle_output.append(vehicle)

    # ======================================
    # PEDESTRIANS
    # ======================================

    pedestrian_output = ET.Element("routes")

    for vtype in pedestrian_root.findall("vType"):
        pedestrian_output.append(copy.deepcopy(vtype))

    pedestrian_types = PEDESTRIAN_TYPES[
        config["pedestrian_profile"]
    ]

    for index, original in enumerate(selected_pedestrians):

        person = copy.deepcopy(original)

        person.set(
            "id",
            f"{zone}_{scenario_id}_ped_{index}"
        )

        person.set(
            "type",
            rng.choice(pedestrian_types)
        )

        depart = adjust_depart(
            original,
            config["pedestrian_time_factor"],
            config["jitter"],
            rng
        )

        person.set("depart", f"{depart:.2f}")

        pedestrian_output.append(person)

    # ======================================
    # SORT BEFORE SAVING
    # ======================================

    sort_routes_by_depart(vehicle_output, "vehicle")
    sort_routes_by_depart(pedestrian_output, "person")

    # ======================================
    # SAVE XML
    # ======================================

    vehicle_path = output_dir / "vehicles.rou.xml"
    pedestrian_path = output_dir / "pedestrians.rou.xml"

    ET.indent(vehicle_output, space="    ")
    ET.indent(pedestrian_output, space="    ")

    ET.ElementTree(vehicle_output).write(
        vehicle_path,
        encoding="UTF-8",
        xml_declaration=True
    )

    ET.ElementTree(pedestrian_output).write(
        pedestrian_path,
        encoding="UTF-8",
        xml_declaration=True
    )

    # ======================================
    # VALIDATE SORTING
    # ======================================

    for path, tag in [
        (vehicle_path, "vehicle"),
        (pedestrian_path, "person")
    ]:

        check_root = ET.parse(path).getroot()

        departures = [
            float(element.get("depart", "0"))
            for element in check_root.findall(tag)
        ]

        if departures != sorted(departures):
            raise ValueError(
                f"Departure sorting failed: {path}"
            )

    # ======================================
    # REPORT
    # ======================================

    print("\n======================================")
    print(f"ZONE: {zone}")
    print(f"SCENARIO: {scenario_id} - {config['name']}")
    print("======================================")

    print("Intersections:", junctions)
    print("Seed:", config["seed"])
    print("Vehicle profile:", config["vehicle_profile"])
    print("Pedestrian profile:", config["pedestrian_profile"])

    print("Generated vehicles:", len(selected_vehicles))
    print("Generated pedestrians:", len(selected_pedestrians))

    print("Vehicle XML:", vehicle_path)
    print("Pedestrian XML:", pedestrian_path)

    print("Departure sorting: OK")


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate SUMO V2 scenarios"
    )

    parser.add_argument(
        "--zone",
        choices=ZONES.keys(),
        required=True
    )

    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E", "all"],
        default="all"
    )

    args = parser.parse_args()

    if not VEHICLES_FILE.exists():
        raise FileNotFoundError(VEHICLES_FILE)

    if not PEDESTRIANS_FILE.exists():
        raise FileNotFoundError(PEDESTRIANS_FILE)

    scenarios = (
        list(SCENARIOS.keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    for scenario_id in scenarios:
        generate_scenario(args.zone, scenario_id)

    print("\n======================================")
    print("V2 SCENARIO GENERATION COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
