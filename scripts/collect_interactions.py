
import os
import sys
import csv
import math
import traci

from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

SUMO_HOME = r"C:\Program Files (x86)\SUMO"

os.environ["SUMO_HOME"] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, "tools"))

PROJECT = Path(__file__).resolve().parent.parent

NETWORK = PROJECT / "network" / "modern_district.net.xml"

# Routes can be changed using PowerShell environment variables
VEHICLES = Path(os.getenv(
    "SP_VEHICLES",
    str(PROJECT / "routes" / "vehicles_long.rou.xml")
))

PEDESTRIANS = Path(os.getenv(
    "SP_PEDESTRIANS",
    str(PROJECT / "scenarios" / "slow_pedestrians.rou.xml")
))

URBAN = PROJECT / "additional" / "urban_objects.add.xml"

# Independent output paths
OUTPUT = Path(os.getenv(
    "SP_OUTPUT",
    str(PROJECT / "datasets" / "pedestrian_vehicle_interactions_v2.csv")
))

COLLISION_OUTPUT = Path(os.getenv(
    "SP_COLLISION_OUTPUT",
    str(PROJECT / "results" / "interactions_collisions_v2.xml")
))

STEP_LENGTH = 0.1
MAX_SIMULATION_TIME = 600

# Maximum distance for collecting a pair
MAX_DISTANCE = 50.0

# Approximate combined contact radius (meters)
# This is a modeling assumption, not an official SUMO collision label.
CONTACT_RADIUS = 2.5


# ==========================================
# 2. TTC CALCULATION
# ==========================================

def calculate_interaction(rx, ry, vx, vy):

    distance = math.hypot(rx, ry)

    relative_speed = math.hypot(vx, vy)

    dot = rx * vx + ry * vy

    velocity_squared = vx**2 + vy**2

    closing_speed = (
        -dot / distance if distance > 1e-9 else 0.0
    )

    # Time to closest approach
    t_closest = None
    min_distance = distance

    if velocity_squared > 1e-9:

        candidate = -dot / velocity_squared

        if candidate > 0:

            t_closest = candidate

            closest_x = rx + vx * candidate
            closest_y = ry + vy * candidate

            min_distance = math.hypot(
                closest_x,
                closest_y
            )

    # TTC based on assumed contact radius
    a = velocity_squared
    b = dot
    c = distance**2 - CONTACT_RADIUS**2

    ttc = None

    if c <= 0:

        ttc = 0.0

    elif a > 1e-9:

        discriminant = b**2 - a * c

        if discriminant >= 0:

            candidate_ttc = (
                -b - math.sqrt(discriminant)
            ) / a

            if candidate_ttc >= 0:
                ttc = candidate_ttc

    return (
        distance,
        relative_speed,
        closing_speed,
        t_closest,
        min_distance,
        ttc
    )


# ==========================================
# 3. VALIDATE CONFIGURATION
# ==========================================

def validate_configuration():

    required_files = [
        NETWORK,
        VEHICLES,
        PEDESTRIANS,
        URBAN
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    COLLISION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n========== CONFIGURATION ==========")
    print(f"Network: {NETWORK}")
    print(f"Vehicles: {VEHICLES}")
    print(f"Pedestrians: {PEDESTRIANS}")
    print(f"Urban objects: {URBAN}")
    print(f"Dataset output: {OUTPUT}")
    print(f"Collision output: {COLLISION_OUTPUT}")
    print("===================================\n")


# ==========================================
# 4. MAIN COLLECTION
# ==========================================

def main():

    print("======================================")
    print(" PEDESTRIAN-VEHICLE DATA COLLECTION")
    print("======================================")

    validate_configuration()

    sumo_cmd = [
        "sumo",
        "-n", str(NETWORK),
        "-r", f"{VEHICLES},{PEDESTRIANS}",
        "-a", str(URBAN),
        "--step-length", str(STEP_LENGTH),
        "--end", str(MAX_SIMULATION_TIME),
        "--collision-output", str(COLLISION_OUTPUT),
        "--intermodal-collision.action", "warn"
    ]

    previous_vehicles = {}
    previous_pedestrians = {}

    total_records = 0
    simulation_steps = 0

    columns = [
        "time",
        "vehicle_id",
        "pedestrian_id",
        "vehicle_x",
        "vehicle_y",
        "pedestrian_x",
        "pedestrian_y",
        "vehicle_speed",
        "pedestrian_speed",
        "distance",
        "relative_speed",
        "closing_speed",
        "time_to_closest_approach",
        "predicted_min_distance",
        "ttc",
        "collision_label"
    ]

    try:

        with open(
            OUTPUT,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            traci.start(sumo_cmd)

            print("SUMO connected successfully.")
            print("Starting interaction collection...\n")

            while traci.simulation.getMinExpectedNumber() > 0:

                traci.simulationStep()

                current_time = traci.simulation.getTime()

                if current_time > MAX_SIMULATION_TIME:
                    break

                simulation_steps += 1

                vehicle_positions = {}
                pedestrian_positions = {}

                # --------------------------------
                # Get vehicle positions
                # --------------------------------

                for vehicle_id in traci.vehicle.getIDList():

                    x, y = traci.vehicle.getPosition(vehicle_id)

                    speed = traci.vehicle.getSpeed(vehicle_id)

                    vehicle_positions[vehicle_id] = (
                        x,
                        y,
                        speed
                    )

                # --------------------------------
                # Get pedestrian positions
                # --------------------------------

                for pedestrian_id in traci.person.getIDList():

                    x, y = traci.person.getPosition(pedestrian_id)

                    speed = traci.person.getSpeed(pedestrian_id)

                    pedestrian_positions[pedestrian_id] = (
                        x,
                        y,
                        speed
                    )

                # --------------------------------
                # Estimate vehicle velocity vectors
                # --------------------------------

                vehicle_velocities = {}

                for vid, data in vehicle_positions.items():

                    x, y, _ = data

                    if vid in previous_vehicles:

                        old_x, old_y = previous_vehicles[vid]

                        vx = (x - old_x) / STEP_LENGTH
                        vy = (y - old_y) / STEP_LENGTH

                        vehicle_velocities[vid] = (vx, vy)

                # --------------------------------
                # Estimate pedestrian velocity vectors
                # --------------------------------

                pedestrian_velocities = {}

                for pid, data in pedestrian_positions.items():

                    x, y, _ = data

                    if pid in previous_pedestrians:

                        old_x, old_y = previous_pedestrians[pid]

                        vx = (x - old_x) / STEP_LENGTH
                        vy = (y - old_y) / STEP_LENGTH

                        pedestrian_velocities[pid] = (vx, vy)

                # --------------------------------
                # Compare vehicles and pedestrians
                # --------------------------------

                for vid, vdata in vehicle_positions.items():

                    if vid not in vehicle_velocities:
                        continue

                    vx, vy = vehicle_velocities[vid]

                    vehicle_x, vehicle_y, vehicle_speed = vdata

                    for pid, pdata in pedestrian_positions.items():

                        if pid not in pedestrian_velocities:
                            continue

                        px, py, pedestrian_speed = pdata

                        rx = px - vehicle_x
                        ry = py - vehicle_y

                        distance = math.hypot(rx, ry)

                        if distance > MAX_DISTANCE:
                            continue

                        pvx, pvy = pedestrian_velocities[pid]

                        relative_vx = pvx - vx
                        relative_vy = pvy - vy

                        (
                            distance,
                            relative_speed,
                            closing_speed,
                            t_closest,
                            min_distance,
                            ttc
                        ) = calculate_interaction(
                            rx,
                            ry,
                            relative_vx,
                            relative_vy
                        )

                        writer.writerow({

                            "time": round(current_time, 2),

                            "vehicle_id": vid,
                            "pedestrian_id": pid,

                            "vehicle_x": vehicle_x,
                            "vehicle_y": vehicle_y,

                            "pedestrian_x": px,
                            "pedestrian_y": py,

                            "vehicle_speed": vehicle_speed,
                            "pedestrian_speed": pedestrian_speed,

                            "distance": distance,
                            "relative_speed": relative_speed,
                            "closing_speed": closing_speed,

                            "time_to_closest_approach": (
                                "" if t_closest is None
                                else t_closest
                            ),

                            "predicted_min_distance": min_distance,

                            "ttc": (
                                "" if ttc is None
                                else ttc
                            ),

                            # Future risk labels are generated
                            # separately by the labeling script.
                            "collision_label": ""
                        })

                        total_records += 1

                # --------------------------------
                # Save positions for next step
                # --------------------------------

                previous_vehicles = {
                    vid: (data[0], data[1])
                    for vid, data in vehicle_positions.items()
                }

                previous_pedestrians = {
                    pid: (data[0], data[1])
                    for pid, data in pedestrian_positions.items()
                }

                # --------------------------------
                # Progress information
                # --------------------------------

                if simulation_steps % 100 == 0:

                    print(
                        f"Time: {current_time:.1f}s | "
                        f"Interactions: {total_records}"
                    )

                    file.flush()

    except KeyboardInterrupt:

        print("\nCollection interrupted manually.")

    except Exception as error:

        print(f"\nERROR: {error}")
        raise

    finally:

        if traci.isLoaded():
            traci.close()

        print("\n======================================")
        print(" COLLECTION FINISHED")
        print("======================================")

        print(f"Simulation steps: {simulation_steps}")
        print(f"Interaction records: {total_records}")
        print(f"Dataset: {OUTPUT}")
        print(f"Collision file: {COLLISION_OUTPUT}")


if __name__ == "__main__":
    main()
