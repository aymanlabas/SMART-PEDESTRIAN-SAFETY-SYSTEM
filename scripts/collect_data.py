
import os
import sys
import csv
import traci
from pathlib import Path

# Configuration SUMO
SUMO_HOME = r"C:\Program Files (x86)\SUMO"
os.environ["SUMO_HOME"] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, "tools"))

# Chemins du projet
PROJECT = Path(__file__).resolve().parent.parent

NETWORK = PROJECT / "network" / "modern_district.net.xml"
VEHICLES = PROJECT / "routes" / "vehicles_long.rou.xml"
PEDESTRIANS = PROJECT / "scenarios" / "slow_pedestrians.rou.xml"
URBAN = PROJECT / "additional" / "urban_objects.add.xml"

# Fichier CSV de sortie
OUTPUT = PROJECT / "datasets" / "simulation_data.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

sumo_cmd = [
    "sumo",
    "-n", str(NETWORK),
    "-r", f"{VEHICLES},{PEDESTRIANS}",
    "-a", str(URBAN),
    "--step-length", "0.1"
]

try:
    traci.start(sumo_cmd)

    print("Connexion SUMO réussie.")
    print("Début de la collecte des données...")

    total_records = 0

    with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "id",
            "type",
            "x",
            "y",
            "speed",
            "road_id"
        ])

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            current_time = traci.simulation.getTime()

            # Collecte des véhicules
            for vehicle_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vehicle_id)

                writer.writerow([
                    current_time,
                    vehicle_id,
                    "vehicle",
                    x,
                    y,
                    traci.vehicle.getSpeed(vehicle_id),
                    traci.vehicle.getRoadID(vehicle_id)
                ])

                total_records += 1

            # Collecte des piétons
            for person_id in traci.person.getIDList():
                x, y = traci.person.getPosition(person_id)

                writer.writerow([
                    current_time,
                    person_id,
                    "pedestrian",
                    x,
                    y,
                    traci.person.getSpeed(person_id),
                    traci.person.getRoadID(person_id)
                ])

                total_records += 1

            if int(current_time * 10) % 100 == 0:
                print(
                    f"Temps : {current_time:.1f}s | "
                    f"Enregistrements : {total_records}"
                )

    print("\nCollecte terminée !")
    print(f"Total des enregistrements : {total_records}")
    print(f"Fichier créé : {OUTPUT}")

except KeyboardInterrupt:
    print("\nCollecte interrompue manuellement.")
    print(f"Enregistrements collectés : {total_records}")

except Exception as error:
    print("Erreur :", error)

finally:
    try:
        traci.close()
        print("Connexion SUMO fermée.")
    except Exception:
        print("Connexion SUMO déjà fermée.")