
import csv
from collections import Counter

file_path = "datasets/simulation_data.csv"

total = 0
types = Counter()
vehicle_ids = set()
pedestrian_ids = set()
min_time = None
max_time = 0
invalid_rows = 0

with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            time = float(row["time"])
            x = float(row["x"])
            y = float(row["y"])
            speed = float(row["speed"])

            total += 1
            types[row["type"]] += 1

            if row["type"] == "vehicle":
                vehicle_ids.add(row["id"])
            elif row["type"] == "pedestrian":
                pedestrian_ids.add(row["id"])

            min_time = time if min_time is None else min(min_time, time)
            max_time = max(max_time, time)

        except (ValueError, KeyError):
            invalid_rows += 1

print("\n===== ANALYSE DU DATASET =====")
print("Nombre total de lignes :", total)
print("Véhicules :", types["vehicle"])
print("Piétons :", types["pedestrian"])
print("Identifiants véhicules :", len(vehicle_ids))
print("Identifiants piétons :", len(pedestrian_ids))
print("Temps minimum :", min_time)
print("Temps maximum :", max_time)
print("Lignes invalides :", invalid_rows)
