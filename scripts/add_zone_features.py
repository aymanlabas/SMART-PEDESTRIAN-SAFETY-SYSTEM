
import csv
import math
from itertools import groupby

INPUT_FILE = "datasets/pedestrian_vehicle_interactions_v2.csv"
OUTPUT_FILE = "datasets/pedestrian_vehicle_interactions_zones.csv"

# A small buffer includes nearby roads around each urban object.
ZONE_BUFFER = 20.0
DENSITY_RADIUS = 30.0
INTERSECTION_RADIUS = 30.0

# Rectangles extracted from urban_objects.add.xml:
# xmin, xmax, ymin, ymax

ZONES = {
    "school_zone": [
        ("school_campus", (20, 180, 420, 580))
    ],

    "hospital_zone": [
        ("hospital_campus", (420, 580, 420, 580))
    ],

    "commercial_zone": [
        ("shops", (425, 580, 230, 290))
    ],

    "residential_zone": [
        ("residential_A1", (25, 165, 25, 165)),
        ("residential_A2", (25, 155, 230, 370)),
        ("residential_C1", (425, 555, 25, 165))
    ]
}

# All 16 SUMO intersections
INTERSECTIONS = {
    "A0": (0, 0), "A1": (0, 200),
    "A2": (0, 400), "A3": (0, 600),

    "B0": (200, 0), "B1": (200, 200),
    "B2": (200, 400), "B3": (200, 600),

    "C0": (400, 0), "C1": (400, 200),
    "C2": (400, 400), "C3": (400, 600),

    "D0": (600, 0), "D1": (600, 200),
    "D2": (600, 400), "D3": (600, 600)
}


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def inside_rectangle(x, y, bounds, buffer=0):
    xmin, xmax, ymin, ymax = bounds

    return (
        xmin - buffer <= x <= xmax + buffer
        and ymin - buffer <= y <= ymax + buffer
    )


def classify_zone(x, y):

    for zone_type, clusters in ZONES.items():

        for cluster_name, bounds in clusters:

            if inside_rectangle(
                x, y, bounds, ZONE_BUFFER
            ):
                return zone_type, cluster_name

    return "normal_zone", "none"


def detect_nearest_intersection(x, y):

    nearest_id = ""
    nearest_distance = float("inf")

    for junction_id, (jx, jy) in INTERSECTIONS.items():

        d = distance(x, y, jx, jy)

        if d < nearest_distance:
            nearest_distance = d
            nearest_id = junction_id

    is_intersection = int(
        nearest_distance <= INTERSECTION_RADIUS
    )

    return (
        is_intersection,
        nearest_id,
        round(nearest_distance, 2)
    )


def process_dataset():

    extra_columns = [
        "zone_type",
        "zone_cluster",
        "is_school_zone",
        "is_hospital_zone",
        "is_commercial_zone",
        "is_residential_zone",
        "is_intersection",
        "nearest_intersection_id",
        "intersection_distance",
        "vehicle_density_30m",
        "pedestrian_density_30m",
        "avg_vehicle_speed_30m"
    ]

    zone_counts = {}
    total_records = 0

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as source:

        reader = csv.DictReader(source)

        fieldnames = reader.fieldnames + extra_columns

        with open(
            OUTPUT_FILE, "w", newline="", encoding="utf-8"
        ) as output:

            writer = csv.DictWriter(
                output,
                fieldnames=fieldnames
            )

            writer.writeheader()

            # Records are grouped by simulation timestamp.
            for timestamp, group in groupby(
                reader, key=lambda row: row["time"]
            ):

                rows = list(group)

                vehicles = {}
                pedestrians = {}

                # Unique entities observed in interaction records.
                for row in rows:

                    vehicles[row["vehicle_id"]] = (
                        float(row["vehicle_x"]),
                        float(row["vehicle_y"]),
                        float(row["vehicle_speed"])
                    )

                    pedestrians[row["pedestrian_id"]] = (
                        float(row["pedestrian_x"]),
                        float(row["pedestrian_y"])
                    )

                vehicle_positions = list(vehicles.values())
                pedestrian_positions = list(pedestrians.values())

                for row in rows:

                    px = float(row["pedestrian_x"])
                    py = float(row["pedestrian_y"])

                    zone_type, zone_cluster = classify_zone(px, py)

                    (
                        is_intersection,
                        junction_id,
                        junction_distance
                    ) = detect_nearest_intersection(px, py)

                    nearby_vehicles = [
                        v for v in vehicle_positions
                        if distance(px, py, v[0], v[1])
                        <= DENSITY_RADIUS
                    ]

                    nearby_pedestrians = [
                        p for p in pedestrian_positions
                        if distance(px, py, p[0], p[1])
                        <= DENSITY_RADIUS
                    ]

                    avg_speed = (
                        sum(v[2] for v in nearby_vehicles)
                        / len(nearby_vehicles)
                        if nearby_vehicles else ""
                    )

                    row["zone_type"] = zone_type
                    row["zone_cluster"] = zone_cluster

                    row["is_school_zone"] = int(
                        zone_type == "school_zone"
                    )

                    row["is_hospital_zone"] = int(
                        zone_type == "hospital_zone"
                    )

                    row["is_commercial_zone"] = int(
                        zone_type == "commercial_zone"
                    )

                    row["is_residential_zone"] = int(
                        zone_type == "residential_zone"
                    )

                    row["is_intersection"] = is_intersection
                    row["nearest_intersection_id"] = junction_id
                    row["intersection_distance"] = junction_distance

                    row["vehicle_density_30m"] = len(nearby_vehicles)
                    row["pedestrian_density_30m"] = len(
                        nearby_pedestrians
                    )

                    row["avg_vehicle_speed_30m"] = (
                        round(avg_speed, 3)
                        if avg_speed != "" else ""
                    )

                    writer.writerow(row)

                    zone_counts[zone_type] = (
                        zone_counts.get(zone_type, 0) + 1
                    )

                    total_records += 1

                if total_records % 50000 < len(rows):
                    print(
                        f"Processed {total_records:,} records..."
                    )

    print("\n========== PROCESSING COMPLETE ==========")
    print("Total records:", f"{total_records:,}")
    print("Output file:", OUTPUT_FILE)

    print("\nZone distribution:")

    for zone, count in zone_counts.items():
        percentage = count / total_records * 100
        print(f"{zone}: {count:,} ({percentage:.2f}%)")

    print("==========================================")


if __name__ == "__main__":
    process_dataset()
