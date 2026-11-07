
import xml.etree.ElementTree as ET
from pathlib import Path
import math

# Fichier de sortie
output = Path("additional/urban_objects.add.xml")
output.parent.mkdir(parents=True, exist_ok=True)

root = ET.Element("additional")


def polygon(name, points, color, layer=1, fill="true"):
    ET.SubElement(root, "poly", {
        "id": name,
        "type": "urban",
        "color": color,
        "layer": str(layer),
        "fill": fill,
        "shape": " ".join(f"{x},{y}" for x, y in points)
    })


def rectangle(name, x1, y1, x2, y2, color, layer=1):
    polygon(name, [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ], color, layer)


def tree(name, x, y, radius=3):
    points = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        points.append((
            x + radius * math.cos(angle),
            y + radius * math.sin(angle)
        ))
    polygon(name, points, "35,139,75", 5)


# ==========================================
# 1. SCHOOL - BLOCK A3
# ==========================================

rectangle("school_campus", 20, 420, 180, 580, "190,210,180", 1)
rectangle("school_building", 45, 475, 145, 555, "225,180,120", 2)
rectangle("school_playground", 35, 430, 165, 465, "130,190,130", 2)
rectangle("school_entrance", 75, 465, 115, 470, "210,210,210", 3)


# ==========================================
# 2. NORTH PARK - BLOCK B3
# ==========================================

rectangle("north_park", 220, 420, 380, 580, "120,190,125", 1)

# Walking paths
rectangle("north_path_horizontal", 230, 490, 370, 500, "210,200,170", 2)
rectangle("north_path_vertical", 295, 430, 305, 570, "210,200,170", 2)

# Trees
for i, (x, y) in enumerate([
    (245, 445), (270, 455), (335, 445),
    (355, 465), (245, 535), (270, 550),
    (335, 545), (355, 560)
]):
    tree(f"north_tree_{i}", x, y)


# ==========================================
# 3. HOSPITAL - BLOCK C3
# ==========================================

rectangle("hospital_campus", 420, 420, 580, 580, "190,205,215", 1)
rectangle("hospital_main", 440, 475, 555, 555, "220,225,235", 2)
rectangle("hospital_annex", 445, 435, 490, 465, "200,215,225", 2)
rectangle("hospital_parking", 500, 430, 565, 465, "125,130,135", 2)


# ==========================================
# 4. RESIDENTIAL AREA - BLOCK A2
# ==========================================

rectangle("residential_A2_1", 25, 230, 75, 280, "190,175,155", 2)
rectangle("residential_A2_2", 95, 230, 155, 285, "180,165,150", 2)
rectangle("residential_A2_3", 45, 315, 145, 370, "200,185,165", 2)


# ==========================================
# 5. CENTRAL PARK - BLOCK B2
# ==========================================

rectangle("central_park", 220, 220, 380, 380, "110,180,115", 1)

rectangle("central_path_horizontal", 230, 295, 370, 305, "220,210,180", 2)
rectangle("central_path_vertical", 295, 230, 305, 370, "220,210,180", 2)

for i, (x, y) in enumerate([
    (245, 245), (270, 260), (335, 245),
    (355, 270), (245, 340), (270, 355),
    (335, 350), (355, 330), (250, 300),
    (350, 300), (300, 250), (300, 350)
]):
    tree(f"central_tree_{i}", x, y)


# ==========================================
# 6. COMMERCIAL AREA - BLOCK C2
# ==========================================

rectangle("shop_1", 425, 230, 470, 290, "210,160,120", 2)
rectangle("shop_2", 485, 230, 530, 290, "220,180,130", 2)
rectangle("shop_3", 545, 230, 580, 290, "200,150,110", 2)


# ==========================================
# 7. RESIDENTIAL AREA - BLOCK A1
# ==========================================

rectangle("residential_A1_1", 25, 25, 80, 80, "190,175,155", 2)
rectangle("residential_A1_2", 105, 25, 165, 80, "180,165,150", 2)
rectangle("residential_A1_3", 45, 110, 145, 165, "200,185,165", 2)


# ==========================================
# 8. PUBLIC PARKING - BLOCK B1
# ==========================================

rectangle("parking_background", 220, 20, 380, 180, "115,120,125", 1)

# Parking spaces
for i in range(8):
    x = 230 + i * 18
    rectangle(
        f"parking_space_{i}",
        x, 40, x + 14, 85,
        "225,225,225", 2
    )

rectangle("parking_separator", 225, 105, 375, 108, "240,240,240", 2)


# ==========================================
# 9. RESIDENTIAL AREA - BLOCK C1
# ==========================================

rectangle("residential_C1_1", 425, 25, 480, 80, "190,175,155", 2)
rectangle("residential_C1_2", 495, 25, 555, 80, "180,165,150", 2)
rectangle("residential_C1_3", 445, 110, 555, 165, "200,185,165", 2)


# ==========================================
# SAVE XML
# ==========================================

ET.indent(root, space="    ")

tree_xml = ET.ElementTree(root)
tree_xml.write(output, encoding="UTF-8", xml_declaration=True)

print("Urban environment created successfully!")
print(f"Output file: {output}")
print(f"Total urban objects: {len(root)}")
