"""V4: Final Kashmir correction using exact naturalearth vertices.

Approach: Build the AJK+GB polygon directly from known PAK vertices,
then build the Aksai Chin polygon from CHN vertices.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import ssl
from shapely.geometry import Polygon, box, MultiPolygon

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except Exception:
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

world.columns = [c.lower() for c in world.columns]

pak_idx = world[world["iso_a3"] == "PAK"].index
chn_idx = world[world["iso_a3"] == "CHN"].index
ind_idx = world[world["iso_a3"] == "IND"].index

pak_geom = world.loc[pak_idx[0], "geometry"]
chn_geom = world.loc[chn_idx[0], "geometry"]
ind_geom = world.loc[ind_idx[0], "geometry"]

# ===== PAK KASHMIR (AJK + GB) =====
# The PAK polygon vertices form a closed loop. The AJK+GB territory is the
# northern "lobe" of Pakistan, bounded by:
# - East: LoC (shared with IND)  
# - North: shared with CHN
# - Northwest: the northern border of GB
# - West: approximate KP/AJK internal boundary
#
# Since naturalearth has no KP-AJK internal boundary, we need to cut along
# an approximate line. The best cut follows from the point where the LoC
# meets the international border (74.42, 30.98) westward to the NWFP/Afghan
# border area.
#
# To make this clean: use the LoC vertices on the east side, then draw a
# cutting line from the LoC bottom to the Afghan border, then follow the
# Afghan border north, then the CHN border, then the LoC.
#
# Cutting line: from (74.42, 30.98) to (70.88, 33.99)
# This separates AJK/GB from Punjab+KP approximately.
# Then from (70.88, 33.99) we follow the Afghan border northward.

# AJK+GB capture polygon (clockwise):
ajk_gb_capture = Polygon([
    # Start at LoC bottom (where international border meets LoC)
    (74.42, 30.98),
    # Cut southwest to the Afghan border area to exclude KP
    # The actual PAK-AFG border vertex closest to the KP/AJK divide:
    (70.88, 33.99),
    # Follow PAK-AFG border northward (these are actual PAK vertices)
    (71.16, 34.35),
    (71.12, 34.73),
    (71.61, 35.15),
    (71.50, 35.65),
    (71.26, 36.07),
    (71.85, 36.51),
    (72.92, 36.72),
    (74.07, 36.84),
    (74.58, 37.02),
    # PAK-CHN border
    (75.16, 37.13),
    (75.90, 36.67),
    (76.19, 35.90),
    # Triple junction
    (77.84, 35.49),
    # LoC southward (shared PAK-IND vertices)
    (76.87, 34.65),
    (75.76, 34.50),
    (74.24, 34.75),
    (73.75, 34.32),
    (74.10, 33.44),
    (74.45, 32.76),
    (75.26, 32.27),
    (74.41, 31.69),
    # Back to start
    (74.42, 30.98),
])

pak_kashmir = pak_geom.intersection(ajk_gb_capture)
print(f"PAK Kashmir (AJK+GB) area: {pak_kashmir.area:.4f}")
print(f"PAK Kashmir empty: {pak_kashmir.is_empty}")

# ===== CHN AKSAI CHIN =====
# Aksai Chin is the western extremity of CHN. The CHN-IND shared border (LAC)
# forms its western boundary. It's bounded by:
# - West/South: LAC (shared with IND)
# - North: CHN-PAK border junction → continues into Xinjiang
# - East: CHN mainland
#
# Since Aksai Chin is a thin western protrusion of CHN, we can capture it
# by intersecting CHN with a polygon that goes along the LAC and then
# curves west (into India, where CHN has no territory).

# The LAC vertices (CHN-IND shared, from north to south):
# (77.84, 35.49) -> (78.91, 34.32) -> (78.81, 33.51) -> (79.21, 32.99) 
# -> (79.18, 32.48) -> (78.46, 32.62) -> (78.74, 31.52) -> (79.72, 30.88)
# -> (81.11, 30.18)
#
# Aksai Chin is WEST of these vertices in the CHN polygon.
# We capture it by creating a polygon west of the LAC.

aksai_chin_capture = Polygon([
    # Start at triple junction
    (77.84, 35.49),
    # Go west (into India territory — no CHN here, so intersection will be empty here)
    (74.0, 35.49),
    # Go south 
    (74.0, 29.0),
    # Go east (below the LAC)
    (82.0, 29.0),
    # Go north (east of LAC — but we need to stay west of CHN mainland)
    # Actually, we go up to just past the bottom LAC vertex
    (82.0, 30.18),
    # Now follow the LAC northward (these are exact CHN-IND vertices)
    (81.11, 30.18),
    (79.72, 30.88),
    (78.74, 31.52),
    (78.46, 32.62),
    (79.18, 32.48),
    (79.21, 32.99),
    (78.81, 33.51),
    (78.91, 34.32),
    # Back to triple junction
    (77.84, 35.49),
])

chn_kashmir = chn_geom.intersection(aksai_chin_capture)
print(f"CHN Aksai Chin area: {chn_kashmir.area:.4f}")
print(f"CHN Aksai Chin empty: {chn_kashmir.is_empty}")

# ===== APPLY CORRECTIONS =====
def remove_holes(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def remove_slivers(geom, min_area=0.05):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_area else Polygon()
    elif geom.geom_type == "MultiPolygon":
        valid_polys = [p for p in geom.geoms if p.area >= min_area]
        if not valid_polys:
            return Polygon()
        elif len(valid_polys) == 1:
            return valid_polys[0]
        else:
            return MultiPolygon(valid_polys)
    return geom

world2 = world.copy()
world2.loc[pak_idx[0], "geometry"] = remove_slivers(pak_geom.difference(pak_kashmir))
world2.loc[chn_idx[0], "geometry"] = remove_slivers(chn_geom.difference(chn_kashmir))
world2.loc[ind_idx[0], "geometry"] = remove_slivers(remove_holes(ind_geom.union(pak_kashmir).union(chn_kashmir).buffer(0.005).buffer(-0.005)))

print(f"IND new valid: {world2.loc[ind_idx[0], 'geometry'].is_valid}")
print(f"PAK new valid: {world2.loc[pak_idx[0], 'geometry'].is_valid}")
print(f"CHN new valid: {world2.loc[chn_idx[0], 'geometry'].is_valid}")

# ===== RENDER COMPARISON =====
xmin, xmax = 68, 83
ymin, ymax = 28, 40

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

ax1 = axes[0]
ax1.set_title("BEFORE", fontsize=16, color="white")
ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymin, ymax)
ax1.set_facecolor("#0f0f0f")
for _, row in world.iterrows():
    iso = row.get("iso_a3", "")
    geom = row["geometry"]
    if geom is None: continue
    clipped = geom.intersection(box(xmin, ymin, xmax, ymax))
    if clipped.is_empty: continue
    color = "#1a1a2e"
    if iso == "IND": color = "#FF9933"
    elif iso == "PAK": color = "#00FF00"
    elif iso == "CHN": color = "#FF0000"
    elif iso == "AFG": color = "#6666FF"
    gpd.GeoSeries([clipped]).plot(ax=ax1, color=color, edgecolor="white", linewidth=0.8)

ax2 = axes[1]
ax2.set_title("AFTER (v4 — vertex-aligned)", fontsize=16, color="white")
ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
ax2.set_facecolor("#0f0f0f")
for _, row in world2.iterrows():
    iso = row.get("iso_a3", "")
    geom = row["geometry"]
    if geom is None: continue
    clipped = geom.intersection(box(xmin, ymin, xmax, ymax))
    if clipped.is_empty: continue
    color = "#1a1a2e"
    if iso == "IND": color = "#FF9933"
    elif iso == "PAK": color = "#00FF00"
    elif iso == "CHN": color = "#FF0000"
    elif iso == "AFG": color = "#6666FF"
    gpd.GeoSeries([clipped]).plot(ax=ax2, color=color, edgecolor="white", linewidth=0.8)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "tmp" / "kashmir_v4.png"
fig.savefig(out, dpi=150, facecolor="#0f0f0f")
plt.close(fig)
print(f"Saved to {out}")
