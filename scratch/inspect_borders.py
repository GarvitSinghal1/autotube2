import ssl
import geopandas as gpd
from shapely.geometry import box

ssl._create_default_https_context = ssl._create_unverified_context
url = 'https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip'
world = gpd.read_file(url)
world.columns = [c.lower() for c in world.columns]

# Bounding box of Jammu & Kashmir
jk_box = box(72.0, 32.0, 81.0, 38.0)

for code in ['IND', 'PAK', 'CHN']:
    geom = world[world['iso_a3'] == code].geometry.values[0]
    intersection = geom.intersection(jk_box)
    print(f"\n--- {code} J&K Region Intersection ---")
    if intersection.is_empty:
        print("Empty")
    else:
        print("Type:", intersection.geom_type)
        if intersection.geom_type == 'Polygon':
            print("Vertices count:", len(intersection.exterior.coords))
            print("Coords:", list(intersection.exterior.coords))
        elif intersection.geom_type == 'MultiPolygon':
            for i, p in enumerate(intersection.geoms):
                print(f"  Part {i}: Vertices:", len(p.exterior.coords))
                print(f"  Coords:", list(p.exterior.coords))
