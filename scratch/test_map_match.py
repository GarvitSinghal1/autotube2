import pandas as pd
import ssl
import geopandas as gpd
from shapely.geometry import Polygon
from pipeline.modules.renderer_short import _load_world_geometry, _match_entity_to_iso

world = _load_world_geometry()
if world is None:
    print("world geometry is NONE!")
else:
    print("world geometry loaded successfully, rows:", len(world))
    
    # Test J&K polygon modification via intersection to prevent gaps
    kashmir_boundary_coords = [
        (73.4, 32.0),  # Bottom-left near international border
        (73.4, 34.3),  # Western edge of Azad Kashmir
        (72.9, 35.0),  # Western edge of Gilgit-Baltistan
        (72.9, 37.2),  # Northwest corner of Gilgit-Baltistan
        (80.5, 37.2),  # Northeast corner of Aksai Chin
        (80.5, 32.0),  # Southeast corner of Ladakh
        (73.4, 32.0)   # Close
    ]
    kashmir_boundary = Polygon(kashmir_boundary_coords)
    
    try:
        pak_idx = world[world["iso_a3"] == "PAK"].index
        chn_idx = world[world["iso_a3"] == "CHN"].index
        ind_idx = world[world["iso_a3"] == "IND"].index
        
        if not pak_idx.empty and not chn_idx.empty and not ind_idx.empty:
            pak_geom = world.loc[pak_idx[0], "geometry"]
            chn_geom = world.loc[chn_idx[0], "geometry"]
            ind_geom = world.loc[ind_idx[0], "geometry"]
            
            # Extract the actual Kashmir parts from PAK and CHN
            pak_kashmir = pak_geom.intersection(kashmir_boundary)
            chn_kashmir = chn_geom.intersection(kashmir_boundary)
            
            # Subtract from PAK and CHN
            world.loc[pak_idx[0], "geometry"] = pak_geom.difference(pak_kashmir)
            world.loc[chn_idx[0], "geometry"] = chn_geom.difference(chn_kashmir)
            
            # Union into India
            world.loc[ind_idx[0], "geometry"] = ind_geom.union(pak_kashmir).union(chn_kashmir)
            
            print("Successfully executed intersection-based J&K correction.")
            print("Checking validity of new geometries:")
            print("India valid:", world.loc[ind_idx[0], "geometry"].is_valid)
            print("Pakistan valid:", world.loc[pak_idx[0], "geometry"].is_valid)
            print("China valid:", world.loc[chn_idx[0], "geometry"].is_valid)
            
    except Exception as e:
        print("Error during geometry modification:", e)
