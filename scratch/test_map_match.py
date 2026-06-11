import pandas as pd
from pipeline.modules.renderer_short import _load_world_geometry, _match_entity_to_iso

world = _load_world_geometry()
if world is None:
    print("world geometry is NONE!")
else:
    print("world geometry loaded successfully, rows:", len(world))
    entities = ['United States', 'China', 'India', 'Germany', 'Brazil']
    for ent in entities:
        iso = _match_entity_to_iso(ent, world)
        print(f"Entity: {ent} -> ISO: {iso}")
