import os
Path_to_models = "/run/media/riccardodandrea/Ricca_Data/hub/models--lykon--dreamshaper-7/snapshots/9b481047f77996efa025e75e03941dbf51f506ad"
if not os.path.exists(Path_to_models):
    print(f"❌ Path does not exist: {Path_to_models}")
else:
    print(f"✅ Path exists: {Path_to_models}")
