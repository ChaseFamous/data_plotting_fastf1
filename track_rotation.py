import pandas as pd

rotations = pd.read_json("track_rotation.json")
rotations['country'] = rotations['country'].str.lower()
rotations['raceName'] = rotations['raceName'].str.lower()
rotations['circuitId'] = rotations['circuitId'].str.lower()

def get_rotation(gp: str):
    gp = gp.casefold()
    if gp in rotations['country'].values:
        rotation = rotations.loc[rotations['country'] == gp, 'rotation']
    elif gp in rotations['raceName'].values:
        rotation = rotations.loc[rotations['raceName'] == gp, 'rotation']
    elif gp in rotations['circuitId'].values:
        rotation = rotations.loc[rotations['circuitId'] == gp, 'rotation']
    else:
        rotation = 0  # Set a default value or custom error message
        print("Circuitname {gp} was not found")
    
    return rotation