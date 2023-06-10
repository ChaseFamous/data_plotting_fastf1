import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib import markers
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import numpy as np
import math
from typing import Optional
from track_rotation import get_rotation

def plot_track_map(year: int, circuit: str, driver: str, target_lap: int, session_type:str = "R", sector_markers:bool = False):
    plotting.setup_mpl()

    absolute_path = os.path.abspath('')
    filename = os.path.join(absolute_path, "track_map.png")

    # Variables, used for calculations later on
    time_columns = ['LapStartTime', 'Sector1SessionTime', 'Sector2SessionTime']
    name_columns = ["Start Finish", "Sector 1", "Sector 2"]
    track_map_dic = {"Name": [], "Time": [], "X": [], "Y": [], 'Distance': []}
    
    # Get the rotation for the current circuit to show it in the same
    # orientation as F1 does.
    rotation = get_rotation(circuit)

    # Load the session
    session = ff1.get_session(year, circuit, session_type)
    session.load()

    laps = session.laps
    personal_laps = laps.pick_driver(driver)
    
    # Retrieve laps from session, filter by target lap and selected driver
    if personal_laps['IsAccurate'].loc[personal_laps['LapNumber'] == target_lap].any():
        personal_laps = personal_laps.loc[personal_laps['LapNumber'] == target_lap]
    else:
        lap_number = personal_laps.loc[personal_laps['IsAccurate']].iloc[0]['LapNumber']
        personal_laps = personal_laps.loc[personal_laps['LapNumber'] == lap_number]
        print(f"Lap {target_lap} is an inaccurate lap, chose lap {lap_number.astype(int)} instead")
    

    # Telemetry from the target lap from the selected driver with added distance
    telemetry = personal_laps.get_telemetry().add_distance()

    # Get track sector data
    for column in time_columns:
        track_map_dic["Time"].append(personal_laps[column].iloc[0])
        track_map_dic["X"].append(telemetry['X'].loc[telemetry['SessionTime'] >= personal_laps[column].iloc[0]].iloc[0])
        track_map_dic["Y"].append(telemetry['Y'].loc[telemetry['SessionTime'] >= personal_laps[column].iloc[0]].iloc[0])
        track_map_dic["Distance"].append(
            telemetry['Distance'].loc[telemetry['SessionTime'] >= personal_laps[column].iloc[0]].iloc[0])
    track_map_dic["Name"] = name_columns

    # Add driver tag to telemetry dataframe
    telemetry['Driver'] = driver

    # Calculate the number of sectors and calculate their distance
    NUMBER_OF_MINISECTORS = 50
    total_distance = max(telemetry['Distance'])
    minisector_length = total_distance / NUMBER_OF_MINISECTORS
    minisectors = [0]

    for i in range(0, (NUMBER_OF_MINISECTORS - 1)):
        minisectors.append(minisector_length * (i + 1))

    telemetry['Minisector'] = telemetry['Distance'].apply(
        lambda dist: (
                int((dist // minisector_length) + 1)
        )
    )

    # Create a track map dataframe
    track_map = pd.DataFrame(data=track_map_dic)

    average_speed = telemetry.groupby(['Minisector'])['Speed'].mean() \
        .reset_index()

    fastest_driver = average_speed.loc[average_speed.groupby(['Minisector']) \
                                         ['Speed'].idxmax()]

    telemetry = telemetry.merge(fastest_driver, on=['Minisector'])

    telemetry = telemetry.sort_values(by=['Distance'])

    # Assign integrals to distances
    telemetry.loc[
        (telemetry['Distance'] >= track_map['Distance'].iloc[0]) & (
                telemetry['Distance'] < track_map['Distance'].iloc[1]), 'distance_int'] = 1
    telemetry.loc[
        (telemetry['Distance'] >= track_map['Distance'].iloc[1]) & (
                telemetry['Distance'] < track_map['Distance'].iloc[2]), 'distance_int'] = 2
    telemetry.loc[telemetry['Distance'] >= track_map['Distance'].iloc[2], 'distance_int'] = 3

    x_no_rot = np.array(telemetry['X'].values)
    y_no_rot = np.array(telemetry['Y'].values)

    x_track_no_rot, y_track_no_rot = track_map['X'].values, track_map['Y'].values

    ANGLE = rotation * math.pi  # pi = 180 deg

    x = x_no_rot * math.cos(ANGLE) - y_no_rot * math.sin(ANGLE)
    y = x_no_rot * math.sin(ANGLE) + y_no_rot * math.cos(ANGLE)

    x_track = (x_track_no_rot * math.cos(ANGLE) - y_track_no_rot * math.sin(ANGLE)).reshape(-1, 1, 1)
    y_track = (x_track_no_rot * math.sin(ANGLE) + y_track_no_rot * math.cos(ANGLE)).reshape(-1, 1, 1)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    distance_array = telemetry['distance_int'].to_numpy().astype(float)

    # Create custom color map
    cmap = ListedColormap(["red", "deepskyblue", "yellow"])

    lc_comp = LineCollection(segments, cmap=cmap)
    lc_comp.set_array(distance_array)
    lc_comp.set_linewidth(5)

    plt.rcParams['figure.figsize'] = [18, 10]

    lines = plt.gca().add_collection(lc_comp)
    lines.set_zorder(1)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # Retrieve the colors from lc_comp
    lc_colors = lc_comp.to_rgba(distance_array)

    for i in range(len(segments)):
        x1, y1 = segments[i][0]
        x2, y2 = segments[i][1]
        
        # Get the corresponding color from lc_colors
        color = lc_colors[i]

        # Add a black outline, and filled colored line
        plt.plot([x1, x2], [y1, y2], color='black', linewidth=15, zorder=2, antialiased=True)
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=4, zorder=3, antialiased=True)

        if sector_markers:
            dx = x2 - x1
            dy = y2 - y1

            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 180
            if (x1, y1) in zip(x_track.flatten(), y_track.flatten()):
                plt.annotate("|", (x1, y1), color='white', fontsize=40, fontweight='bold', ha='center', va='center', rotation=angle, zorder=4, bbox=dict(facecolor='none', edgecolor='none', linewidth=200))

    
    plt.suptitle(f"{session.event['EventName']} {session.event.year} Track Map", fontsize=20)

    cbar = plt.colorbar(mappable=lc_comp, ticks=[1.33, 2, 2.66])
    cbar.set_ticklabels(['Sector 1', 'Sector 2', 'Sector 3'])

    plt.savefig(filename, dpi=320)

    return filename
