import numpy as np
import pandas as pd

"""Can be applied to a 2D (no z-axis) coord file with lines in the format:
Fp1 -29.4367 83.9171"""

# Load and fix the content
file_path = './standard_1020_tweaked2D.elc' #Removed z co-ords and reformatted.
with open(file_path, 'r') as file:
    content = file.readlines()
    
# Parse electrode data
electrodes = []
coordinates = []
for line in content:
    parts = line.split()
    electrodes.append(parts[0])
    coordinates.append([float(parts[1]), float(parts[2])])

coordinates = np.array(coordinates)

# Normalize coords to [0, 1]
min_coords = coordinates.min(axis=0)
max_coords = coordinates.max(axis=0)
normalized_coords = (coordinates - min_coords) / (max_coords - min_coords)

# Define grid resolution
grid_resolution = (9, 9)

# Scale normalized coordinates and clamp them to grid limits
grid_coords = np.floor(normalized_coords * np.array(grid_resolution)).astype(int)
grid_coords = np.clip(grid_coords, 0, np.array(grid_resolution) - 1)

# Create an empty grid filled with placeholders ('-')
grid = [['-' for _ in range(grid_resolution[1])] for _ in range(grid_resolution[0])]

# Map electrodes to grid
for electrode, (x, y) in zip(electrodes, grid_coords):
    grid[x][y] = electrode

# Display the grid as a nested list
for i, row in enumerate(grid):
    if i == 0:
        print('[', row, ',')
    elif i == len(grid) - 1:
        print(row, ']')
    else:
        print(row, ',')




