# map_ids_to_seeders.py
"""
Merges rockfall path results with their corresponding seeder coordinates.
Assumes:
- Line Seeder rocks come first, then Point Seeder rocks (by ID order)
- Outputs: extracted_data/rocfall_path_with_seeders.csv
"""
import pandas as pd
from pathlib import Path

# File paths
DATA_DIR = Path("extracted_data")
PATH_RESULTS = DATA_DIR / "rocfall_path_results.csv"
LINE_COORDS = DATA_DIR / "rocfall_line_seeder_coords.csv"
POINT_COORDS = DATA_DIR / "rocfall_point_seeder_coords.csv"
OUTPUT = DATA_DIR / "rocfall_path_with_seeders.csv"

# Load data
path_df = pd.read_csv(PATH_RESULTS)
line_df = pd.read_csv(LINE_COORDS)
point_df = pd.read_csv(POINT_COORDS)

# How many rocks per seeder type? (adjust if needed)
num_line = len(line_df)
num_point = len(point_df)


# Assign seeder type and coordinates, but only up to the number of path results
total_paths = len(path_df)
all_coords = pd.concat([line_df, point_df], ignore_index=True)
all_coords = all_coords.rename(columns={"index": "seeder_index", "x": "seeder_x", "y": "seeder_y", "z": "seeder_z"})
all_coords = all_coords.reset_index(drop=True)

# Truncate to match number of path results
coords = all_coords.iloc[:total_paths].copy()
seeder_types = (["line"] * num_line + ["point"] * num_point)[:total_paths]

# Merge by row order (assumes path_df is in correct order)
merged = path_df.copy().reset_index(drop=True)
merged = pd.concat([merged, coords], axis=1)
merged["seeder_type"] = seeder_types

# Save merged CSV
merged.to_csv(OUTPUT, index=False)
print(f"✅ Merged data saved to: {OUTPUT}")
