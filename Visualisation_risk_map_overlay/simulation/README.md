# Simulation (Advanced)

This folder contains a multimodal, terrain-aware simulator that goes beyond the paper-based generator.

## Files
- `advanced_simulator.py`: Core simulator producing terrain features (slope, aspect, curvature, relief, multi-scale roughness) and quick physics-inspired target proxies.
- `run_advanced_simulation.py`: CLI wrapper to generate a CSV.

## Usage
```bash
python -m simulation.run_advanced_simulation --num-cells 50000 --seed 123 --out advanced_rockfall_dataset.csv
```

The output CSV includes original features (`slope_angle`, `slope_roughness`, `seeder_height`) plus additional terrain features, and provisional targets. Environmental/geotechnical time-series and risk labels are placeholders to be added next.
