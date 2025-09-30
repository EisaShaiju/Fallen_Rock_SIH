"""
Advanced Rockfall Simulator

Generates multimodal synthetic data beyond paper constraints, including:
- Terrain features derived from DEM-like parameters (slope, aspect, curvature, relief, roughness at scales)
- Environmental time series (rainfall, temperature, freeze–thaw, vibrations) [placeholder]
- Geotechnical sensor streams (displacement, pore pressure, strain rates) [placeholder]
- Risk labels: event probability in horizon and severity proxy [placeholder]

Outputs a tabular dataset suitable for model training and a time index for future temporal extensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TerrainParams:
    num_cells: int = 20000
    slope_deg_mean: float = 55.0
    slope_deg_std: float = 12.0
    slope_deg_min: float = 15.0
    slope_deg_max: float = 85.0
    roughness_deg_mean: float = 10.0
    roughness_deg_std: float = 5.0
    roughness_deg_min: float = 0.5
    roughness_deg_max: float = 30.0
    relief_m_mean: float = 30.0
    relief_m_std: float = 15.0
    relief_m_min: float = 1.0
    relief_m_max: float = 150.0
    seeder_height_min: float = 3.0
    seeder_height_max: float = 60.0


class AdvancedRockfallSimulator:
    """High-level simulator for generating multimodal synthetic rockfall data."""

    def __init__(self, random_state: int = 42, terrain: TerrainParams | None = None) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.terrain = terrain or TerrainParams()

    def generate_terrain(self) -> pd.DataFrame:
        """Generate DEM-derived terrain features at cell/segment level."""
        n = self.terrain.num_cells

        slope = self.rng.normal(self.terrain.slope_deg_mean, self.terrain.slope_deg_std, n)
        slope = np.clip(slope, self.terrain.slope_deg_min, self.terrain.slope_deg_max)

        roughness = self.rng.normal(self.terrain.roughness_deg_mean, self.terrain.roughness_deg_std, n)
        roughness = np.clip(roughness, self.terrain.roughness_deg_min, self.terrain.roughness_deg_max)

        relief = self.rng.normal(self.terrain.relief_m_mean, self.terrain.relief_m_std, n)
        relief = np.clip(relief, self.terrain.relief_m_min, self.terrain.relief_m_max)

        seeder_height = self.rng.uniform(self.terrain.seeder_height_min, self.terrain.seeder_height_max, n)

        # Aspect uniformly distributed (0-360), encode as sin/cos to avoid discontinuity
        aspect_deg = self.rng.uniform(0.0, 360.0, n)
        aspect_sin = np.sin(np.radians(aspect_deg))
        aspect_cos = np.cos(np.radians(aspect_deg))

        # Curvature proxy: mean curvature ~ small around 0, heavier tails for convex/concave
        curvature = self.rng.normal(0.0, 0.05, n)

        # Multi-scale roughness (windowed std proxies)
        roughness_s = roughness
        roughness_m = np.clip(roughness * self.rng.normal(1.1, 0.1, n), 0.0, None)
        roughness_l = np.clip(roughness * self.rng.normal(1.25, 0.15, n), 0.0, None)

        # Simple physics-inspired targets (coarse proxies) for quick smoke tests
        g = 9.81
        mass = 1.0
        potential_energy = mass * g * seeder_height
        roughness_loss = 1.0 - np.clip(np.radians(roughness_s) / np.pi * 0.35, 0.0, 0.8)
        restitution = np.clip(self.rng.normal(0.7, 0.2, n), 0.3, 1.0)
        kinetic_energy = potential_energy * roughness_loss * (0.2 + 0.8 * restitution)

        alpha = np.radians(np.clip(slope, 1.0, None))
        base_distance = seeder_height / np.tan(alpha)
        scatter = self.rng.normal(0.0, np.radians(np.clip(roughness_m, 0.1, None)) * 2.0, n)
        impact_position = np.maximum(base_distance + scatter, 0.0)

        velocity = np.sqrt(np.maximum(2.0 * kinetic_energy / mass, 0.0))
        phi = np.clip(self.rng.normal(30.0, 3.0, n), 20.0, 40.0)
        runout = (velocity ** 2) / (2.0 * g * np.tan(np.radians(phi)))
        runout *= 1.0 + (np.radians(np.clip(slope, 15.0, 85.0)) - np.radians(45.0)) / np.radians(40.0) * 0.4
        runout *= self.rng.normal(1.0, 0.15, n)

        df = pd.DataFrame({
            # Core original features
            'slope_angle': slope,
            'slope_roughness': roughness_s,
            'seeder_height': seeder_height,
            # New terrain features
            'aspect_sin': aspect_sin,
            'aspect_cos': aspect_cos,
            'curvature': curvature,
            'local_relief': relief,
            'roughness_m': roughness_m,
            'roughness_l': roughness_l,
            # Quick target proxies (optional for early training)
            'kinetic_energy': kinetic_energy,
            'impact_position': impact_position,
            'runout_distance': np.maximum(runout, 0.0),
        })

        return df

    def generate_environment_features(self, n: int) -> pd.DataFrame:
        """Simulate environmental features as recent-history aggregates.

        We avoid storing long time series per cell by sampling plausible
        aggregate statistics directly.
        """
        # Rainfall: use gamma-like distribution for daily amounts and accumulate
        daily_rain_mean = self.rng.gamma(shape=2.0, scale=3.0, size=n)  # mm/day climatology
        storminess = np.clip(self.rng.normal(1.0, 0.3, size=n), 0.4, 2.0)
        r1 = np.clip(self.rng.gamma(1.2 * storminess, 4.0, n), 0.0, None)
        r3 = np.clip(self.rng.gamma(2.0 * storminess, 5.0, n), 0.0, None)
        r7 = np.clip(self.rng.gamma(3.0 * storminess, 6.0, n), 0.0, None)
        r30 = np.clip(self.rng.gamma(5.0 * storminess, 8.0, n), 0.0, None)

        # Antecedent Precipitation Index (API) approximations
        api7 = 0.6 * r7 + 0.3 * r3 + 0.1 * r1
        api30 = 0.5 * r30 + 0.3 * r7 + 0.15 * r3 + 0.05 * r1

        # Temperature and freeze–thaw cycles
        temp_mean = self.rng.normal(8.0, 7.0, n)  # °C
        temp_amp = np.clip(self.rng.normal(6.0, 3.0, n), 1.0, 20.0)
        tmin7 = temp_mean - temp_amp * 0.7
        tmax7 = temp_mean + temp_amp * 0.7
        # Approximate freeze-thaw cycles: crossings around 0°C grow with amplitude near freezing
        freeze_factor = np.clip((0.0 - temp_mean) / 5.0, -2.0, 2.0)
        freeze_thaw_7 = np.clip((2.0 + freeze_factor) * (temp_amp / 6.0), 0.0, None)

        # Vibration proxy: blasting/operations intensity
        vib_events_7d = np.clip(self.rng.poisson(lam=5.0 * storminess), 0, None)
        vib_rms_24h = np.clip(self.rng.normal(0.02 * (1.0 + vib_events_7d / 7.0), 0.01, n), 0.0, None)

        return pd.DataFrame({
            'rain_1d_mm': r1,
            'rain_3d_mm': r3,
            'rain_7d_mm': r7,
            'rain_30d_mm': r30,
            'api_7d': api7,
            'api_30d': api30,
            'temp_mean_7d_c': temp_mean,
            'temp_min_7d_c': tmin7,
            'temp_max_7d_c': tmax7,
            'freeze_thaw_7d': freeze_thaw_7,
            'vibration_events_7d': vib_events_7d,
            'vibration_rms_24h': vib_rms_24h,
        })

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate dataset and return with metadata about placeholders."""
        terrain_df = self.generate_terrain()
        env_df = self.generate_environment_features(len(terrain_df))
        df = pd.concat([terrain_df, env_df], axis=1)

        # Geotechnical sensor features (displacement, pore pressure, strain rates)
        # We simulate recent trends and volatility as aggregates
        n = len(df)
        disp_rate_mm_day = np.clip(self.rng.lognormal(mean=0.0, sigma=0.6, size=n) - 0.5, 0.0, None)
        disp_accel_mm_day2 = np.clip(self.rng.normal(0.0, 0.1, size=n) + 0.02 * (disp_rate_mm_day - 0.5), -0.5, 1.0)
        pore_pressure_kpa = np.clip(self.rng.normal(50.0 + 0.2 * df['rain_7d_mm'].to_numpy(), 8.0, size=n), 10.0, 300.0)
        # Relate pore trend to API directly
        pore_trend_kpa_day = np.clip(self.rng.normal(0.0, 0.8, size=n) + 0.002 * df['api_7d'].to_numpy(), -3.0, 5.0)
        strain_rate_micro = np.clip(self.rng.lognormal(mean=-1.0, sigma=0.7, size=n), 0.0, None)

        df['disp_rate_mm_day'] = disp_rate_mm_day
        df['disp_accel_mm_day2'] = disp_accel_mm_day2
        df['pore_pressure_kpa'] = pore_pressure_kpa
        df['pore_trend_kpa_day'] = pore_trend_kpa_day
        df['strain_rate_micro'] = strain_rate_micro

        # Risk labels: combine factors into an event probability and severity proxy
        # This is a heuristic mapping; users can refine with site-specific priors.
        norm = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
        drivers = {
            'api': norm(df['api_7d'].to_numpy()) * 0.25 + norm(df['api_30d'].to_numpy()) * 0.15,
            'slope': norm(df['slope_angle'].to_numpy()) * 0.2,
            'rough': norm(df['slope_roughness'].to_numpy()) * 0.1,
            'disp': norm(df['disp_rate_mm_day'].to_numpy()) * 0.15 + norm(df['disp_accel_mm_day2'].to_numpy()) * 0.05,
            'pore': norm(df['pore_pressure_kpa'].to_numpy()) * 0.05 + norm(df['pore_trend_kpa_day'].to_numpy()) * 0.03,
            'vib': norm(df['vibration_events_7d'].to_numpy()) * 0.02 + norm(df['vibration_rms_24h'].to_numpy()) * 0.05,
        }
        driver_sum = sum(drivers.values())
        # Map to probability via logistic
        logits = 2.0 * (driver_sum - 0.8)
        prob_event_7d = 1.0 / (1.0 + np.exp(-logits))

        # Severity proxy: combine expected runout and kinetic energy contribution
        severity = norm(df['runout_distance'].to_numpy()) * 0.6 + norm(df['kinetic_energy'].to_numpy()) * 0.4

        df['risk_prob_7d'] = prob_event_7d
        df['risk_severity'] = severity

        metadata = {
            'note_env': 'Environmental aggregates simulated (rainfall, API, temp, freeze-thaw, vibration).',
            'note_geotech': 'Geotechnical aggregates simulated (displacement rate, acceleration, pore pressure, strain rate).',
            'labels': 'Risk labels computed: risk_prob_7d (0-1) and risk_severity (0-1 proxy).'
        }

        return df, metadata


def main() -> Tuple[pd.DataFrame, Dict[str, str]]:
    sim = AdvancedRockfallSimulator(random_state=42)
    df, meta = sim.generate()
    out_csv = 'advanced_rockfall_dataset.csv'
    df.to_csv(out_csv, index=False)
    print(f"Generated dataset: {out_csv} with shape {df.shape}")
    for k, v in meta.items():
        print(f"{k}: {v}")
    return df, meta


if __name__ == '__main__':
    main()


