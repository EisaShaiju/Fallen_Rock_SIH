# 🚀 Advanced Rockfall Prediction System for Open-Pit Mines

## 📋 Project Overview

This project implements a comprehensive machine learning system for predicting rockfall hazards in open-pit mining environments. The system consists of three main components that work together to provide real-time risk assessment and visualization:

1. **🤖 Machine Learning Component** - Advanced risk prediction models
2. **🌐 Interactive Dashboard** - Real-time monitoring and visualization interface  
3. **🗺️ Spatial Interpolation** - Risk mapping and 3D visualization overlay

The system addresses the critical need for proactive safety measures by providing accurate predictions of rockfall trajectories, impact energies, runout distances, and overall risk assessment for mining operations.

## 🏗️ System Architecture

### Core Components

```
rs2/
├── 🤖 Machine Learning (ML) Component
│   ├── rockfall_data_generator.py          # Physics-based synthetic data generation
│   ├── advanced_risk_models.joblib         # Trained ML models
│   └── Visualisation_risk_map_overlay/
│       ├── training/
│       │   ├── train.py                    # Advanced risk model training
│       │   └── predict.py                  # Risk prediction inference
│       └── simulation/
│           ├── advanced_simulator.py       # Multi-modal terrain simulation
│           └── run_advanced_simulation.py  # CLI simulation runner
│
├── 🌐 Interactive Dashboard
│   └── dashboard/mine-dashboard/
│       ├── app/                            # Next.js application
│       ├── components/                     # React components
│       │   ├── hazard-prediction.tsx       # Risk prediction interface
│       │   ├── mine-viewer-3d.tsx         # 3D mine visualization
│       │   └── real-time-alerts.tsx       # Alert management
│       └── utils/loadMineModel.ts          # 3D model loading utilities
│
└── 🗺️ Spatial Interpolation & Visualization
    └── Visualisation_risk_map_overlay/
        ├── simulation/
        │   ├── create_mine_risk_heatmap.py # Risk heatmap generation
        │   ├── overlay_risk_on_mesh_plotly.py # 3D risk overlay
        │   └── visualize_risk.py           # Risk visualization tools
        └── outputs/                        # Generated visualizations
```

## 🎯 Key Features

### 🤖 Machine Learning Component

**Advanced Risk Prediction Models:**
- **Risk Probability (7-day)**: Predicts rockfall probability over 7-day horizon
- **Risk Severity**: Estimates potential impact severity
- **Multiple Algorithms**: Ridge Regression, Random Forest, Gradient Boosting
- **Performance**: R² scores of 0.65-0.99 across all targets

**Input Features:**
- **Terrain Features**: Slope angle, roughness, aspect, curvature, relief
- **Environmental Data**: Rainfall, temperature, freeze-thaw cycles, vibrations
- **Geotechnical Sensors**: Displacement rates, pore pressure, strain rates
- **Historical Data**: Antecedent precipitation indices, trend analysis

**Output Predictions:**
- **Kinetic Energy**: Energy at first impact (20-593J)
- **Impact Position**: Distance from wall toe (0-29m)
- **Runout Distance**: Total travel distance (3-191m)
- **Risk Metrics**: Probability and severity assessments

### 🌐 Interactive Dashboard

**Real-time Monitoring Interface:**
- **3D Mine Visualization**: Interactive Bingham Canyon Mine model
- **Live Sensor Data**: Real-time displacement, pressure, strain monitoring
- **Risk Heatmaps**: Dynamic risk overlay on 3D terrain
- **Alert Management**: Automated warning system with severity levels
- **Forecast Visualization**: Multi-horizon predictions (6h, 24h, 7d)

**Dashboard Components:**
- **Mine Overview**: Operational status and key metrics
- **Hazard Prediction**: Risk assessment with interactive 3D model
- **Alerts & Forecasts**: Real-time notifications and trend analysis
- **Sensor Management**: Device status and data visualization

### 🗺️ Spatial Interpolation & Visualization

**Advanced Visualization Tools:**
- **Risk Heatmaps**: Spatial distribution of rockfall risk
- **3D Risk Overlay**: Interactive terrain with risk visualization
- **Mesh Integration**: Risk data overlaid on mine geometry
- **Animation Support**: Time-series risk evolution
- **Export Capabilities**: HTML, PNG, and data export formats

**Visualization Features:**
- **Interactive Plotly**: Web-based 3D visualizations
- **Risk Color Mapping**: Intuitive color-coded risk levels
- **Multi-scale Analysis**: Different roughness scales
- **Coordinate Systems**: Mine-specific coordinate handling

## 📊 Performance Metrics

### Model Performance
- **Risk Probability (7d)**: R² = 0.74 (Ridge Regression)
- **Risk Severity**: R² = 0.65 (Random Forest)
- **Kinetic Energy**: R² = 0.74 (Ridge Regression)
- **Impact Position**: R² = 0.99 (Random Forest)
- **Runout Distance**: R² = 0.65 (Ridge Regression)

### System Capabilities
- **Dataset Size**: 50,000+ synthetic rockfall simulations
- **Feature Count**: 20+ terrain, environmental, and geotechnical features
- **Real-time Processing**: Sub-second prediction latency
- **3D Visualization**: Interactive Bingham Canyon Mine model
- **Multi-horizon Forecasting**: 6h, 24h, and 7-day predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Modern web browser with WebGL support

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd rs2

# Set up Python environment
python3 -m venv rockfall_env
source rockfall_env/bin/activate  # On Windows: rockfall_env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Machine Learning Component

```bash
# Generate synthetic dataset
python rockfall_data_generator.py

# Train advanced risk models
cd Visualisation_risk_map_overlay/training
python train.py --data ../data/advanced_rockfall_dataset.csv

# Run advanced simulation
cd ../simulation
python run_advanced_simulation.py --num-cells 50000 --out advanced_rockfall_dataset.csv

# Make predictions
cd ../training
python predict.py --target risk_prob_7d --csv ../data/advanced_rockfall_dataset.csv --out predictions.csv
```

### 3. Interactive Dashboard

```bash
# Navigate to dashboard directory
cd dashboard/mine-dashboard

# Install dependencies
npm install

# Start development server
npm run dev

# Open browser to http://localhost:3000
```

### 4. Spatial Visualization

```bash
# Generate risk heatmaps
cd Visualisation_risk_map_overlay/simulation
python create_mine_risk_heatmap.py

# Create 3D risk overlay
python overlay_risk_on_mesh_plotly.py

# Generate interactive visualizations
python visualize_risk.py
```

## 🔧 Configuration

### Machine Learning Models

**Model Configuration** (`training/train.py`):
```python
models = {
    'ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42)),
    ]),
    'random_forest': RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_split=4
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05
    ),
}
```

**Simulation Parameters** (`simulation/advanced_simulator.py`):
```python
@dataclass
class TerrainParams:
    num_cells: int = 20000
    slope_deg_mean: float = 55.0
    slope_deg_std: float = 12.0
    relief_m_mean: float = 30.0
    # ... additional parameters
```

### Dashboard Configuration

**3D Model Settings** (`utils/loadMineModel.ts`):
```typescript
// Model scaling and positioning
model.scale.setScalar(0.1)
model.position.set(0, -5, 0)

// Camera configuration
camera.position.set(50, 40, 50)
camera.lookAt(0, 0, 0)
```

**Sensor Configuration** (`components/hazard-prediction.tsx`):
```typescript
const sensors: EnhancedSensor[] = [
  {
    id: "RDR-001",
    name: "Slope Monitoring Radar",
    type: "radar",
    position: [-15, 8, 10],
    // ... sensor configuration
  }
]
```

## 📈 Usage Examples

### Machine Learning Prediction

```python
from Visualisation_risk_map_overlay.training.predict import predict
import joblib

# Load trained models
bundle = joblib.load('../outputs/advanced_risk_models.joblib')

# Make prediction
prediction = predict(
    bundle=bundle,
    target='risk_prob_7d',
    X=feature_dataframe
)

print(f"Risk Probability: {prediction[0]:.3f}")
```

### Dashboard Integration

```typescript
// Real-time risk monitoring
const riskData = {
  probability: 0.34,
  trend: "increasing",
  riskLevel: "moderate",
  confidence: 82,
  lastCalculated: "3 minutes ago"
}

// 3D visualization with risk overlay
<MineViewer3D 
  sensors={sensors} 
  showHeatmap={true} 
  heatmapType="rockfall" 
/>
```

### Spatial Visualization

```python
# Generate risk heatmap
from simulation.create_mine_risk_heatmap import create_risk_heatmap

heatmap = create_risk_heatmap(
    risk_data=prediction_results,
    coordinates=mine_coordinates,
    output_path='mine_risk_heatmap.html'
)
```

## 🔬 Research Foundation

This project is based on peer-reviewed research:

> **"Prediction of rockfall hazard in open pit mines using a regression based machine learning model"**  
> by I.P. Senanayake et al. (2024)

### Key Research Insights
- **15 highwalls** from 7 open-pit coal mines analyzed
- **4,550,500 trajectory simulations** performed
- **Multi-linear regression** best for kinetic energy prediction
- **Multi-non-linear regression** best for position/runout prediction

### Enhanced Features Beyond Research
- **Environmental integration**: Weather, temperature, freeze-thaw cycles
- **Geotechnical sensors**: Displacement, pore pressure, strain monitoring
- **Real-time processing**: Live sensor data integration
- **3D visualization**: Interactive mine model with risk overlay
- **Multi-horizon forecasting**: Short to medium-term predictions

## 📁 Generated Files

### Data Files
- `rockfall_dataset.csv`: 50,000 synthetic rockfall simulations
- `advanced_rockfall_dataset.csv`: Enhanced multi-modal dataset
- `advanced_risk_models.joblib`: Trained ML models
- `mine_coordinates.csv`: Spatial coordinate data
- `mine_risk_points.csv`: Risk assessment results

### Visualizations
- `mine_mesh_risk.html`: Interactive 3D risk visualization
- `mine_risk_heatmap.html`: 2D risk heatmap
- `adv_risk_prob_7d_pred.png`: Risk probability predictions
- `adv_risk_severity_featimp.png`: Feature importance analysis
- `adv_correlation.png`: Feature correlation matrix

### Documentation
- `Rockfall_Prediction_System_Report.pdf`: Comprehensive technical report
- `BINGHAM_CANYON_INTEGRATION.md`: 3D model integration guide
- `CAMERA_AND_SENSOR_UPDATES.md`: Sensor system documentation

## 🚀 Future Enhancements

### Short-term Improvements
- **Enhanced Features**: Weather API integration, geological data
- **Advanced Models**: Deep learning, ensemble methods, time-series models
- **Real-time Integration**: IoT sensors, drone data, satellite imagery
- **Mobile Support**: Responsive dashboard for field operations

### Medium-term Goals
- **Multi-mine Support**: Scalable system for multiple mining operations
- **Predictive Maintenance**: Equipment failure prediction
- **Automated Alerts**: SMS/email notifications, emergency protocols
- **API Development**: RESTful API for third-party integrations

### Long-term Vision
- **Autonomous Monitoring**: AI-driven risk assessment and response
- **Digital Twin Integration**: Real-time mine digital twin updates
- **Industry Standard**: Open-source framework for mining safety
- **Global Deployment**: Multi-language, multi-currency support

## 🤝 Contributing

We welcome contributions to improve the rockfall prediction system:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**: Follow coding standards and add tests
4. **Submit a pull request**: Include detailed description of changes

### Development Guidelines
- **Code Style**: Follow PEP 8 for Python, ESLint for TypeScript
- **Testing**: Add unit tests for new features
- **Documentation**: Update README and inline comments
- **Performance**: Optimize for real-time processing requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Foundation**: Based on work by Senanayake et al. (2024)
- **3D Model**: Bingham Canyon Mine model for realistic visualization
- **Open Source Libraries**: scikit-learn, Three.js, Next.js, Plotly
- **Mining Industry**: Collaboration with mining operations for validation

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: [Contact Information](mailto:contact@example.com)

---

**⚠️ Safety Notice**: This system is designed for research and educational purposes. For production use in mining operations, ensure proper validation, testing, and integration with existing safety protocols.
