# Bingham Canyon Mine 3D Model Integration

## Overview
The dashboard now uses the real Bingham Canyon Mine 3D model instead of the procedural mine pit. This provides a more realistic and accurate representation of an actual mining operation.

## Model Details
- **Source**: Bingham Canyon Mine, Utah (Kennecott Copper Mine)
- **File**: `800004c9-0001-f500-b63f-84710c7967bb.glb`
- **Format**: GLB (GLTF Binary)
- **Location**: `/public/the-bingham-canyon-mine-utah/source/`

## Changes Made

### 1. Package Dependencies
- Added `@types/three` for TypeScript support
- GLTFLoader is imported from Three.js examples

### 2. Model Loading (`utils/loadMineModel.ts`)
- Updated to use GLTFLoader for loading GLB files
- Added `loadBinghamCanyonMine()` function for easy model loading
- Includes error handling and fallback to procedural model
- Proper scaling and positioning for the model

### 3. 3D Viewer Updates (`components/mine-viewer-3d.tsx`)
- **Camera positioning**: Adjusted for larger model (50, 40, 50)
- **Camera distance**: Increased from 40 to 80 units
- **Zoom limits**: Extended from 5-30 to 10-150 units
- **Sensor scaling**: 2x scale factor for sensors and heatmaps
- **Sensor positioning**: Scaled by 2x to match model size

### 4. File Structure
```
public/
└── the-bingham-canyon-mine-utah/
    ├── source/
    │   └── 800004c9-0001-f500-b63f-84710c7967bb.glb
    └── textures/
        └── gltf_embedded_0.jpeg
```

## Features
- **Real-time loading**: Model loads automatically when the 3D viewer initializes
- **Error handling**: Falls back to procedural model if loading fails
- **Loading states**: Shows loading spinner during model load
- **Scaled sensors**: All sensor markers and heatmaps are properly scaled
- **Interactive controls**: All camera controls work with the new model

## Usage
The Bingham Canyon model is now the default model for all 3D views in the dashboard:
- Mine Overview page
- Hazard Prediction page
- Any component using `MineViewer3D`

## Customization
To adjust the model:
1. **Scale**: Modify `model.scale.setScalar(0.1)` in `loadMineModel.ts`
2. **Position**: Adjust `model.position.set(0, -5, 0)` in `loadMineModel.ts`
3. **Camera**: Update camera position and distance in `mine-viewer-3d.tsx`
4. **Sensors**: Modify scale factors in sensor creation functions

## Performance Notes
- The GLB model is optimized for web use
- Loading time depends on model size and network speed
- Shadows and lighting are properly configured for the model
- All animations and interactions work seamlessly

## Troubleshooting
If the model doesn't load:
1. Check browser console for errors
2. Verify file paths in `public/` directory
3. Ensure GLTFLoader is properly imported
4. Check network connectivity for model loading
