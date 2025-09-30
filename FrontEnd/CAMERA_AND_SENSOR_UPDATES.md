# Camera and Sensor Position Updates

## Changes Made

### 1. Camera Zoom Adjustments
The camera has been zoomed out to provide a better overview of the entire Bingham Canyon mine model.

#### Updated Camera Settings:
- **Initial Position**: Changed from (50, 40, 50) to (100, 80, 100)
- **Default Distance**: Changed from 80 to 150 units
- **Zoom Range**: Extended from 10-150 to 50-300 units
- **Zoom Steps**: Increased from 5 to 10 units per zoom action

These changes provide:
- Better overview of the entire mine pit
- Clearer view of all sensors at once
- More comfortable viewing distance for the large-scale model

### 2. Sensor Ground-Level Positioning
All sensors have been repositioned to sit at ground level (Y=0) rather than floating above or below the surface.

#### Updated Sensor Positions:
All sensors now have Y-coordinate of 0 or 1 (for elevated positions like weather stations):
- **Radar sensors**: Y=0 (ground level)
- **Piezometers**: Y=0 (ground level)
- **Extensometers**: Y=0 (ground level)
- **Seismometers**: Y=0 (ground level)
- **Weather stations**: Y=1 (slightly elevated)

This applies to both:
- Mine Overview page sensors
- Hazard Prediction page sensors

### 3. Heatmap Ground-Level Positioning
Risk visualization heatmaps have been adjusted to match the ground-level positioning.

#### Updated Heatmap Zones:
- **Slope stability zones**: All positioned at Y=0
- **Rockfall risk zones**: All positioned at Y=0

This ensures:
- Heatmaps align with sensor positions
- Risk zones appear on the mine surface
- Better visual consistency across the 3D model

## Visual Improvements
These changes result in:
1. **Better Overview**: The zoomed-out camera provides a complete view of the Bingham Canyon mine
2. **Realistic Positioning**: Sensors appear to be placed on the actual mine surface
3. **Consistent Visualization**: All elements (sensors, heatmaps) align at ground level
4. **Professional Appearance**: The positioning matches real-world sensor deployment

## Testing the Changes
1. Navigate to the Mine Overview page to see all sensors at ground level
2. Go to Hazard Prediction page to see sensors and heatmaps properly positioned
3. Use zoom controls to adjust view as needed (50-300 unit range)
4. Rotate the view to confirm all sensors are on the surface

The dashboard now provides a more realistic and professional visualization of the Bingham Canyon mine with properly positioned monitoring equipment.
