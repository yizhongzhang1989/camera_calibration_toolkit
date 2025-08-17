"""
Enhanced Calibration Pattern System Documentation
=================================================

This document describes the enhanced calibration pattern abstraction that supports 
both 2D planar patterns and 3D spatial patterns for camera calibration.

## Overview

The system has been redesigned to support diverse calibration scenarios beyond 
traditional flat chessboards, including:

- 2D planar patterns (traditional chessboards, ChArUco boards)
- 3D spatial patterns (markers on 3D objects, multi-planar setups)
- Mixed calibration scenarios

## Architecture

### Base Classes

1. **CalibrationPattern** (Abstract Base Class)
   - Unified interface for all calibration patterns
   - Supports both planar (z=0) and 3D spatial patterns
   - Handles corner detection with optional point IDs
   - Generates appropriate 3D object points

### Key Interface Changes

```python
# New interface supports point IDs for non-sequential detection
def detect_corners(self, image) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    return success, image_points, point_ids

# Object point generation can filter by detected point IDs  
def generate_object_points(self, point_ids=None) -> np.ndarray:
    return object_points_3d
```

### Pattern Types

1. **StandardChessboard**
   - Traditional black/white checkerboard
   - 2D planar (z=0)
   - Sequential corner detection (no point IDs needed)

2. **CharucoBoard** 
   - ChArUco boards with ArUco markers
   - Supports both 2D planar and 3D spatial configurations
   - Non-sequential detection with corner IDs
   - Cross-version OpenCV compatibility

3. **Custom3DPattern**
   - Arbitrary 3D calibration objects
   - User-defined 3D coordinates
   - Custom feature detection functions
   - Suitable for complex calibration rigs

## Usage Examples

### Creating Patterns

```python
# Standard 2D chessboard
standard_pattern = create_chessboard_pattern(
    "standard", width=11, height=8, square_size=0.020
)

# ChArUco board (can be 2D or 3D)
charuco_pattern = create_chessboard_pattern(
    "charuco", width=5, height=7, 
    square_size=0.040, marker_size=0.020,
    is_planar=True  # or False for 3D
)

# Custom 3D calibration object
def my_detector(image, **kwargs):
    # Custom detection logic here
    return success, image_points, point_ids

cube_pattern = Custom3DPattern(
    pattern_id="my_cube",
    name="Cubic Calibration Object", 
    object_points_3d=known_3d_coordinates,
    feature_detector=my_detector
)
```

### Calibration

```python
calibrator = IntrinsicCalibrator()

# Use with any pattern type
success, mtx, dist = calibrator.calibrate_with_pattern(
    image_paths=images,
    calibration_pattern=pattern,
    distortion_model='standard',
    verbose=True
)
```

## Key Features

### 1. Pattern Type Detection
- `pattern.is_planar` - True for 2D patterns, False for 3D
- `pattern.is_3d_pattern()` - Check if pattern has 3D structure
- `pattern.get_info()` - Comprehensive pattern information

### 2. Flexible Object Points
- Planar patterns: Generate once, use for all images
- 3D patterns: Generate based on detected features per image
- Point ID filtering: Support partial feature detection

### 3. Cross-Pattern Compatibility
- Unified calibration interface for all pattern types
- Backward compatibility with legacy methods
- Consistent result format and metadata

### 4. OpenCV Version Support
- Handles both old and new ArUco API versions
- Graceful fallbacks for missing features
- Cross-platform compatibility

## 3D Calibration Benefits

### Advantages
- **Robust occlusion handling**: Partial feature detection supported
- **Better geometry**: 3D constraints provide stronger calibration
- **Fewer images needed**: Rich 3D information per image
- **Tight spaces**: No need for large flat calibration targets
- **Complex scenarios**: Multi-camera rigs, confined environments

### Challenges
- **Precise coordinates**: Requires accurate 3D measurements
- **Custom detection**: Need specialized feature detection algorithms
- **Manufacturing**: Tolerances affect calibration accuracy
- **Complexity**: More complex setup and validation

## Implementation Examples

### Cubic Calibration Object
```python
# 50mm cube with markers on three faces
cube_points = create_cubic_markers(
    cube_size=0.050, marker_spacing=0.010
)

cube_pattern = Custom3DPattern(
    "cube_markers", "Cubic Object", cube_points,
    feature_detector=blob_detector
)
```

### Pyramid Calibration Object
```python
# Pyramid with base and apex markers
pyramid_points = create_pyramid_markers(
    base_size=0.060, height=0.040
)

pyramid_pattern = Custom3DPattern(
    "pyramid_markers", "Pyramid Object", pyramid_points,
    feature_detector=template_matcher
)
```

## Migration Guide

### From Legacy Interface
```python
# Old way
success, mtx, dist = calibrator.calibrate_from_images(
    images, XX=11, YY=8, L=0.02
)

# New way (backward compatible)
pattern = create_chessboard_pattern(
    "standard", width=11, height=8, square_size=0.02
)
success, mtx, dist = calibrator.calibrate_with_pattern(
    images, pattern
)
```

### Adding Custom Patterns
```python
class MyCustomPattern(CalibrationPattern):
    def __init__(self, ...):
        super().__init__(
            pattern_id="my_pattern",
            name="My Custom Pattern",
            description="Custom calibration pattern",
            is_planar=False  # Set appropriately
        )
    
    def detect_corners(self, image):
        # Implement detection logic
        return success, corners, point_ids
    
    def generate_object_points(self, point_ids=None):
        # Return 3D coordinates
        return object_points_3d
```

## File Structure

```
core/
├── chessboard_patterns.py     # Pattern abstraction system
└── intrinsic_calibration.py   # Updated calibrator with pattern support

examples/
├── run_calibration_example.py      # Basic calibration with patterns
├── chessboard_pattern_example.py   # Pattern creation and testing
└── 3d_pattern_example.py          # 3D pattern demonstrations
```

## Future Extensions

The architecture supports easy addition of:
- Circle grid patterns
- Random dot patterns  
- AprilTag-based patterns
- Deep learning-based feature detection
- Multi-modal calibration objects
- Dynamic/moving calibration patterns

## Best Practices

1. **For 2D patterns**: Use established patterns (chessboard/ChArUco) when possible
2. **For 3D patterns**: Validate 3D coordinates with precise measurement tools
3. **Feature detection**: Ensure robust detection across lighting conditions
4. **Manufacturing**: Consider tolerances in 3D object fabrication
5. **Validation**: Compare calibration accuracy with reference methods
6. **Documentation**: Document pattern specifications and coordinate systems

This enhanced system provides a flexible foundation for diverse calibration scenarios
while maintaining backward compatibility and ease of use.
"""
