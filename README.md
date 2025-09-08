# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)

A comprehensive Python toolkit for camera calibration providing clean APIs for intrinsic calibration, eye-in-hand, and eye-to-hand robot calibration workflows.

## 📁 Examples Directory

A variety of usage examples can be found in the `examples/` directory. These scripts demonstrate typical workflows for intrinsic calibration, eye-in-hand, and eye-to-hand calibration, and can be used as templates for your own projects.

---

## 🚀 Quick Start

### Installation

Clone the toolkit to your desired location:

```bash
# Clone to your preferred path
git clone https://github.com/yizhongzhang1989/camera-calibration-toolkit.git CAMERA_CALIBRATION_TOOLKIT_PATH
cd CAMERA_CALIBRATION_TOOLKIT_PATH
pip install -r requirements.txt
```

### 1. Intrinsic Camera Calibration

Calibrate camera's internal parameters (focal length, principal point, distortion):

```python
import sys
import os
import json
import glob
sys.path.append('CAMERA_CALIBRATION_TOOLKIT_PATH')  # Replace with actual path

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import StandardChessboard

# Create calibration pattern (11x8 chessboard with 20mm squares)
pattern = StandardChessboard(width=11, height=8, square_size=0.020)

# Initialize calibrator with image paths
image_paths = glob.glob('calibration_images/*.jpg')
calibrator = IntrinsicCalibrator(
    image_paths=image_paths,
    calibration_pattern=pattern
)

# Perform calibration - returns dictionary with results
result = calibrator.calibrate()

# print results
print(f"RMS Error: {result['rms_error']:.4f} pixels")
print("Camera Matrix:")
print(result['camera_matrix'])
print("Distortion Coefficients:")
print(result['distortion_coefficients'])
    
# Generate comprehensive report with visualizations
calibrator.generate_calibration_report("./calibration_results")
```

### 2. Eye-in-Hand Robot Calibration

For cameras mounted on robot end-effectors, find the transformation between camera and robot frames.
**Note: Each image must have a corresponding JSON file with robot pose data in the same dir (e.g., `img001.jpg` + `img001.json`).**

```python
import sys
import numpy as np
import glob
sys.path.append('CAMERA_CALIBRATION_TOOLKIT_PATH')  # Replace with actual path

from core.eye_in_hand_calibration import EyeInHandCalibrator
from core.calibration_patterns import StandardChessboard

# Create calibration pattern
pattern = StandardChessboard(width=11, height=8, square_size=0.020)

# Set intrinsic camera parameters (from previous intrinsic calibration)
camera_matrix = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
])
distortion_coefficients = np.array([-0.2, 0.1, 0.0, 0.0, 0.0])

# Get image paths (each image should have corresponding .json pose file)
image_paths = glob.glob('hand_eye_images/*.jpg')

# Perform hand-eye calibration
hand_eye_calibrator = EyeInHandCalibrator(
    image_paths=image_paths,
    calibration_pattern=pattern,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    verbose=True
)

result = hand_eye_calibrator.calibrate(verbose=True)

# Print results
print(f"RMS Error: {result['rms_error']:.4f} pixels")
print("Camera to End-Effector Transformation:")
print(result['cam2end_matrix'])

# Generate calibration report
hand_eye_calibrator.generate_calibration_report("./hand_eye_results")
```

### 3. Eye-to-Hand Robot Calibration

For stationary cameras observing robot workspaces:

```python
from core.eye_to_hand_calibration import EyeToHandCalibrator

# Use same intrinsic calibration workflow as eye-in-hand
# ... (intrinsic calibration code same as above)

# For eye-to-hand, use image paths instead of loaded images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]

# Initialize eye-to-hand calibrator
calibrator = EyeToHandCalibrator(
    image_paths=image_paths,               # Use paths for eye-to-hand
    calibration_pattern=pattern,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients
)

# Calibration workflow identical to eye-in-hand
calibration_result = calibrator.calibrate(verbose=True)

if calibration_result['rms_error'] < 1.0:
    print(f"✅ Eye-to-hand calibration successful!")
    print(f"RMS Error: {calibration_result['rms_error']:.4f} pixels")
    
    # Get base-to-camera transformation matrix
    base2cam_matrix = calibration_result['base2cam_matrix']
    print("Base to Camera Transformation:")
    print(base2cam_matrix)
    
    # Generate calibration report
    report_info = calibrator.generate_calibration_report("./eye_to_hand_results")
else:
    print(f"❌ Eye-to-hand calibration failed!")
```

## 🎯 Calibration Patterns

### Loading Patterns from JSON Configuration

The recommended approach is to load patterns from JSON configuration files:

```python
import json
from core.calibration_patterns import load_pattern_from_json

# Load pattern configuration from JSON file
with open('pattern_config.json', 'r') as f:
    pattern_config = json.load(f)
pattern = load_pattern_from_json(pattern_config)

# Use with any calibrator
calibrator = IntrinsicCalibrator(image_paths, pattern)
```

### Creating Patterns Programmatically

You can also create patterns directly:

#### Standard Chessboard (Most Common)

```python
from core.calibration_patterns import StandardChessboard

# Create 11x8 chessboard with 20mm squares
chessboard = StandardChessboard(width=11, height=8, square_size=0.020)
```

#### ChArUco Board (More Robust)

ChArUco patterns combine chessboard and ArUco markers for better detection:

```python
from core.calibration_patterns import CharucoBoard

# Create 8x6 ChArUco board  
charuco = CharucoBoard(
    width=8, height=6,
    square_size=0.040,      # 40mm squares
    marker_size=0.020,      # 20mm ArUco markers
    dictionary_id=1         # DICT_4X4_50
)
```

#### Grid Board (Circle Patterns)

```python
from core.calibration_patterns import GridBoard

# Asymmetric circle grid
grid = GridBoard(width=4, height=11, spacing=0.02)
```

### Pattern Serialization

Save and restore exact pattern configurations:

```python
from core.calibration_patterns import save_pattern_to_json, load_pattern_from_json

# Save pattern configuration
pattern = StandardChessboard(width=11, height=8, square_size=0.025)
pattern_json = save_pattern_to_json(pattern)

# Later, restore identical pattern
restored_pattern = load_pattern_from_json(pattern_json)

# Embed in calibration results for reproducibility
calibration_data = {
    'timestamp': '2025-09-08T15:30:00Z',
    'pattern_config': pattern_json,  # Exact pattern used
    'camera_matrix': camera_matrix.tolist(),
    'distortion_coefficients': distortion_coefficients.tolist(),
    'rms_error': rms_error
}
```

## 📋 Data Formats

### Robot Pose Format

Robot poses should be provided as 4x4 transformation matrices in JSON format:

```json
{
    "end2base": [
        [1.0, 0.0, 0.0, 0.500],
        [0.0, 1.0, 0.0, 0.200], 
        [0.0, 0.0, 1.0, 0.300],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
```

Alternative format with position and rotation:

```json
{
    "end_xyzrpy": {
        "x": 0.500,    # Position in meters
        "y": 0.200, 
        "z": 0.300,
        "rx": 0.000,   # Rotation in radians (roll-pitch-yaw)
        "ry": 0.000,
        "rz": 1.570
    }
}
```

### Image Requirements

- **Minimum**: 10-15 images for intrinsic calibration, 5-8 for hand-eye calibration
- **Quality**: Sharp, well-lit, no motion blur
- **Coverage**: Various angles, distances, pattern positions across the image
- **Pattern**: Fully visible calibration pattern in each image
- **Format**: Standard image formats (.jpg, .png, .bmp)
- **Naming**: For hand-eye calibration, images should have corresponding pose files (e.g., `img001.jpg` → `img001.json`)

## 🗂️ Project Structure

When using the toolkit in your project, your structure might look like:

```
your_robot_project/
├── CAMERA_CALIBRATION_TOOLKIT_PATH/        # This toolkit cloned here
│   ├── core/                              # Core calibration modules
│   │   ├── intrinsic_calibration.py
│   │   ├── eye_in_hand_calibration.py 
│   │   ├── eye_to_hand_calibration.py
│   │   └── calibration_patterns/
│   ├── examples/                          # Reference examples
│   └── requirements.txt
├── your_robot_code/
│   ├── calibration_script.py              # Your calibration workflow
│   ├── camera_data/                       # Your calibration images
│   └── poses/                             # Your robot pose data  
└── README.md
```

## 🔧 Complete Integration Example

```python
# your_robot_code/calibration_script.py
import sys
import os
import json
import glob
import cv2
import numpy as np

# Add calibration toolkit to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CAMERA_CALIBRATION_TOOLKIT_PATH'))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator
from core.calibration_patterns import load_pattern_from_json

def main():
    # 1. Load calibration pattern from JSON configuration
    with open('pattern_config.json', 'r') as f:
        pattern_config = json.load(f)
    pattern = load_pattern_from_json(pattern_config)
    
    # 2. Intrinsic calibration
    intrinsic_images = glob.glob('camera_data/intrinsic/*.jpg')
    images = [cv2.imread(path) for path in intrinsic_images]
    
    intrinsic_cal = IntrinsicCalibrator(
        images=images,
        calibration_pattern=pattern
    )
    
    intrinsic_result = intrinsic_cal.calibrate(verbose=True)
    
    if intrinsic_result['rms_error'] > 0.5:
        print("❌ Intrinsic calibration failed")
        return
    
    camera_matrix = intrinsic_result['camera_matrix']
    dist_coeffs = intrinsic_result['distortion_coefficients']
    
    # 3. Hand-eye calibration  
    hand_eye_images = []
    end2base_matrices = []
    
    image_files = glob.glob('camera_data/hand_eye/*.jpg')
    for image_file in sorted(image_files):
        # Load image
        image = cv2.imread(image_file)
        hand_eye_images.append(image)
        
        # Load corresponding pose
        pose_file = image_file.replace('.jpg', '.json')
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        end2base_matrix = np.array(pose_data['end2base'])
        end2base_matrices.append(end2base_matrix)
    
    hand_eye_cal = EyeInHandCalibrator(
        images=hand_eye_images,
        end2base_matrices=end2base_matrices,
        calibration_pattern=pattern,
        camera_matrix=camera_matrix,
        distortion_coefficients=dist_coeffs.flatten()
    )
    
    he_result = hand_eye_cal.calibrate(verbose=True)
    
    if he_result['rms_error'] > 1.0:
        print("❌ Hand-eye calibration failed")
        return
    
    # 4. Save results
    results = {
        'intrinsic': {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'rms_error': float(intrinsic_result['rms_error'])
        },
        'hand_eye': {
            'cam2end_matrix': he_result['cam2end_matrix'].tolist(),
            'rms_error': float(he_result['rms_error'])
        }
    }
    
    with open('calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Calibration completed!")
    print(f"   Intrinsic RMS: {intrinsic_result['rms_error']:.3f} pixels")  
    print(f"   Hand-eye RMS: {he_result['rms_error']:.3f} pixels")
    print(f"   Results saved to: calibration_results.json")
    
    # Generate reports
    intrinsic_cal.generate_calibration_report("./reports/intrinsic")
    hand_eye_cal.generate_calibration_report("./reports/hand_eye")

if __name__ == "__main__":
    main()
```

## 🔧 Advanced Usage

### Custom Calibration Parameters

```python
# Custom calibration flags for intrinsic calibration
result = calibrator.calibrate(
    cameraMatrix=None,
    distCoeffs=None,
    flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST,
    criteria=None,
    verbose=True
)

# Use rational camera model for higher distortion scenarios
result = calibrator.calibrate(
    flags=cv2.CALIB_RATIONAL_MODEL,
    verbose=True
)

# Access calibration results
camera_matrix = result['camera_matrix']
distortion_coefficients = result['distortion_coefficients']
rms_error = result['rms_error']
```

### Pattern Manager Usage

```python
from core.calibration_patterns import CalibrationPatternManager

# Get pattern manager instance
manager = CalibrationPatternManager()

# Create patterns programmatically
chessboard = manager.create_pattern(
    'standard_chessboard', 
    width=11, height=8, square_size=0.02
)

charuco = manager.create_pattern(
    'charuco_board', 
    width=8, height=6, 
    square_size=0.04, marker_size=0.02
)

# Get available pattern configurations
available_patterns = manager.get_pattern_configurations()
for pattern_id, config in available_patterns.items():
    print(f"Pattern: {pattern_id} - {config['name']}")
```

## 🚨 Troubleshooting

**Pattern Detection Issues:**
- Verify pattern parameters match your actual calibration target
- Check image quality (lighting, focus, no motion blur)  
- Try ChArUco patterns for better robustness in difficult lighting
- Ensure pattern is fully visible in images
- Load pattern from JSON configuration for consistency

**High Calibration Errors:**
- Increase number of calibration images (15+ recommended)
- Improve image coverage (different angles, distances)
- Check pattern measurements accuracy with ruler/calipers
- Verify robot pose accuracy for hand-eye calibration
- Check that `rms_error` in calibration result is < 0.5 for intrinsic, < 1.0 for hand-eye

**Import/Path Issues:**
- Ensure toolkit is properly cloned to your chosen path
- Check Python path: `sys.path.append('CAMERA_CALIBRATION_TOOLKIT_PATH')`
- Install dependencies: `pip install -r CAMERA_CALIBRATION_TOOLKIT_PATH/requirements.txt`

**API Usage Issues:**
- Use `calibrator.calibrate()` method which returns a dictionary result
- Load patterns from JSON with `load_pattern_from_json()`
- For hand-eye calibration, pass `end2base_matrices` as 4x4 numpy arrays
- Generate reports with `calibrator.generate_calibration_report(output_dir)`

## 🤝 Contributing

This toolkit is designed for integration into robotics projects. Contributions welcome:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-calibration-method`
3. Add tests for new functionality
4. Submit pull request with clear description

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- **OpenCV**: Core computer vision functionality
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing and optimization

---

**Perfect for robotics projects!** This toolkit can be easily integrated into robot vision applications by cloning to your preferred location.
