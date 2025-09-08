# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)

A comprehensive Python toolkit for camera calibration providing clean APIs for intrinsic calibration, eye-in-hand, and eye-to-hand robot calibration workflows.

## 🚀 Quick Start

### As Git Submodule (Recommended)

The toolkit is designed to work seamlessly as a git submodule in your projects:

```bash
# Add as submodule to your project
git submodule add https://github.com/yourusername/camera-calibration-toolkit.git camera_calibration_toolkit
cd camera_calibration_toolkit
pip install -r requirements.txt

# Update submodule to latest version
git submodule update --remote camera_calibration_toolkit
```

### Usage in Your Project

```python
# Import from submodule
import sys
sys.path.append('camera_calibration_toolkit')  # Add submodule to path

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator  
from core.calibration_patterns import StandardChessboard, load_pattern_from_json

# Your calibration code here...
```

### Alternative: Direct Clone

```bash
git clone https://github.com/yourusername/camera-calibration-toolkit.git
cd camera-calibration-toolkit
pip install -r requirements.txt
```

## 📖 Core Calibration Workflows

### 1. Intrinsic Camera Calibration

Calibrate camera's internal parameters (focal length, principal point, distortion):

```python
import sys
sys.path.append('camera_calibration_toolkit')  # If using as submodule

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import StandardChessboard

# Create calibration pattern (11x8 chessboard with 20mm squares)
pattern = StandardChessboard(width=11, height=8, square_size=0.020)

# Initialize calibrator with image paths
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]
calibrator = IntrinsicCalibrator(image_paths, pattern)

# Detect patterns and calibrate
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate(verbose=True)

# Get calibration results
camera_matrix = calibrator.get_camera_matrix()
dist_coeffs = calibrator.get_distortion_coefficients()
print(f"Calibration RMS Error: {rms_error:.3f} pixels")

# Generate comprehensive report with visualizations
report_info = calibrator.generate_calibration_report("./calibration_results")
print(f"Report saved to: {report_info['html_report']}")
```

### 2. Eye-in-Hand Robot Calibration

For cameras mounted on robot end-effectors, find the transformation between camera and robot frames:

```python
from core.eye_in_hand_calibration import EyeInHandCalibrator
import json

# Load robot poses (end-effector positions when images were taken)
robot_poses = []
for pose_file in ['pose1.json', 'pose2.json', 'pose3.json']:
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
        robot_poses.append(pose_data)

# Initialize calibrator with intrinsic parameters and robot poses  
calibrator = EyeInHandCalibrator(
    image_paths=image_paths,
    robot_poses=robot_poses,
    camera_matrix=camera_matrix,        # From intrinsic calibration
    distortion_coefficients=dist_coeffs, # From intrinsic calibration
    calibration_pattern=pattern
)

# Run hand-eye calibration
calibrator.detect_pattern_points(verbose=True)
initial_error = calibrator.calibrate(verbose=True)

# Optimize for better accuracy
final_error = calibrator.optimize_calibration(verbose=True) 
print(f"Optimization: {initial_error:.3f} → {final_error:.3f} pixels")

# Get camera-to-end-effector transformation matrix
transformation = calibrator.get_transformation_matrix()
print("Camera to End-Effector Transformation:")
print(transformation)

# Generate report with pattern detection and reprojection analysis
report_info = calibrator.generate_calibration_report("./hand_eye_results")
```

### 3. Eye-to-Hand Robot Calibration

For stationary cameras observing robot workspaces:

```python
from core.eye_to_hand_calibration import EyeToHandCalibrator

# Same API as eye-in-hand but different transformation result
calibrator = EyeToHandCalibrator(
    image_paths=image_paths,
    robot_poses=robot_poses,  # Robot poses when calibration target was attached
    camera_matrix=camera_matrix,
    distortion_coefficients=dist_coeffs,
    calibration_pattern=pattern
)

# Calibration workflow identical to eye-in-hand
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate(verbose=True)
optimized_error = calibrator.optimize_calibration(verbose=True)

# Get base-to-camera transformation matrix
transformation = calibrator.get_transformation_matrix()
print("Base to Camera Transformation:")
print(transformation)
```

## 🎯 Calibration Patterns

### Standard Chessboard (Most Common)

```python
from core.calibration_patterns import StandardChessboard

# Create 11x8 chessboard with 20mm squares
chessboard = StandardChessboard(width=11, height=8, square_size=0.020)

# Use with any calibrator
calibrator = IntrinsicCalibrator(image_paths, chessboard)
```

### ChArUco Board (More Robust)

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

# More robust detection, especially with partial visibility
calibrator = IntrinsicCalibrator(image_paths, charuco)
```

### Grid Board (Circle Patterns)

```python
from core.calibration_patterns import GridBoard

# Asymmetric circle grid
grid = GridBoard(width=4, height=11, spacing=0.02)
calibrator = IntrinsicCalibrator(image_paths, grid)
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
    'distortion_coefficients': dist_coeffs.tolist(),
    'rms_error': rms_error
}
```

## 📋 Data Formats

### Robot Pose Format

Robot poses should be provided as JSON with end-effector position and orientation:

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

- **Minimum**: 10-15 images for intrinsic, 5-8 for hand-eye calibration
- **Quality**: Sharp, well-lit, no motion blur
- **Coverage**: Various angles, distances, pattern positions
- **Pattern**: Fully visible calibration pattern in each image

## 🗂️ Project Structure as Submodule

When used as a submodule, your project structure might look like:

```
your_robot_project/
├── camera_calibration_toolkit/        # This toolkit as submodule
│   ├── core/                          # Core calibration modules
│   │   ├── intrinsic_calibration.py
│   │   ├── eye_in_hand_calibration.py 
│   │   ├── eye_to_hand_calibration.py
│   │   └── calibration_patterns/
│   ├── examples/                      # Reference examples
│   └── requirements.txt
├── your_robot_code/
│   ├── calibration_script.py      # Your calibration workflow
│   ├── camera_data/              # Your calibration images
│   └── poses/                    # Your robot pose data  
└── README.md
```

## � Complete Integration Example

```python
# your_robot_code/calibration_script.py
import sys
import os
import json
import glob

# Add calibration toolkit to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'camera_calibration_toolkit'))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator
from core.calibration_patterns import StandardChessboard

def main():
    # 1. Setup calibration pattern
    pattern = StandardChessboard(width=11, height=8, square_size=0.020)
    
    # 2. Intrinsic calibration
    intrinsic_images = glob.glob('camera_data/intrinsic/*.jpg')
    intrinsic_cal = IntrinsicCalibrator(intrinsic_images, pattern)
    intrinsic_cal.detect_pattern_points(verbose=True)
    rms_error = intrinsic_cal.calibrate(verbose=True)
    
    camera_matrix = intrinsic_cal.get_camera_matrix()
    dist_coeffs = intrinsic_cal.get_distortion_coefficients()
    
    # 3. Hand-eye calibration  
    hand_eye_images = glob.glob('camera_data/hand_eye/*.jpg')
    pose_files = glob.glob('poses/*.json')
    
    robot_poses = []
    for pose_file in sorted(pose_files):
        with open(pose_file, 'r') as f:
            robot_poses.append(json.load(f))
    
    hand_eye_cal = EyeInHandCalibrator(
        image_paths=hand_eye_images,
        robot_poses=robot_poses,
        camera_matrix=camera_matrix,
        distortion_coefficients=dist_coeffs,
        calibration_pattern=pattern
    )
    
    hand_eye_cal.detect_pattern_points(verbose=True)
    he_error = hand_eye_cal.calibrate(verbose=True)
    he_optimized = hand_eye_cal.optimize_calibration(verbose=True)
    
    transformation = hand_eye_cal.get_transformation_matrix()
    
    # 4. Save results
    results = {
        'intrinsic': {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'rms_error': float(rms_error)
        },
        'hand_eye': {
            'transformation_matrix': transformation.tolist(),
            'rms_error': float(he_optimized),
            'improvement': float(he_error - he_optimized)
        }
    }
    
    with open('calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Calibration completed!")
    print(f"   Intrinsic RMS: {rms_error:.3f} pixels")  
    print(f"   Hand-eye RMS: {he_optimized:.3f} pixels")
    print(f"   Results saved to: calibration_results.json")

if __name__ == "__main__":
    main()
```

## 🔧 Advanced Usage

### Custom Calibration Workflows

```python
# Custom optimization parameters
calibrator.optimize_calibration(
    iterations=10,           # More optimization iterations
    method='LM',            # Levenberg-Marquardt 
    verbose=True
)

# Custom calibration flags
calibrator.calibrate(
    flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST,
    verbose=True
)

# Access detailed calibration data
per_image_errors = calibrator.get_per_image_errors()
point_correspondences = calibrator.get_image_points()
```

### Multiple Pattern Support

```python
# Detect multiple pattern types for robustness
from core.calibration_patterns import PatternManager

manager = PatternManager()
chessboard = manager.create_pattern('standard_chessboard', width=11, height=8, square_size=0.02)
charuco = manager.create_pattern('charuco_board', width=8, height=6, square_size=0.04, marker_size=0.02)

# Try multiple patterns for each image
calibrator = IntrinsicCalibrator(image_paths, chessboard)
# If detection fails, try: calibrator.set_pattern(charuco)
```

## 🚨 Troubleshooting

**Pattern Detection Issues:**
- Verify pattern parameters (width, height, square_size)
- Check image quality (lighting, focus, no motion blur)  
- Try ChArUco patterns for better robustness
- Ensure pattern is fully visible in images

**High Calibration Errors:**
- Increase number of calibration images (15+ recommended)
- Improve image coverage (different angles, distances)
- Check pattern measurements accuracy
- Verify robot pose accuracy for hand-eye calibration

**Import/Path Issues:**
- Ensure submodule is properly added: `git submodule update --init`
- Check Python path: `sys.path.append('path/to/camera_calibration_toolkit')`
- Install dependencies: `pip install -r camera_calibration_toolkit/requirements.txt`

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

**Perfect for robotics projects!** This toolkit integrates seamlessly as a git submodule in robot vision applications.
