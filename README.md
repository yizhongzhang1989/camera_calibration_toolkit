# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)

A modern Python toolkit for camera calibration with clean APIs and web interface.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/camera-calibration-toolkit.git
cd camera-calibration-toolkit
pip install -r requirements.txt
```

### Basic Usage

#### 1. Intrinsic Camera Calibration

```python
from core.calibration_factory import CalibrationFactory
from core.calibration_patterns import create_chessboard_pattern

# Create pattern and calibrator
pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)
calibrator = CalibrationFactory.create_calibrator(
    'intrinsic',
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    calibration_pattern=pattern
)

# Run calibration
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate()

# Get results
camera_matrix = calibrator.get_camera_matrix()
dist_coeffs = calibrator.get_distortion_coefficients()
print(f"RMS Error: {rms_error:.3f} pixels")
```

#### 2. Eye-in-Hand Robot Calibration

```python
# For robot-mounted cameras
eye_calibrator = CalibrationFactory.create_calibrator(
    'eye_in_hand',
    image_paths=image_paths,
    robot_poses=robot_poses,  # List of robot poses
    camera_matrix=camera_matrix,
    distortion_coefficients=dist_coeffs,
    calibration_pattern=pattern
)

# Calibrate and optimize
rms_error = eye_calibrator.calibrate()
optimized_error = eye_calibrator.optimize_calibration()

# Get camera-to-robot transformation
transformation = eye_calibrator.get_transformation_matrix()
print(f"Optimization improved error: {rms_error:.3f} → {optimized_error:.3f}")
```

#### 3. Web Interface

```bash
python main.py --mode web
# Open browser to http://localhost:5000
```

## 📖 Examples

### Complete Intrinsic Calibration

```python
import glob
from core.calibration_factory import CalibrationFactory
from core.calibration_patterns import create_chessboard_pattern

# Get all calibration images
image_paths = glob.glob("calibration_images/*.jpg")

# Create 11x8 chessboard pattern with 20mm squares
pattern = create_chessboard_pattern('standard', 
    width=11, height=8, square_size=0.020)

# Create and run calibrator
calibrator = CalibrationFactory.create_calibrator('intrinsic',
    image_paths=image_paths,
    calibration_pattern=pattern
)

# Detect patterns and calibrate
success = calibrator.detect_pattern_points(verbose=True)
if success:
    rms_error = calibrator.calibrate(verbose=True)
    
    # Save results
    calibrator.save_results("./results")
    
    # Generate debug images
    pattern_images = calibrator.draw_pattern_on_images()
    axes_images = calibrator.draw_axes_on_undistorted_images()
```

### Complete Eye-in-Hand Calibration

```python
from core.calibration_factory import CalibrationFactory
from core.calibration_patterns import create_chessboard_pattern
import json

# Load robot poses from JSON files
robot_poses = []
for pose_file in glob.glob("poses/*.json"):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
        robot_poses.append(pose_data)

# Create calibrator with camera intrinsics and poses
calibrator = CalibrationFactory.create_calibrator('eye_in_hand',
    image_paths=image_paths,
    robot_poses=robot_poses,
    camera_matrix=camera_matrix,
    distortion_coefficients=dist_coeffs,
    calibration_pattern=pattern
)

# Run calibration
calibrator.detect_pattern_points(verbose=True)
initial_error = calibrator.calibrate(verbose=True)

# Optimize for better accuracy
final_error = calibrator.optimize_calibration(iterations=5, verbose=True)

# Get transformation matrix (camera to end-effector)
cam2end_matrix = calibrator.get_transformation_matrix()
print("Camera to End-Effector Transformation:")
print(cam2end_matrix)

# Save all results
calibrator.save_results("./eye_in_hand_results")
```

### Using Different Patterns

```python
# Standard chessboard
chessboard = create_chessboard_pattern('standard', 
    width=9, height=6, square_size=0.025)

# ChArUco board (more robust detection)
charuco = create_chessboard_pattern('charuco',
    width=8, height=6, square_size=0.04, marker_size=0.02)

# Use either pattern the same way
calibrator = CalibrationFactory.create_calibrator('intrinsic',
    image_paths=image_paths,
    calibration_pattern=charuco  # or chessboard
)
```

## 📋 Data Formats

### Robot Pose JSON Format

```json
{
    "end_xyzrpy": {
        "x": 0.5, "y": 0.2, "z": 0.3,
        "rx": 0.0, "ry": 0.0, "rz": 1.57
    }
}
```

### Required Image Setup

- **Minimum Images**: 10-15 for intrinsic, 5-8 for eye-in-hand  
- **Pattern Visibility**: Chessboard must be fully visible in each image
- **Variety**: Different angles, distances, and orientations
- **Quality**: Sharp, well-lit images without motion blur

## 🛠️ Available Calibrators

```python
# Get list of available calibrator types
from core.calibration_factory import CalibrationFactory
print(CalibrationFactory.get_available_types())
# Output: ['intrinsic', 'eye_in_hand']

# Create any calibrator type
cal = CalibrationFactory.create_calibrator('intrinsic', **kwargs)
```

## 📁 Project Structure

```
camera-calibration-toolkit/
├── core/                          # Core calibration modules
│   ├── calibration_factory.py     # Factory for creating calibrators
│   ├── base_calibrator.py         # Base class for all calibrators
│   ├── intrinsic_calibration.py   # Single camera calibration
│   ├── eye_in_hand_calibration.py # Robot camera calibration
│   └── calibration_patterns/      # Pattern detection system
├── examples/                      # Working examples
│   ├── intrinsic_calibration_example.py
│   ├── hand_in_eye_calibration_example.py  
│   └── chessboard_pattern_example.py
├── web/                          # Web interface
└── main.py                       # Entry point
```

## 🚨 Troubleshooting

**Pattern not detected:**
- Check chessboard parameters (width, height, square_size)
- Ensure good lighting and sharp images
- Try ChArUco patterns for better robustness

**High calibration errors:**
- Add more images with better coverage
- Check image quality (focus, lighting)
- Verify chessboard measurements are accurate

**Import errors:**
- Make sure you're in the project directory
- Install dependencies: `pip install -r requirements.txt`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features  
4. Submit a pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Need more examples?** Check the `examples/` directory for complete working code!
