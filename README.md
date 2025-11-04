# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)

A comprehensive Python toolkit for camera calibration providing clean APIs for intrinsic calibration, eye-in-hand, and eye-to-hand robot calibration workflows and generate calibration report with clear visualization.

In addition, a web based GUI is provided for easy operation. (Under development)


## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

python ./test_runner --all      # If all tests passed, you have successfully setup the repo. Calibration results will write to data/results

python ./web/app.py     # start the web server if needed, default port 5000
```

### 1. Intrinsic Camera Calibration

Calibrate camera's internal parameters (focal length, principal point, distortion). The following script assume you use a standard 11x8 chessboard and captured a set of images. More examples can be found in `examples\intrinsic_calibration_example.py`

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

if result is not None:
    # print results
    print(f"RMS Error: {result['rms_error']:.4f} pixels")
    print("Camera Matrix:")
    print(result['camera_matrix'])
    print("Distortion Coefficients:")
    print(result['distortion_coefficients'])
        
    # Generate comprehensive report with visualizations
    calibrator.generate_calibration_report("./calibration_results")

else:
    print("Intrinsic calibration failed!")
```

### 2. Eye-in-Hand Robot Calibration

For cameras mounted on robot end-effectors, find the transformation between camera and robot frames. Intrinsic of the camera must be calibrated previously. More examples can be found in `examples\eye_in_hand_calibration_example.py`.
**Note: Each image must have a corresponding JSON file with robot pose data in the same dir (e.g., `img001.jpg` + `img001.json`). Sample data can be found in `sample_data\eye_in_hand_test_data`**

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

if result is not None:
    print(f"RMS Error: {result['rms_error']:.4f} pixels")
    print("Camera to End-Effector Transformation:")
    print(result['cam2end_matrix'])
    print("Target to Robot-Base Transformation:")
    print(result['cam2end_matrix'])

    # Generate calibration report
    hand_eye_calibrator.generate_calibration_report("./hand_eye_results")
else:
    print("eye-in-hand calibration failed!")
```

### 3. Eye-to-Hand Robot Calibration

For stationary cameras observing robot workspaces. In this example, we demonstrate an alternative parameter: directly input a list of images and corresponding end2base matrices. More examples can be found in `examples\eye_to_hand_calibration_example.py`.

```python
import cv2
import json
from core.eye_to_hand_calibration import EyeToHandCalibrator

# Prepare image paths
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]

# Load images and end2base_matrices
images = []
end2base_matrices = []
for img_path in image_paths:
    images.append(cv2.imread(img_path))
    json_path = os.path.splitext(img_path)[0] + ".json"
    with open(json_path, "r") as f:
        data = json.load(f)
        end2base_matrices.append(np.array(data["end2base"]))

# Initialize eye-to-hand calibrator
calibrator = EyeToHandCalibrator(
    images=images,
    end2base_matrices=end2base_matrices,
    calibration_pattern=pattern,    # create pattern as previous example
    camera_matrix=camera_matrix,    # calculate intrinsic as previous example
    distortion_coefficients=distortion_coefficients,    # calculate distortion coefficients as previous example
    verbose=True
)

calibration_result = calibrator.calibrate(verbose=True)

if calibration_result is not None:
    print(f"✅ Eye-to-hand calibration successful!")
    print(f"RMS Error: {calibration_result['rms_error']:.4f} pixels")

    # Print main transformation matrix (base to camera)
    print("Base to Camera Transformation:")
    print(calibration_result['base2cam_matrix'])
    print("Target to End-Effector Transformation:")
    print(calibration_result['target2end_matrix'])

    # Generate calibration report
    report_info = calibrator.generate_calibration_report("./eye_to_hand_results")
else:
    print(f"❌ Eye-to-hand calibration failed.")
```

### 4. Web (Under Development)

If you start the web, you can open the web in browser http://localhost:5000. You can generate calibration pattern, upload images and perform calibration online. **Note: Functions of web are still under development.**


## 🎯 Calibration Patterns

Currently we support 3 kinds of chessboard: standard chessboard, ChArUCo chessboard and grid board. If you do not have a chessboard, you can generate the chessboard image using script, or on the web, then print the image. 


### Standard Chessboard (Most Common)

<img src="sample_data\eye_in_hand_test_data\0.jpg" alt="Robot Diagram" width="300">

```python
from core.calibration_patterns import StandardChessboard

# Create 11x8 chessboard with 20mm squares
chessboard = StandardChessboard(width=11, height=8, square_size=0.020)
```

### ChArUco Board (More Robust)

<img src="sample_data\intrinsic_calib_charuco_test_images\20250818_065551594_iOS.jpg" alt="Robot Diagram" width="300">

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

### Grid Board

<img src="sample_data\intrinsic_calib_grid_test_images\snapshot_20250822_145040.jpg" alt="Robot Diagram" width="300">

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

Alternative format with position and rotation. Since the order of rotation is not well defined, this definition is less prefered. If both matrix and xyzrpy exist, matrix will be used.

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


## 📁 Examples Directory

A variety of usage examples can be found in the `examples/` directory. These scripts demonstrate typical workflows for intrinsic calibration, eye-in-hand, and eye-to-hand calibration, and can be used as templates for your own projects.


## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

