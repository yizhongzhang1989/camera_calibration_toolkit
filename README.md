# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-lightgrey.svg)

A comprehensive camera calibration toolkit with modern architecture, web interface, and support for single camera, multiple cameras, and robot-mounted camera calibration.

![Web Interface Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=Camera+Calibration+Web+Interface)

## 🌟 Features

### **🏗️ Modern Architecture (New!)**
- **Inheritance-Based Design**: Clean base classes with specialized calibrators
- **Factory Pattern**: Unified calibrator creation interface
- **Zero Code Duplication**: ~300+ lines of duplicate code eliminated
- **Modular Pattern System**: Auto-discoverable calibration patterns

### **🎯 Calibration Types**
- **Eye-in-Hand Calibration**: Robot-mounted cameras with optimization
- **Intrinsic Calibration**: Single camera parameter estimation
- **Pattern Detection**: Chessboard and ChArUco board support
- **Multi-Method Support**: Multiple calibration algorithms available

### **📱 User Interfaces**
- **Modern Web Interface**: Interactive calibration with real-time visualization  
- **Python API**: Clean object-oriented programming interface
- **Command Line**: Batch processing and automation support
- **Factory Pattern**: `CalibrationFactory.create_calibrator('intrinsic')`

### **� Technical Features**
- **3D Visualization**: Interactive Three.js visualization of calibration results
- **Real-time Feedback**: Progress tracking and error metrics during calibration
- **Export Ready**: Download calibration data and visualizations as ZIP files
- **Multi-format Support**: Various image formats and calibration patterns
- **Optimization**: Nonlinear refinement for eye-in-hand calibration

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/camera-calibration-toolkit.git
cd camera-calibration-toolkit

# Install dependencies
pip install -r requirements.txt

# Start web interface
python main.py --mode web
```

Open your browser to `http://localhost:5000` and follow the interface:

1. **Select calibration type** (Eye-in-Hand for robot-mounted cameras)
2. **Set chessboard parameters** (corners and square size)
3. **Upload calibration images** showing the chessboard from different angles
4. **For Eye-in-Hand**: Upload corresponding robot pose JSON files
5. **Click "Start Calibration"** to begin the process
6. **View results** and export calibration data

### Option 2: Modern Python API (Recommended)

```python
from core.calibration_factory import CalibrationFactory
from core.calibration_patterns import create_chessboard_pattern

# Create calibration pattern
pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)

# Factory pattern - creates appropriate calibrator
calibrator = CalibrationFactory.create_calibrator(
    'intrinsic',                          # Calibration type
    image_paths=['img1.jpg', 'img2.jpg'], # Smart constructor
    calibration_pattern=pattern           # Modern pattern system
)

# Clean workflow
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate(verbose=True)

# Eye-in-hand calibration with factory
eye_calibrator = CalibrationFactory.create_calibrator(
    'eye_in_hand',
    image_paths=image_paths,
    robot_poses=poses,
    camera_matrix=camera_matrix,
    distortion_coefficients=dist_coeffs,
    calibration_pattern=pattern
)
rms_error = eye_calibrator.calibrate()
optimized_error = eye_calibrator.optimize_calibration()
```

### Option 3: Direct Class Usage

```python
from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator

# Intrinsic calibration
intrinsic_cal = IntrinsicCalibrator(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    calibration_pattern=pattern
)
intrinsic_cal.detect_pattern_points(verbose=True)
rms_error = intrinsic_cal.calibrate_camera(verbose=True)

# Eye-in-hand calibration  
eye_cal = EyeInHandCalibrator()
eye_cal.load_camera_intrinsics(camera_matrix, dist_coeffs)
eye_cal.load_calibration_data('path/to/data')
cam2end_matrix = eye_cal.calibrate()
```

### Option 4: Command Line Interface

```bash
python main.py \
  --calib_data_dir ./calibration_data \
  --xx 11 --yy 8 --square_size 0.02 \
  --calib_out_dir ./results \
  --reproj_out_dir ./visualizations
```

## 📁 Project Structure

```
camera-calibration-toolkit/
├── core/                              # Core calibration modules (reusable)
│   ├── __init__.py
│   ├── base_calibrator.py            # 🆕 Abstract base class for all calibrators  
│   ├── calibration_factory.py        # 🆕 Factory pattern for calibrator creation
│   ├── intrinsic_calibration.py      # ♻️ Refactored: inherits from BaseCalibrator
│   ├── eye_in_hand_calibration.py    # ♻️ Refactored: inherits from BaseCalibrator
│   ├── utils.py                      # Common utility functions
│   └── calibration_patterns/         # 🆕 Modular pattern system
│       ├── __init__.py               # Pattern discovery and management
│       ├── base.py                   # Abstract pattern base class
│       ├── manager.py                # Pattern registration system
│       ├── standard_chessboard.py    # Standard chessboard implementation
│       └── charuco_board.py          # ChArUco board implementation
├── web/                              # Flask web application
│   ├── app.py                       # Main web server with API
│   ├── visualization_utils.py       # Web visualization utilities
│   ├── static/
│   │   ├── css/style.css            # Web interface styling
│   │   └── js/
│   │       ├── base-calibration.js  # 🆕 Base class for web UI
│   │       ├── intrinsic.js         # ♻️ Refactored: inherits from base
│   │       └── app.js               # Main frontend application
│   └── templates/index.html         # Single-page interface
├── examples/                        # Usage examples (validated ✅)
│   ├── intrinsic_calibration_example.py     # Modern API example
│   ├── hand_in_eye_calibration_example.py   # Complete eye-in-hand workflow
│   ├── chessboard_pattern_example.py        # Pattern system demo
│   └── generate_chessboard_images.py        # Pattern image generation
├── data/                           # Data storage (auto-created)
│   ├── uploads/                   # Uploaded images and poses
│   └── results/                   # Calibration results
├── tests/                         # Unit tests
├── main.py                       # Main entry point and CLI
├── requirements.txt              # Python dependencies  
├── setup.py                     # Package installation
└── README.md                   # This file
```

### 🔧 Architecture Highlights

- **🆕 BaseCalibrator**: Abstract base class eliminates ~300+ lines of code duplication
- **🆕 CalibrationFactory**: Unified interface for creating any calibrator type
- **♻️ Inheritance Design**: Specialized calibrators inherit common functionality  
- **🆕 Pattern System**: Auto-discoverable, modular calibration patterns
- **🆕 Web UI Inheritance**: JavaScript classes mirror Python architecture

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install as Package

```bash
pip install -e .
```

## 📋 Data Formats

### Robot Pose Files

Robot pose files should be JSON files with the following format:

```json
{
    "end_xyzrpy": {
        "x": 0.5,      "y": 0.2,      "z": 0.3,
        "rx": 0.0,     "ry": 0.0,     "rz": 1.57
    }
}
```

Where:
- `x`, `y`, `z`: Position in meters
- `rx`, `ry`, `rz`: Rotation in radians (roll, pitch, yaw)

### Chessboard Requirements

- Use a standard chessboard pattern
- Ensure good lighting and sharp images
- Capture images from various angles and distances
- Minimum 10-15 images recommended
- For eye-in-hand: ensure chessboard is visible in all robot poses

## 🖥️ Web Interface

### Control Panel (Fixed Left Side)
- **Calibration Type Selection**: Choose between intrinsic, eye-in-hand, etc.
- **Parameter Configuration**: Set chessboard size and square dimensions
- **File Upload**: Upload calibration images and robot pose files
- **Process Control**: Start calibration, export results, clear session
- **Status Display**: Real-time calibration status and error metrics
- **3D Visualization**: Interactive 3D view of calibration results

### Results Display (Scrollable Right Side)
- **Image Grid**: Display original images with intermediate and final results
- **Column Views**: Toggle between original, corner detection, and reprojection views
- **Error Metrics**: Per-image reprojection errors
- **Export Options**: Download calibration parameters and visualization images

## 🔌 API Reference

### Factory Pattern (Recommended)

```python
from core.calibration_factory import CalibrationFactory

# Get available calibrator types
available = CalibrationFactory.get_available_types()
# Returns: ['intrinsic', 'eye_in_hand']

# Create calibrators using factory
intrinsic_cal = CalibrationFactory.create_calibrator('intrinsic', 
    image_paths=paths, calibration_pattern=pattern)
    
eye_cal = CalibrationFactory.create_calibrator('eye_in_hand',
    image_paths=paths, robot_poses=poses, camera_matrix=K, 
    distortion_coefficients=D, calibration_pattern=pattern)
```

### REST API Endpoints

- `POST /api/upload_images` - Upload calibration images
- `POST /api/upload_poses` - Upload robot pose files
- `POST /api/set_parameters` - Set calibration parameters
- `POST /api/calibrate` - Run calibration process
- `GET /api/get_results/<session_id>` - Get calibration results
- `GET /api/export_results/<session_id>` - Export results as ZIP file
- `POST /api/clear_session/<session_id>` - Clear session data

### BaseCalibrator API (Inherited by All Calibrators)

```python
# Common methods available in all calibrators (via inheritance)
class BaseCalibrator(ABC):
    # Image management
    def set_images_from_paths(self, image_paths: List[str]) -> bool
    def set_images_from_arrays(self, images: List[np.ndarray]) -> bool
    
    # Pattern management  
    def set_calibration_pattern(self, pattern: CalibrationPattern, 
                               pattern_type: str = None, **kwargs)
    
    # Processing
    def detect_pattern_points(self, verbose: bool = False) -> bool
    
    # Results and visualization
    def is_calibrated(self) -> bool
    def draw_pattern_on_images(self) -> List[Tuple[str, np.ndarray]]
    def draw_axes_on_undistorted_images(self, axis_length: float = None) -> List[...]
    
    # Abstract methods (implemented by specialized calibrators)
    @abstractmethod
    def calibrate(self, **kwargs) -> float
    
    @abstractmethod 
    def save_results(self, save_directory: str) -> None
```

### IntrinsicCalibrator API

```python
from core.intrinsic_calibration import IntrinsicCalibrator

# Modern smart constructor interface
calibrator = IntrinsicCalibrator(
    image_paths=image_paths,           # Smart constructor parameter
    calibration_pattern=pattern       # Modern pattern system
)

# Clean workflow: detect points then calibrate
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate_camera(
    cameraMatrix=None,      # OpenCV-style parameters
    distCoeffs=None,
    flags=0,
    verbose=True
)

# Results access
camera_matrix = calibrator.get_camera_matrix()
dist_coeffs = calibrator.get_distortion_coefficients()
extrinsics = calibrator.get_extrinsics()  # rvecs, tvecs for each image
```

### EyeInHandCalibrator API

```python
from core.eye_in_hand_calibration import EyeInHandCalibrator

# Modern member-based API
calibrator = EyeInHandCalibrator(
    camera_matrix=camera_matrix,         # Smart constructor
    distortion_coefficients=dist_coeffs,
    calibration_pattern=pattern
)

# Load and process data
calibrator.load_calibration_data(data_directory)
calibrator.detect_pattern_points(verbose=True)

# Calibration with multiple methods
rms_error = calibrator.calibrate(method=cv2.CALIB_HAND_EYE_HORAUD)

# Optional optimization
optimized_error = calibrator.optimize_calibration(
    iterations=5, ftol_rel=1e-6, verbose=True)

# Results and visualization  
transformation = calibrator.get_transformation_matrix()
calibrator.save_results(output_directory)
debug_images = calibrator.draw_reprojection_on_images()
```

### Pattern System API

```python
from core.calibration_patterns import (
    create_chessboard_pattern, get_pattern_manager
)

# Create patterns using factory functions
chessboard = create_chessboard_pattern('standard', 
    width=11, height=8, square_size=0.02)
    
charuco = create_chessboard_pattern('charuco',
    width=8, height=6, square_size=0.04, marker_size=0.02)

# Advanced pattern management
manager = get_pattern_manager()
available_types = manager.get_available_pattern_types()
configurations = manager.get_pattern_configurations()
pattern = manager.create_pattern('standard_chessboard', 
    width=11, height=8, square_size=0.025)
```

## 🚨 Troubleshooting

### Common Issues

**Chessboard not detected:**
- Ensure good lighting and sharp images
- Check chessboard parameter settings (corners count)
- Verify image quality and format
- Try different chessboard detection flags

**Calibration fails to converge:**
- Increase number of calibration images (minimum 10-15)
- Ensure diverse viewpoints and angles
- Check for motion blur or poor focus
- Verify robot pose accuracy

**High reprojection errors:**
- Review image quality and sharpness
- Check chessboard measurements accuracy
- Ensure accurate robot pose data
- Consider excluding poor quality images

**Web interface issues:**
- Check Python dependencies: `pip install -r requirements.txt`
- Verify Flask installation: `pip install Flask`
- Check firewall settings for port 5000
- Try different browser or clear browser cache

**Import errors:**
- Ensure you're in the correct directory
- Check Python path configuration
- Verify all dependencies are installed

### Performance Tips

- Use high-quality, well-lit images
- Ensure chessboard is flat and not warped
- Capture images with good coverage of the camera field of view
- For eye-in-hand: vary robot poses significantly
- Use appropriate chessboard size (not too small, not too large in image)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/camera-calibration-toolkit.git
cd camera-calibration-toolkit
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Running Tests

```bash
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- Flask community for web framework
- Three.js for 3D visualization capabilities
- Contributors and users of this toolkit

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/camera-calibration-toolkit/issues) page
2. Create a new issue with detailed information
3. Include error messages, system info, and steps to reproduce

## 🔄 Changelog

### v2.0.0 (2025-08-20) - 🏗️ Major Architecture Refactoring
**Breaking Changes & Major Improvements:**
- **🆕 BaseCalibrator**: Abstract base class eliminates ~300+ lines of code duplication
- **🆕 CalibrationFactory**: Unified factory pattern for creating calibrators
- **♻️ Inheritance Architecture**: IntrinsicCalibrator & EyeInHandCalibrator inherit from BaseCalibrator
- **🆕 Modular Pattern System**: Auto-discoverable calibration patterns with clean API
- **🔄 JavaScript Inheritance**: Web UI mirrors Python architecture with base classes
- **✅ Validation**: All examples tested and working (intrinsic, eye-in-hand, chessboard, pattern generation)
- **🗑️ Cleanup**: Removed non-functional 3D pattern examples
- **📚 Documentation**: Comprehensive README update with new API examples

**Maintained Compatibility:**
- ✅ All existing examples work unchanged
- ✅ Web interface fully functional  
- ✅ Legacy API still supported
- ✅ Zero functional regression

**New Features:**
```python
# Factory pattern for calibrator creation
calibrator = CalibrationFactory.create_calibrator('intrinsic', 
    image_paths=paths, calibration_pattern=pattern)

# Inherited common functionality
calibrator.detect_pattern_points()  # From BaseCalibrator
calibrator.draw_pattern_on_images()  # From BaseCalibrator
```

### v1.0.0 (2025-08-16) - Initial Release
- Initial release with complete camera calibration toolkit
- Web interface with 3D visualization
- Modular core calibration modules
- Eye-in-hand calibration with optimization
- Comprehensive error handling and validation
- Export functionality and legacy compatibility

---

**Made with ❤️ for the robotics and computer vision community**
