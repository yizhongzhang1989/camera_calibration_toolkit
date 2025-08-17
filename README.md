# Camera Calibration Toolkit

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-lightgrey.svg)

A comprehensive camera calibration toolkit with web interface for single camera, multiple cameras, and robot-mounted camera calibration.

![Web Interface Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=Camera+Calibration+Web+Interface)

## 🌟 Features

- **🎯 Eye-in-Hand Calibration**: Calibrate robot-mounted cameras with optimization
- **📱 Modern Web Interface**: Interactive calibration with real-time visualization  
- **🔧 Modular Design**: Core modules can be used independently in other projects
- **📊 3D Visualization**: Interactive Three.js visualization of calibration results
- **📈 Real-time Feedback**: Progress tracking and error metrics during calibration
- **💾 Export Ready**: Download calibration data and visualizations as ZIP files
- **🤖 Robot Integration**: Support for eye-in-hand and eye-to-hand configurations
- **📸 Multi-format Support**: Works with various image formats and chessboard patterns

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

### Option 2: Command Line (Legacy Compatible)

```bash
python duco_camera_calibrate_new.py \
  --calib_data_dir ./calibration_data \
  --xx 11 --yy 8 --square_size 0.02 \
  --calib_out_dir ./results \
  --reproj_out_dir ./visualizations
```

### Option 3: Python API

```python
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_chessboard_pattern

# Create calibration pattern
pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)

# Smart constructor - set member parameters directly  
intrinsic_cal = IntrinsicCalibrator(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    calibration_pattern=pattern
)

# Detect pattern points and calibrate
intrinsic_cal.detect_pattern_points(verbose=True)
rms_error = intrinsic_cal.calibrate_camera(verbose=True)

# Eye-in-hand calibration
eye_in_hand_cal = EyeInHandCalibrator()
eye_in_hand_cal.load_camera_intrinsics(camera_matrix, dist_coeffs)
image_paths, base2end_matrices, end2base_matrices = eye_in_hand_cal.load_calibration_data('path/to/data')
cam2end_4x4 = eye_in_hand_cal.calibrate(image_paths, end2base_matrices, XX=11, YY=8, L=0.02)
```

## 📁 Project Structure

```
camera-calibration-toolkit/
├── core/                      # Core calibration modules (reusable)
│   ├── __init__.py
│   ├── utils.py              # Common utility functions
│   ├── intrinsic_calibration.py    # Single camera calibration
│   └── eye_in_hand_calibration.py  # Robot-mounted camera calibration
├── web/                      # Flask web application
│   ├── app.py               # Main web server with API
│   ├── static/css/style.css # Web interface styling
│   ├── static/js/app.js     # Frontend JavaScript
│   └── templates/index.html # Single-page interface
├── data/                    # Data storage (auto-created)
│   ├── uploads/            # Uploaded images and poses
│   └── results/            # Calibration results
├── examples/               # Usage examples
│   ├── intrinsic_calibration_example.py  # Primary intrinsic calibration example
│   ├── 3d_pattern_example.py            # 3D pattern calibration
│   └── chessboard_pattern_example.py    # Pattern abstraction demo
├── tests/                  # Unit tests
├── main.py                # Main entry point
├── duco_camera_calibrate_new.py  # Legacy compatibility script
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
└── README.md            # This file
```

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

### REST API Endpoints

- `POST /api/upload_images` - Upload calibration images
- `POST /api/upload_poses` - Upload robot pose files
- `POST /api/set_parameters` - Set calibration parameters
- `POST /api/calibrate` - Run calibration process
- `GET /api/get_results/<session_id>` - Get calibration results
- `GET /api/export_results/<session_id>` - Export results as ZIP file
- `POST /api/clear_session/<session_id>` - Clear session data

### Python API Classes

#### IntrinsicCalibrator
```python
# Modern smart constructor interface
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_chessboard_pattern

# Smart constructor approach - set member parameters directly
pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)
calibrator = IntrinsicCalibrator(
    image_paths=image_paths,           # Member parameter
    calibration_pattern=pattern       # Member parameter
)

# Clean workflow: detect points then calibrate
calibrator.detect_pattern_points(verbose=True)
rms_error = calibrator.calibrate_camera(
    cameraMatrix=None,      # Function parameter (initial guess)
    distCoeffs=None,        # Function parameter (initial guess)
    flags=0,                # Function parameter
    verbose=True
)

# Get results
camera_matrix = calibrator.get_camera_matrix()
dist_coeffs = calibrator.get_distortion_coefficients()
```

**New Interface Benefits:**
- **Smart Constructor**: Set images and patterns directly during initialization
- **OpenCV-Style**: `calibrate_camera()` method matches OpenCV's interface exactly  
- **Organized Members**: Clean separation of data (members) vs processing options (function args)
- **Multiple Workflows**: Supports image paths, arrays, initial parameters, manual setup

#### EyeInHandCalibrator
```python
calibrator = EyeInHandCalibrator()
calibrator.load_camera_intrinsics(camera_matrix, dist_coeffs)
image_paths, base2end, end2base = calibrator.load_calibration_data(data_directory)
cam2end_matrix = calibrator.calibrate(image_paths, end2base, XX, YY, L)
errors = calibrator.calculate_reprojection_errors(...)
optimized_matrix = calibrator.optimize_calibration(...)
calibrator.save_results(output_directory)
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

### v1.0.0 (2025-08-16)
- Initial release with complete camera calibration toolkit
- Web interface with 3D visualization
- Modular core calibration modules
- Eye-in-hand calibration with optimization
- Comprehensive error handling and validation
- Export functionality and legacy compatibility

---

**Made with ❤️ for the robotics and computer vision community**
