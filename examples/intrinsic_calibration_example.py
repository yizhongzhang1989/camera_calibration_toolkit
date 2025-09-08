#!/usr/bin/env python3
"""
Intrinsic Camera Calibration Example
===================================

This example demonstrates how to use the IntrinsicCalibrator class for 
camera intrinsic parameter calibration using different calibration patterns:

1. Standard chessboard pattern (using image file paths)
2. ChArUco board pattern (using loaded images)  
3. ArUco grid board pattern (using image file paths)

Each example shows a clean, simple workflow focused on the essential
calibration steps without extra complexity.
"""

import os
import cv2
import sys
import numpy as np
import json
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import load_pattern_from_json


def test_chessboard_calibration():
    print("=" * 50)
    print("Test calibrate with standard chessboard, given a list of image file paths and calibration pattern.")
    print("=" * 50)
    
    # List all images inside the sample dir
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    image_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    pattern = load_pattern_from_json(config_data)

    # Create calibrator from image paths and calibration pattern
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern       # Member parameter set in constructor
    )

    # Pure OpenCV-style calibration with function parameters only
    result = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=0,                    # Function parameter
        criteria=None,              # Function parameter
        verbose=False
    )
    
    if result['rms_error'] < 0.5:
        print(f"\n✅ Calibration successful!")
        print("camera_matrix:")
        print(result['camera_matrix'])
        print("distortion_coefficients")
        print(result['distortion_coefficients'])
        print("rms_error")
        print(result['rms_error'])
        
        print("\n📄 Generating calibration report...")
        report_result = calibrator.generate_calibration_report("data/results/intrinsic_calibration_example_chessboard")
        if report_result:
            print(f"   📄 HTML Report: {report_result['html_report']}")
            print(f"   📊 JSON Data: {report_result['json_data']}")
    else:
        print(f"\n❌ Calibration failed - RMS error too high!")
        raise ValueError("Standard chessboard calibration failed")


def test_charuco_calibration():
    print("=" * 50)
    print("Test calibrate with ChArUco board pattern, given loaded images and calibration pattern.")
    print("=" * 50)
    
    # Load sample images with ChArUco boards
    sample_dir = os.path.join("sample_data", "intrinsic_calib_charuco_test_images")
    image_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))
    
    # Load images into memory
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    pattern = load_pattern_from_json(config_data)

    # Create calibrator from loaded images and calibration pattern
    calibrator = IntrinsicCalibrator(
        images=images,                     # Member parameter set in constructor
        calibration_pattern=pattern       # Member parameter set in constructor
    )

    # Pure OpenCV-style calibration with function parameters only
    result = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=0,                    # Function parameter
        criteria=None,              # Function parameter
        verbose=False
    )
    
    if result['rms_error'] < 0.5:
        print(f"\n✅ Calibration successful!")
        print("camera_matrix:")
        print(result['camera_matrix'])
        print("distortion_coefficients")
        print(result['distortion_coefficients'])
        print("rms_error")
        print(result['rms_error'])
        
        print("\n📄 Generating calibration report...")
        report_result = calibrator.generate_calibration_report("data/results/intrinsic_calibration_example_charuco")
        if report_result:
            print(f"   📄 HTML Report: {report_result['html_report']}")
            print(f"   📊 JSON Data: {report_result['json_data']}")
    else:
        print(f"\n❌ Calibration failed - RMS error too high!")
        raise ValueError("ChArUco calibration failed")


def test_gridboard_calibration():
    print("=" * 50)
    print("Test calibrate with ArUco grid board pattern, given image paths and calibration pattern, using rational camera model.")
    print("=" * 50)
    
    # Load sample images with ArUco GridBoard
    sample_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")  
    image_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    pattern = load_pattern_from_json(config_data)

    # Create calibrator from image paths and calibration pattern
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern       # Member parameter set in constructor
    )

    # Pure OpenCV-style calibration with function parameters only
    result = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=cv2.CALIB_RATIONAL_MODEL,  # Function parameter
        criteria=None,              # Function parameter
        verbose=False
    )
    
    if result['rms_error'] < 1.5:
        print(f"\n✅ Calibration successful!")
        print("camera_matrix:")
        print(result['camera_matrix'])
        print("distortion_coefficients")
        print(result['distortion_coefficients'])
        print("rms_error")
        print(result['rms_error'])
        
        print("\n📄 Generating calibration report...")
        report_result = calibrator.generate_calibration_report("data/results/intrinsic_calibration_example_gridboard")
        if report_result:
            print(f"   📄 HTML Report: {report_result['html_report']}")
            print(f"   📊 JSON Data: {report_result['json_data']}")
    else:
        print(f"\n❌ Calibration failed - RMS error too high!")
        raise ValueError("GridBoard calibration failed")


def main():
    """Main function with proper error handling."""
    print("Intrinsic Camera Calibration Example")
    print("=" * 60)
    print("Testing different calibration patterns with simple workflows")
    print()
    
    success_count = 0
    total_tests = 3
    
    try:
        test_chessboard_calibration()
        success_count += 1
        print("✅ Chessboard calibration completed successfully\n")
    except Exception as e:
        print(f"❌ Chessboard calibration failed: {e}\n")
    
    try:
        test_charuco_calibration()
        success_count += 1
        print("✅ ChArUco calibration completed successfully\n")
    except Exception as e:
        print(f"❌ ChArUco calibration failed: {e}\n")
    
    try:
        test_gridboard_calibration()
        success_count += 1
        print("✅ Grid Board calibration completed successfully\n")
    except Exception as e:
        print(f"❌ Grid Board calibration failed: {e}\n")
    
    print(f"📊 Results: {success_count}/{total_tests} calibrations successful")
    
    if success_count == total_tests:
        print("🎉 All calibrations completed successfully!")
        return 0
    elif success_count > 0:
        print("⚠️  Some calibrations failed. Check error messages above.")
        return 1
    else:
        print("❌ All calibrations failed!")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
