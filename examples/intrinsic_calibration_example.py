#!/usr/bin/env python3
"""
Intrinsic Camera Calibration Example
===================================

This example demonstrates how to use the IntrinsicCalibrator class for 
camera intrinsic parameter calibration using different calibration patterns:
1. Standard chessboard pattern
2. ChArUco board pattern

The class features a clean interface with smart constructor arguments 
and organized member variables.
"""

import os
import cv2
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_chessboard_pattern
from core.utils import load_images_from_directory


def test_chessboard_calibration():
    """Test calibration workflow with chessboard pattern using image paths."""
    print("🔧 Chessboard Calibration from Image Paths")
    print("=" * 50)
    
    # Load sample images
    sample_dir = os.path.join("sample_data", "intrinsic_calib_test_images")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} sample images")
    
    # Create calibration pattern
    pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern,      # Member parameter set in constructor
        pattern_type='standard'           # Member parameter set in constructor
    )
    
    print("✅ Calibrator initialized with smart constructor")
    print(f"   Image paths loaded: {calibrator.image_paths is not None}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Now just detect points and calibrate - no convenience methods needed
    if calibrator.detect_pattern_points(verbose=True):
        print("✅ Pattern detection completed")
        
        # Pure OpenCV-style calibration with function parameters only
        rms_error = calibrator.calibrate_camera(
            cameraMatrix=None,           # Function parameter
            distCoeffs=None,            # Function parameter  
            flags=0,                    # Function parameter
            criteria=None,              # Function parameter
            verbose=True
        )
        
        if rms_error > 0:
            print(f"\n✅ Calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON
            output_dir = "results/chessboard_calibration"
            os.makedirs(output_dir, exist_ok=True)
            calibrator.save_calibration(
                os.path.join(output_dir, "calibration_results.json"),
                include_extrinsics=True
            )
            print(f"   Calibration data saved to: {output_dir}/calibration_results.json")
            
            # Generate debug images
            print(f"\n🔍 Generating debug images...")
            
            # Draw detected patterns on original images
            pattern_debug_dir = os.path.join(output_dir, "pattern_detection")
            os.makedirs(pattern_debug_dir, exist_ok=True)
            pattern_images = calibrator.draw_pattern_on_images()
            
            for filename, debug_img in pattern_images:
                output_path = os.path.join(pattern_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Pattern detection images: {len(pattern_images)} images in {pattern_debug_dir}")
            
            # Draw 3D axes on undistorted images
            axes_debug_dir = os.path.join(output_dir, "undistorted_axes")
            os.makedirs(axes_debug_dir, exist_ok=True)
            axes_images = calibrator.draw_axes_on_undistorted_images()
            
            for filename, debug_img in axes_images:
                output_path = os.path.join(axes_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Undistorted axes images: {len(axes_images)} images in {axes_debug_dir}")


def test_charuco_calibration():
    """Test calibration workflow with ChArUco board pattern using image paths."""
    print("\n\n🔧 ChArUco Board Calibration from Image Paths")
    print("=" * 50)
    
    # Load sample images with ChArUco boards
    sample_dir = os.path.join("sample_data", "intrinsic_calib_charuco_test_images")
    if not os.path.exists(sample_dir):
        print(f"❌ ChArUco sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} ChArUco sample images")
    
    # Create ChArUco calibration pattern with specified parameters
    # Based on: {"dict": 1, "chessboard_w": 10, "chessboard_h": 7, 
    #           "square_length": 1.81e-02, "marker_length": 9.05e-03}
    pattern = create_chessboard_pattern(
        pattern_type='charuco',
        width=10,                          # chessboard_w
        height=7,                          # chessboard_h  
        square_size=0.0181,               # square_length (1.8100000917911530e-02)
        marker_size=0.00905,              # marker_length (9.0500004589557648e-03)
        dictionary_id=cv2.aruco.DICT_4X4_100  # dict=1 corresponds to DICT_4X4_100
    )
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern,      # Member parameter set in constructor
        pattern_type='charuco'            # Member parameter set in constructor
    )
    
    print("✅ Calibrator initialized with ChArUco pattern")
    print(f"   Image paths loaded: {calibrator.image_paths is not None}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Detect ChArUco corners and calibrate
    if calibrator.detect_pattern_points(verbose=True):
        print("✅ ChArUco pattern detection completed")
        
        # Pure OpenCV-style calibration with function parameters only
        rms_error = calibrator.calibrate_camera(
            cameraMatrix=None,           # Function parameter
            distCoeffs=None,            # Function parameter  
            flags=0,                    # Function parameter
            criteria=None,              # Function parameter
            verbose=True
        )
        
        if rms_error > 0:
            print(f"\n✅ ChArUco calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON
            output_dir = "results/charuco_calibration"
            os.makedirs(output_dir, exist_ok=True)
            calibrator.save_calibration(
                os.path.join(output_dir, "calibration_results.json"),
                include_extrinsics=True
            )
            print(f"   Calibration data saved to: {output_dir}/calibration_results.json")
            
            # Generate debug images
            print(f"\n🔍 Generating debug images...")
            
            # Draw detected patterns on original images
            pattern_debug_dir = os.path.join(output_dir, "pattern_detection")
            os.makedirs(pattern_debug_dir, exist_ok=True)
            pattern_images = calibrator.draw_pattern_on_images()
            
            for filename, debug_img in pattern_images:
                output_path = os.path.join(pattern_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Pattern detection images: {len(pattern_images)} images in {pattern_debug_dir}")
            
            # Draw 3D axes on undistorted images
            axes_debug_dir = os.path.join(output_dir, "undistorted_axes")
            os.makedirs(axes_debug_dir, exist_ok=True)
            axes_images = calibrator.draw_axes_on_undistorted_images()
            
            for filename, debug_img in axes_images:
                output_path = os.path.join(axes_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Undistorted axes images: {len(axes_images)} images in {axes_debug_dir}")


if __name__ == "__main__":
    print("🚀 Intrinsic Camera Calibration Example")
    print("=" * 60)
    print("Camera calibration using different pattern types")
    print()
    
    test_chessboard_calibration()
    test_charuco_calibration()
    
    print(f"\n✨ All calibrations completed!")
    print(f"   Chessboard results: results/chessboard_calibration/")
    print(f"   ChArUco results: results/charuco_calibration/")
