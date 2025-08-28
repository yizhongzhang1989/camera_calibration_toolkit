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
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import load_pattern_from_json
from core.utils import load_images_from_directory


def load_pattern_config(config_path):
    """Load pattern configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        tuple: (pattern, pattern_type) where pattern is the calibration pattern
               and pattern_type is the pattern type string
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"📋 Loading pattern configuration from: {config_path}")
        print(f"   Pattern: {config_data.get('name', 'Unknown')}")
        print(f"   Type: {config_data.get('pattern_id', 'Unknown')}")
        
        # Create pattern using the JSON data
        pattern = load_pattern_from_json(config_data)
        pattern_type = config_data['pattern_id']
        
        return pattern, pattern_type
        
    except Exception as e:
        print(f"❌ Failed to load pattern configuration: {e}")
        raise


def test_chessboard_calibration():
    """Test calibration workflow with pattern configuration loaded from JSON."""
    print("🔧 Pattern-Based Calibration from JSON Config")
    print("=" * 50)
    
    # Load sample images
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} sample images")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern, pattern_type = load_pattern_config(config_path)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern,      # Member parameter set in constructor
        pattern_type=pattern_type          # Member parameter set in constructor (loaded from JSON)
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
            output_dir = f"data/results/{pattern_type}_calibration"
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
    """Test calibration workflow with ChArUco pattern configuration loaded from JSON."""
    print("\n\n🔧 ChArUco Board Calibration from JSON Config")
    print("=" * 50)
    
    # Load sample images with ChArUco boards
    sample_dir = os.path.join("sample_data", "intrinsic_calib_charuco_test_images")
    if not os.path.exists(sample_dir):
        print(f"❌ ChArUco sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} ChArUco sample images")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern, pattern_type = load_pattern_config(config_path)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern,      # Member parameter set in constructor
        pattern_type=pattern_type          # Member parameter set in constructor (loaded from JSON)
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
            output_dir = f"data/results/{pattern_type}_calibration"
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


def test_gridboard_calibration():
    """Test calibration workflow with ArUco GridBoard pattern configuration loaded from JSON."""
    print("\n\n🔧 ArUco GridBoard Calibration from JSON Config")
    print("=" * 50)
    
    # Load sample images with ArUco GridBoard
    sample_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")
    if not os.path.exists(sample_dir):
        print(f"❌ GridBoard sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} GridBoard sample images")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern, pattern_type = load_pattern_config(config_path)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern,      # Member parameter set in constructor
        pattern_type=pattern_type          # Member parameter set in constructor (loaded from JSON)
    )
    
    print("✅ Calibrator initialized with GridBoard pattern")
    print(f"   Image paths loaded: {calibrator.image_paths is not None}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Detect GridBoard markers and calibrate
    if calibrator.detect_pattern_points(verbose=True):
        print("✅ GridBoard pattern detection completed")
        
        # Pure OpenCV-style calibration with function parameters only
        rms_error = calibrator.calibrate_camera(
            cameraMatrix=None,           # Function parameter
            distCoeffs=None,            # Function parameter  
            flags=0,                    # Function parameter
            criteria=None,              # Function parameter
            verbose=True
        )
        
        if rms_error > 0:
            print(f"\n✅ GridBoard calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON
            output_dir = f"data/results/{pattern_type}_calibration"
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
    print("Camera calibration using JSON pattern configurations")
    print()
    
    test_chessboard_calibration()
    test_charuco_calibration()
    test_gridboard_calibration()
    
    print(f"\n✨ All calibrations completed!")
    print(f"   Results saved in: data/results/[pattern_type]_calibration/")
    print(f"   Pattern configurations loaded from JSON files in sample_data/")
