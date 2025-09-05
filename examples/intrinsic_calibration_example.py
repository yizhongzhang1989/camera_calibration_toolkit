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


def load_images_from_directory(directory):
    """Load image file paths from a directory.
    
    Args:
        directory: Path to the directory containing images
        
    Returns:
        list: List of absolute paths to image files
    """
    if not os.path.exists(directory):
        return []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    return sorted(image_paths)


def load_pattern_from_config(config_path):
    """Load pattern configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        CalibrationPattern: The loaded calibration pattern
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print(f"📋 Loading pattern configuration from: {config_path}")
        print(f"   Pattern: {config_data.get('name', 'Unknown')}")
        print(f"   Type: {config_data.get('pattern_id', 'Unknown')}")
        
        # Create pattern using the JSON data
        pattern = load_pattern_from_json(config_data)
        
        return pattern
        
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
        error_msg = f"Sample data directory not found: {sample_dir}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} sample images")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern = load_pattern_from_config(config_path)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern       # Member parameter set in constructor
    )
    
    print("✅ Calibrator initialized with smart constructor")
    print(f"   Image paths loaded: {calibrator.image_paths is not None}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Test automatic point detection in calibrate_camera - don't call detect_pattern_points explicitly
    print("🔄 Testing automatic pattern detection in calibrate_camera...")
    
    # Pure OpenCV-style calibration with function parameters only
    success = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=0,                    # Function parameter
        criteria=None,              # Function parameter
        verbose=True
    )
    
    if success:
        rms_error = calibrator.get_rms_error()
        # Check RMS error threshold - consider calibration failed if > 0.5
        if rms_error > 0.5:
            print(f"\n❌ Calibration failed - RMS error too high!")
            print(f"   RMS Error: {rms_error:.4f} pixels (threshold: 0.5)")
            print(f"   High RMS error indicates poor calibration quality")
            print(f"   Try improving image quality or pattern detection")
            success = False
        else:
            print(f"\n✅ Calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON  
            output_dir = f"data/results/{pattern.pattern_id}_calibration"
            os.makedirs(output_dir, exist_ok=True)
            
            # Serialize calibration data to JSON
            calibration_data = calibrator.to_json()
            output_path = os.path.join(output_dir, "calibration_results.json")
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
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
            
            # Draw reprojection analysis on images
            reprojection_debug_dir = os.path.join(output_dir, "reprojection_analysis")
            os.makedirs(reprojection_debug_dir, exist_ok=True)
            reprojection_images = calibrator.draw_reprojection_on_images()
            
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Reprojection analysis images: {len(reprojection_images)} images in {reprojection_debug_dir}")
    
    if not success:
        print(f"\n❌ Calibration failed!")
        print(f"   Could not calibrate camera parameters")
        print(f"   Check that:")
        print(f"   - Images contain visible calibration patterns")
        print(f"   - Pattern configuration matches the actual patterns")
        print(f"   - Images are clear and not blurry")
        raise ValueError("Standard chessboard calibration failed")


def test_charuco_calibration():
    """Test calibration workflow with ChArUco pattern configuration loaded from JSON."""
    print("\n\n🔧 ChArUco Board Calibration from JSON Config")
    print("=" * 50)
    
    # Load sample images with ChArUco boards
    sample_dir = os.path.join("sample_data", "intrinsic_calib_charuco_test_images")
    if not os.path.exists(sample_dir):
        error_msg = f"ChArUco sample data directory not found: {sample_dir}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} ChArUco sample images")
    
    # Load images into memory
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image {image_path}")
    
    print(f"Successfully loaded {len(images)} images into memory")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern = load_pattern_from_config(config_path)
    
    # Smart constructor - pass images directly instead of image_paths
    calibrator = IntrinsicCalibrator(
        images=images,                     # Pass loaded images directly
        calibration_pattern=pattern       # Member parameter set in constructor
    )
    
    print("✅ Calibrator initialized with ChArUco pattern")
    print(f"   Image paths loaded: {len(image_paths) > 0}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image count: {len(images)}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Test automatic pattern detection in calibrate_camera - don't call detect_pattern_points explicitly
    print("🔄 Testing automatic pattern detection in calibrate_camera...")
    
    # Pure OpenCV-style calibration with function parameters only
    success = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=0,                    # Function parameter
        criteria=None,              # Function parameter
        verbose=True
    )
    
    if success:
        rms_error = calibrator.get_rms_error()
        # Check RMS error threshold - consider calibration failed if > 0.5
        if rms_error > 0.5:
            print(f"\n❌ ChArUco calibration failed - RMS error too high!")
            print(f"   RMS Error: {rms_error:.4f} pixels (threshold: 0.5)")
            print(f"   High RMS error indicates poor calibration quality")
            print(f"   Try improving image quality or pattern detection")
            success = False
        else:
            print(f"\n✅ ChArUco calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON
            output_dir = f"data/results/{pattern.pattern_id.lower()}_calibration"
            os.makedirs(output_dir, exist_ok=True)
            
            # Serialize calibration data to JSON
            calibration_data = calibrator.to_json()
            output_path = os.path.join(output_dir, "calibration_results.json")
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
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
            
            # Draw reprojection analysis on images
            reprojection_debug_dir = os.path.join(output_dir, "reprojection_analysis")
            os.makedirs(reprojection_debug_dir, exist_ok=True)
            reprojection_images = calibrator.draw_reprojection_on_images()
            
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Reprojection analysis images: {len(reprojection_images)} images in {reprojection_debug_dir}")
    
    if not success:
        print(f"\n❌ ChArUco calibration failed!")
        print(f"   Could not calibrate camera parameters using ChArUco pattern")
        print(f"   Check that:")
        print(f"   - ChArUco patterns are clearly visible in images")
        print(f"   - Pattern configuration matches the actual ChArUco board")
        print(f"   - Images have sufficient resolution and are not blurry")
        raise ValueError("ChArUco calibration failed")


def test_gridboard_calibration():
    """Test calibration workflow with ArUco GridBoard pattern configuration loaded from JSON."""
    print("\n\n🔧 ArUco GridBoard Calibration from JSON Config")
    print("=" * 50)
    
    # Load sample images with ArUco GridBoard
    sample_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")  
    if not os.path.exists(sample_dir):
        error_msg = f"GridBoard sample data directory not found: {sample_dir}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    image_paths = load_images_from_directory(sample_dir)
    print(f"Using {len(image_paths)} GridBoard sample images")
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return
    
    pattern = load_pattern_from_config(config_path)
    
    # Smart constructor - sets member parameters directly
    calibrator = IntrinsicCalibrator(
        image_paths=image_paths,           # Member parameter set in constructor
        calibration_pattern=pattern       # Member parameter set in constructor
    )
    
    print("✅ Calibrator initialized with GridBoard pattern")
    print(f"   Image paths loaded: {calibrator.image_paths is not None}")
    print(f"   Images loaded: {calibrator.images is not None}")
    print(f"   Image size: {calibrator.image_size}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Test automatic pattern detection in calibrate_camera - don't call detect_pattern_points explicitly
    print("🔄 Testing automatic pattern detection in calibrate_camera...")
    
    # Pure OpenCV-style calibration with function parameters only
    success = calibrator.calibrate(
        cameraMatrix=None,           # Function parameter
        distCoeffs=None,            # Function parameter  
        flags=0,                    # Function parameter
        criteria=None,              # Function parameter
        verbose=True
    )
    
    if success:
        rms_error = calibrator.get_rms_error()
        # Check RMS error threshold - consider calibration failed if > 0.5
        if rms_error > 1.5:
            print(f"\n❌ GridBoard calibration failed - RMS error too high!")
            print(f"   RMS Error: {rms_error:.4f} pixels (threshold: 1.5)")
            print(f"   High RMS error indicates poor calibration quality")
            print(f"   Try improving image quality or pattern detection")
            success = False
        else:
            print(f"\n✅ GridBoard calibration successful!")
            camera_matrix = calibrator.get_camera_matrix()
            print(f"   RMS Error: {rms_error:.4f} pixels")
            print(f"   Camera Matrix: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"   Principal Point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            
            # Save calibration data to JSON
            output_dir = f"data/results/{pattern.pattern_id.lower()}_calibration"
            os.makedirs(output_dir, exist_ok=True)
            
            # Serialize calibration data to JSON
            calibration_data = calibrator.to_json()
            output_path = os.path.join(output_dir, "calibration_results.json")
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
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
            
            # Draw reprojection analysis on images
            reprojection_debug_dir = os.path.join(output_dir, "reprojection_analysis")
            os.makedirs(reprojection_debug_dir, exist_ok=True)
            reprojection_images = calibrator.draw_reprojection_on_images()
            
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Reprojection analysis images: {len(reprojection_images)} images in {reprojection_debug_dir}")
    
    if not success:
        print(f"\n❌ GridBoard calibration failed!")
        print(f"   Could not calibrate camera parameters using GridBoard pattern")
        print(f"   Check that:")
        print(f"   - GridBoard markers are clearly detected in images")
        print(f"   - Pattern configuration matches the actual GridBoard")
        print(f"   - Images have good lighting and marker contrast")
        raise ValueError("GridBoard calibration failed")


def main():
    """Main function with proper error handling."""
    print("Intrinsic Camera Calibration Example")
    print("=" * 60)
    print("Camera calibration using JSON pattern configurations")
    print()
    
    success_count = 0
    total_tests = 3
    
    try:
        print("Testing chessboard calibration...")
        test_chessboard_calibration()
        success_count += 1
        print("✅ Chessboard calibration completed successfully")
    except Exception as e:
        print(f"❌ Chessboard calibration failed: {e}")
    
    try:
        print("\nTesting ChArUco calibration...")
        test_charuco_calibration()
        success_count += 1
        print("✅ ChArUco calibration completed successfully")
    except Exception as e:
        print(f"❌ ChArUco calibration failed: {e}")
    
    try:
        print("\nTesting Grid Board calibration...")
        test_gridboard_calibration()
        success_count += 1
        print("✅ Grid Board calibration completed successfully")
    except Exception as e:
        print(f"❌ Grid Board calibration failed: {e}")
    
    print(f"\n📊 Results: {success_count}/{total_tests} calibrations successful")
    
    if success_count == total_tests:
        print(f"All calibrations completed successfully!")
        print(f"   Results saved in: data/results/[pattern_id]_calibration/")
        print(f"   Pattern configurations loaded from JSON files in sample_data/")
        return 0
    elif success_count > 0:
        print(f"⚠️  Some calibrations failed. Check error messages above.")
        return 1
    else:
        print(f"❌ All calibrations failed!")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
