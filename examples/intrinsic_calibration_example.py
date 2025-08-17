#!/usr/bin/env python3
"""
Intrinsic Camera Calibration Example
===================================

This example demonstrates how to use the IntrinsicCalibrator class for 
camera intrinsic parameter calibration using image paths workflow.
The class features a clean interface with smart constructor arguments 
and organized member variables.
"""

import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_chessboard_pattern
from core.utils import load_images_from_directory


def test_image_paths_workflow():
    """Test calibration workflow with image paths."""
    print("🔧 Calibration from Image Paths")
    print("=" * 50)
    
    # Load sample images
    sample_dir = os.path.join("sample_data", "intrinsic_calib_test_images")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        return
    
    image_paths = load_images_from_directory(sample_dir)[:5]
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
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            calibrator.save_calibration(
                os.path.join(output_dir, "intrinsic_calibration_paths.json"),
                include_extrinsics=True
            )
            print(f"   Calibration data saved to: results/intrinsic_calibration_paths.json")


if __name__ == "__main__":
    print("🚀 Intrinsic Camera Calibration Example")
    print("=" * 60)
    print("Camera calibration using image paths workflow")
    print()
    
    test_image_paths_workflow()
    
    print(f"\n✨ Calibration completed!")
