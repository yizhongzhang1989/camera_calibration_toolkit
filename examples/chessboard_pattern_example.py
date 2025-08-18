"""
Chessboard Pattern Examples
===========================

This example demonstrates how to use different chessboard patterns
for camera calibration including standard chessboards and ChArUco boards.

Usage:
    conda activate camcalib
    python examples/chessboard_pattern_example.py
"""

import sys
import os
import numpy as np
import cv2

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration_patterns import (
    CalibrationPatternManager, 
    create_chessboard_pattern,
    get_common_pattern,
    COMMON_PATTERNS
)
from core.intrinsic_calibration import IntrinsicCalibrator


def demonstrate_pattern_creation():
    """Demonstrate creating different chessboard patterns."""
    print("📋 Chessboard Pattern Examples")
    print("="*50)
    
    # Create calibration pattern manager
    manager = CalibrationPatternManager()
    
    print(f"Available pattern types: {manager.get_available_patterns()}")
    print()
    
    # Example 1: Standard chessboard
    print("1. Standard Chessboard:")
    standard_pattern = create_chessboard_pattern(
        "standard",
        width=11,
        height=8,
        square_size=0.020  # 20mm squares
    )
    print(f"   Pattern info: {standard_pattern.get_info()}")
    print()
    
    # Example 2: ChArUco board
    print("2. ChArUco Board:")
    try:
        charuco_pattern = create_chessboard_pattern(
            "charuco",
            width=5,
            height=7,
            square_size=0.040,  # 40mm squares
            marker_size=0.020   # 20mm markers
        )
        print(f"   Pattern info: {charuco_pattern.get_info()}")
    except Exception as e:
        print(f"   ChArUco creation failed (OpenCV version may not support it): {e}")
    print()
    
    # Example 3: Common patterns
    print("3. Common Pre-configured Patterns:")
    for pattern_name, config in COMMON_PATTERNS.items():
        print(f"   {pattern_name}: {config['description']}")
    print()


def test_pattern_detection(image_path: str):
    """Test pattern detection on a sample image."""
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"🔍 Testing Pattern Detection on: {os.path.basename(image_path)}")
    print("-" * 50)
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load test image")
        return
    
    # Test standard chessboard detection
    print("Testing Standard Chessboard (11x8):")
    standard_pattern = create_chessboard_pattern(
        "standard",
        width=11,
        height=8,
        square_size=0.020
    )
    
    success, corners, point_ids = standard_pattern.detect_corners(image)
    print(f"   Detection result: {'✅ Success' if success else '❌ Failed'}")
    if success:
        print(f"   Detected {len(corners)} corners")
    print()


def calibrate_with_patterns(sample_data_dir: str):
    """Demonstrate calibration with different patterns."""
    print("🎯 Camera Calibration with Different Patterns")
    print("="*50)
    
    # Check if sample data exists
    if not os.path.exists(sample_data_dir):
        print(f"❌ Sample data directory not found: {sample_data_dir}")
        print("Please ensure sample_data/hand_in_eye_test_data/ exists")
        return
    
    # Get sample images
    import glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(sample_data_dir, extension)))
        image_paths.extend(glob.glob(os.path.join(sample_data_dir, extension.upper())))
    
    # Remove duplicates
    image_paths = list(set(image_paths))
    image_paths.sort()
    
    if not image_paths:
        print("❌ No sample images found")
        return
    
    print(f"Found {len(image_paths)} sample images")
    
    # Test 1: Standard chessboard calibration
    print("\n1. Calibrating with Standard Chessboard (11x8):")
    
    # Create standard chessboard pattern
    standard_pattern = create_chessboard_pattern(
        "standard",
        width=11,
        height=8,
        square_size=0.020
    )
    
    calibrator1 = IntrinsicCalibrator(
        image_paths=image_paths,
        calibration_pattern=standard_pattern,
        pattern_type='standard'
    )
    
    try:
        if calibrator1.detect_pattern_points(verbose=True):
            rms_error = calibrator1.calibrate_camera(verbose=True)
            
            if rms_error > 0:
                success1 = True
                mtx1 = calibrator1.get_camera_matrix()
                dist1 = calibrator1.get_distortion_coefficients()
                print("   ✅ Standard chessboard calibration successful!")
                print(f"   Camera focal lengths: fx={mtx1[0,0]:.2f}, fy={mtx1[1,1]:.2f}")
                print(f"   RMS Error: {rms_error:.4f}")
            else:
                success1 = False
                print("   ❌ Standard chessboard calibration failed")
        else:
            success1 = False
            print("   ❌ Pattern detection failed")
            
    except Exception as e:
        success1 = False
        print(f"   ❌ Error during standard calibration: {e}")
    
    print()
    
    # Test 2: Compare with new API method
    print("2. Testing new API method:")
    
    try:
        # Create calibration pattern
        pattern = create_chessboard_pattern(
            pattern_type='standard',
            width=11,
            height=8,
            square_size=0.020
        )
        
        calibrator2 = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        if calibrator2.detect_pattern_points(verbose=True):
            rms_error = calibrator2.calibrate_camera(verbose=True)
            
            if rms_error > 0:
                mtx2 = calibrator2.get_camera_matrix()
                dist2 = calibrator2.get_distortion_coefficients()
                print("   ✅ New API method calibration successful!")
                print(f"   Camera focal lengths: fx={mtx2[0,0]:.2f}, fy={mtx2[1,1]:.2f}")
                print(f"   RMS Error: {rms_error:.4f}")
                
                # Compare results
                if success1:
                    mtx_diff = np.abs(mtx1 - mtx2).max()
                    dist_diff = np.abs(dist1 - dist2).max()
                    print(f"   📊 Comparison - Max matrix diff: {mtx_diff:.6f}, Max dist diff: {dist_diff:.6f}")
                    
            else:
                print("   ❌ New API method calibration failed")
        else:
            print("   ❌ Pattern detection failed with new API method")
            
    except Exception as e:
        print(f"   ❌ Error during new API calibration: {e}")
            
    except Exception as e:
        print(f"   ❌ Error during legacy calibration: {e}")


def main():
    """Main demonstration function."""
    print("🏁 Chessboard Pattern Abstraction Demo")
    print("="*60)
    print()
    
    # Demonstrate pattern creation
    demonstrate_pattern_creation()
    
    # Test pattern detection (if sample image exists)
    toolkit_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_data_dir = os.path.join(toolkit_root, "sample_data", "hand_in_eye_test_data")
    
    # Try to find a sample image for detection test
    sample_images = []
    if os.path.exists(sample_data_dir):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            sample_images.extend(glob.glob(os.path.join(sample_data_dir, ext)))
    
    if sample_images:
        test_pattern_detection(sample_images[0])
        calibrate_with_patterns(sample_data_dir)
    else:
        print("ℹ️  No sample images found for detection testing")
        print(f"   Expected location: {sample_data_dir}")
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("- Use create_chessboard_pattern() to create pattern objects")
    print("- Use the new API with pattern creation and calibrator constructor")  
    print("- Call detect_pattern_points() then calibrate_camera() for calibration")  
    print("- Extend ChessboardPattern for custom pattern types")


if __name__ == "__main__":
    main()
