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

from core.chessboard_patterns import (
    ChessboardManager, 
    create_chessboard_pattern,
    get_common_pattern,
    COMMON_PATTERNS
)
from core.intrinsic_calibration import IntrinsicCalibrator


def demonstrate_pattern_creation():
    """Demonstrate creating different chessboard patterns."""
    print("📋 Chessboard Pattern Examples")
    print("="*50)
    
    # Create chessboard manager
    manager = ChessboardManager()
    
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
    
    success, corners = standard_pattern.detect_corners(image)
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
        print("Please ensure sample_data/intrinsic_calib_test_images/ exists")
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
    calibrator1 = IntrinsicCalibrator()
    
    # Create standard chessboard pattern
    standard_pattern = create_chessboard_pattern(
        "standard",
        width=11,
        height=8,
        square_size=0.020
    )
    
    try:
        success1, mtx1, dist1 = calibrator1.calibrate_with_pattern(
            image_paths=image_paths,
            chessboard_pattern=standard_pattern,
            verbose=True
        )
        
        if success1:
            print("   ✅ Standard chessboard calibration successful!")
            print(f"   Camera focal lengths: fx={mtx1[0,0]:.2f}, fy={mtx1[1,1]:.2f}")
        else:
            print("   ❌ Standard chessboard calibration failed")
            
    except Exception as e:
        print(f"   ❌ Error during standard calibration: {e}")
    
    print()
    
    # Test 2: Compare with legacy method
    print("2. Comparing with Legacy Method:")
    calibrator2 = IntrinsicCalibrator()
    
    try:
        success2, mtx2, dist2 = calibrator2.calibrate_from_images(
            image_paths=image_paths,
            XX=11,
            YY=8,
            L=0.020,
            verbose=True
        )
        
        if success2:
            print("   ✅ Legacy method calibration successful!")
            print(f"   Camera focal lengths: fx={mtx2[0,0]:.2f}, fy={mtx2[1,1]:.2f}")
            
            # Compare results
            if success1:
                mtx_diff = np.abs(mtx1 - mtx2).max()
                dist_diff = np.abs(dist1 - dist2).max()
                print(f"   📊 Comparison - Max matrix diff: {mtx_diff:.6f}, Max dist diff: {dist_diff:.6f}")
                
        else:
            print("   ❌ Legacy method calibration failed")
            
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
    sample_data_dir = os.path.join(toolkit_root, "sample_data", "intrinsic_calib_test_images")
    
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
    print("- Use calibrator.calibrate_with_pattern() for calibration")  
    print("- Extend ChessboardPattern for custom pattern types")


if __name__ == "__main__":
    main()
