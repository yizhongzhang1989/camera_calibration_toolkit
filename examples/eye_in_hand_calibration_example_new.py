#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Example (New Unified Version)
====================================================

This example demonstrates how to use the new unified HandEyeCalibrator class for 
eye-in-hand calibration (camera mounted on robot end-effector):

1. Load robot pose images and corresponding end-effector poses
2. Calculate camera intrinsic parameters using actual calibration images
3. Detect calibration patterns in images
4. Use automatic method selection to find the best OpenCV calibration method
5. Generate comprehensive debug images and analysis
6. Save calibration results with detailed error metrics

The new HandEyeCalibrator features:
- Unified interface for both eye-in-hand and eye-to-hand calibration
- Automatic method selection based on reprojection error comparison
- Robust pattern detection with graceful failure handling
- Streamlined workflow with optimal performance
- Comprehensive error analysis and visualization
"""

import os
import cv2
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hand_eye_calibration import HandEyeCalibrator
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import get_pattern_manager
from core.utils import load_images_from_directory


def load_pattern_config(config_path):
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
        
        # Create pattern using the pattern manager
        pattern_manager = get_pattern_manager()
        
        # Map JSON pattern_id to pattern manager names
        pattern_type_map = {
            'chessboard': 'standard_chessboard',
            'standard_chessboard': 'standard_chessboard',
            'charuco_board': 'charuco_board',
            'grid_board': 'grid_board'
        }
        
        pattern_type = pattern_type_map.get(config_data.get('pattern_id', 'chessboard'), 'standard_chessboard')
        
        # Create pattern with appropriate parameters
        # Handle both direct parameters and nested "parameters" key
        if 'parameters' in config_data:
            params = config_data['parameters']
        else:
            params = config_data
        
        if pattern_type == 'standard_chessboard':
            pattern = pattern_manager.create_pattern(
                pattern_type,
                width=params.get('width', params.get('board_width', 9)),
                height=params.get('height', params.get('board_height', 6)),
                square_size=params.get('square_size', 0.025)
            )
        else:
            # For other pattern types, use available parameters
            pattern = pattern_manager.create_pattern(
                pattern_type,
                **{k: v for k, v in params.items() if k not in ['name', 'pattern_id']}
            )
        
        return pattern
        
    except Exception as e:
        print(f"❌ Failed to load pattern configuration: {e}")
        raise


def calculate_camera_intrinsics(sample_dir):
    """Calculate camera intrinsic parameters using IntrinsicCalibrator.
    
    This performs actual intrinsic calibration using the same images that will be
    used for hand-eye calibration, ensuring the camera parameters are accurate.
    
    Args:
        sample_dir: Directory containing calibration images
        
    Returns:
        tuple: (camera_matrix, distortion_coefficients) or (None, None) if failed
    """
    print("📷 Calculating camera intrinsic parameters...")
    
    try:
        # Load images for intrinsic calibration
        image_paths = load_images_from_directory(sample_dir)
        if not image_paths:
            print("❌ No images found for intrinsic calibration")
            return None, None
        
        print(f"   Using {len(image_paths)} images for intrinsic calibration")
        
        # Load pattern configuration from JSON file
        config_path = os.path.join(sample_dir, "chessboard_config.json")
        if not os.path.exists(config_path):
            print(f"❌ Pattern configuration not found: {config_path}")
            return None, None
        
        pattern = load_pattern_config(config_path)
        
        # Create IntrinsicCalibrator with smart constructor
        intrinsic_calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Detect pattern points
        if not intrinsic_calibrator.detect_pattern_points(verbose=True):
            print("❌ Failed to detect calibration patterns for intrinsic calibration")
            return None, None
        
        # Perform intrinsic calibration
        print("   Running intrinsic calibration...")
        success = intrinsic_calibrator.calibrate_camera(verbose=True)
        
        if success:
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coeffs = intrinsic_calibrator.get_distortion_coefficients()
            # Ensure distortion coefficients are 1D
            if distortion_coeffs.ndim > 1:
                distortion_coeffs = distortion_coeffs.flatten()
            rms_error = intrinsic_calibrator.get_rms_error()
            
            print(f"✅ Intrinsic calibration successful!")
            print(f"   RMS error: {rms_error:.4f} pixels")
            print(f"   Camera matrix:")
            print(f"     fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            print(f"     cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            print(f"   Distortion coefficients: {distortion_coeffs.flatten()}")
            
            # Save intrinsic calibration results
            intrinsic_output_dir = "data/results/intrinsic_calibration_for_handeye_new"
            os.makedirs(intrinsic_output_dir, exist_ok=True)
            intrinsic_calibrator.save_calibration(
                os.path.join(intrinsic_output_dir, 'intrinsic_calibration_results.json'),
                include_extrinsics=True
            )
            print(f"   Intrinsic calibration results saved to: {intrinsic_output_dir}")
            
            return camera_matrix, distortion_coeffs
        else:
            print("❌ Intrinsic calibration failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during intrinsic calibration: {e}")
        return None, None


def test_eye_in_hand_calibration_new():
    """Test eye-in-hand calibration workflow using the new unified HandEyeCalibrator."""
    print("Eye-in-Hand Camera Calibration (New Unified Version)")
    print("=" * 60)
    
    # Check sample data directory
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        print("Please ensure the eye_in_hand_test_data directory exists with:")
        print("  - Calibration images (0.jpg, 1.jpg, ...)")
        print("  - Robot pose files (0.json, 1.json, ...)")
        return False
    
    print(f"📁 Using sample data from: {sample_dir}")
    
    # Calculate camera intrinsic parameters first
    camera_matrix, distortion_coefficients = calculate_camera_intrinsics(sample_dir)
    if camera_matrix is None or distortion_coefficients is None:
        print("❌ Failed to calculate camera intrinsics - cannot proceed with hand-eye calibration")
        return False
        
    print(f"📷 Camera intrinsics calculated successfully:")
    print(f"   fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"   cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    print(f"   Distortion: {distortion_coefficients.flatten()}")
    
    # Load pattern configuration
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return False
        
    pattern = load_pattern_config(config_path)
    
    # Get calibration image paths
    image_paths = load_images_from_directory(sample_dir)
    if not image_paths:
        print("❌ No calibration images found")
        return False
    
    print(f"📸 Found {len(image_paths)} calibration images")
    
    # Create unified HandEyeCalibrator using image_paths (it will auto-load poses from JSON)
    print(f"\n🔧 Creating unified HandEyeCalibrator...")
    calibrator = HandEyeCalibrator(
        calibration_type="eye_in_hand",  # Specify eye-in-hand calibration
        image_paths=image_paths,  # HandEyeCalibrator will auto-load images and poses from JSON
        calibration_pattern=pattern,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients
    )
    
    print("✅ Unified calibrator initialized:")
    print(f"   Calibration type: {calibrator.calibration_type}")
    print(f"   Robot poses: {len(calibrator.end2base_matrices) if calibrator.end2base_matrices else 0}")
    print(f"   Images loaded: {len(calibrator.images) if hasattr(calibrator, 'images') and calibrator.images else len(image_paths)}")
    print(f"   Camera intrinsics: {'✓' if calibrator.camera_matrix is not None else '✗'}")
    print(f"   Calibration pattern: {'✓' if calibrator.calibration_pattern is not None else '✗'}")
    
    # Perform calibration with automatic method selection
    print(f"\n🎯 Running calibration with automatic method selection...")
    success = calibrator.calibrate(verbose=True)
    
    if not success:
        print("❌ Calibration failed")
        return False
    
    # Display results
    print(f"\n✅ Calibration completed successfully!")
    print(f"   Best method: {calibrator.best_method_name}")
    print(f"   RMS reprojection error: {calibrator.rms_error:.4f} pixels")
    print(f"   Valid images used: {len(calibrator._valid_calibration_indices) if hasattr(calibrator, '_valid_calibration_indices') else 'N/A'}/{len(image_paths)}")
    
    # Display transformation matrix
    print(f"\n📐 Camera to end-effector transformation matrix:")
    if hasattr(calibrator, 'cam2end_matrix') and calibrator.cam2end_matrix is not None:
        for i, row in enumerate(calibrator.cam2end_matrix):
            print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
    
    # Show method comparison results
    print(f"\n📊 Method comparison results:")
    print(f"{'Method':<15} {'RMS Error':<12} {'Status'}")
    print("-" * 40)
    
    if hasattr(calibrator, 'method_results') and calibrator.method_results:
        for method_name, error in calibrator.method_results.items():
            status = "🏆 Best" if method_name == calibrator.best_method_name else "✓ Good"
            print(f"{method_name:<15} {error:<12.4f} {status}")
    else:
        print("   Method comparison results not available")
    
    # Set up output directory
    output_dir = "data/results/eye_in_hand_calibration_new"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save debug images
    print(f"\n🖼️  Generating debug images...")
    
    try:
        # Save calibration results using the correct method
        results_file = os.path.join(output_dir, 'calibration_results.json')
        if hasattr(calibrator, 'save_results'):
            calibrator.save_results(output_dir)
            print(f"   Calibration results saved to: {output_dir}")
        else:
            print(f"   ⚠️  save_results method not available")
        
        print(f"💾 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Warning: Error saving results: {e}")
    
    # Display summary statistics
    if hasattr(calibrator, '_valid_calibration_indices'):
        total_images = len(image_paths)
        failed_count = total_images - len(calibrator._valid_calibration_indices)
        if failed_count > 0:
            print(f"\n📈 Calibration statistics:")
            print(f"   Total images: {total_images}")
            print(f"   Successfully processed: {len(calibrator._valid_calibration_indices)}")
            print(f"   Failed pattern detection: {failed_count}")
            print(f"   Success rate: {(len(calibrator._valid_calibration_indices)/total_images)*100:.1f}%")
    
    return True


def inspect_sample_data():
    """Inspect the sample data to understand its format."""
    print("\n🔍 Inspecting Sample Data")
    print("=" * 30)
    
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        return
    
    # Count files
    image_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
    json_files = [f for f in os.listdir(sample_dir) if f.endswith('.json') and f != 'chessboard_config.json']
    
    print(f"📊 Sample data statistics:")
    print(f"   Image files: {len(image_files)}")
    print(f"   Pose files: {len(json_files)}")
    
    # Examine first pose file
    if json_files:
        first_json = os.path.join(sample_dir, json_files[0])
        with open(first_json, 'r') as f:
            pose_data = json.load(f)
            
        print(f"\n📄 Sample pose file format ({json_files[0]}):")
        print(f"   Keys: {list(pose_data.keys())}")
        
        if 'end_xyzrpy' in pose_data:
            xyz_rpy = pose_data['end_xyzrpy']
            print(f"   Position: x={xyz_rpy['x']:.3f}, y={xyz_rpy['y']:.3f}, z={xyz_rpy['z']:.3f}")
            print(f"   Orientation: rx={xyz_rpy['rx']:.3f}, ry={xyz_rpy['ry']:.3f}, rz={xyz_rpy['rz']:.3f}")
    
    # Check first image
    if image_files:
        first_image = os.path.join(sample_dir, image_files[0])
        img = cv2.imread(first_image)
        if img is not None:
            h, w = img.shape[:2]
            print(f"\n🖼️ Sample image format ({image_files[0]}):")
            print(f"   Resolution: {w}x{h}")
            print(f"   Channels: {img.shape[2] if len(img.shape) > 2 else 1}")


def main():
    """Main function with proper error handling."""
    print("Eye-in-Hand Camera Calibration Example (New Unified Version)")
    print("=" * 75)
    print("Demonstrates the new unified HandEyeCalibrator for eye-in-hand calibration")
    print()
    
    try:
        # Inspect sample data first
        inspect_sample_data()
        
        # Run calibration with new unified approach
        success = test_eye_in_hand_calibration_new()
        
        print(f"\n✨ Eye-in-hand calibration example completed!")
        print(f"   Results saved to: data/results/eye_in_hand_calibration_new/")
        
        if success:
            print(f"   ✅ Calibration successful with automatic method selection")
            print(f"   🎯 Used optimized unified HandEyeCalibrator class")
            print(f"   📊 Method comparison and error analysis completed")
            return 0
        else:
            print(f"   ❌ Calibration failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Eye-in-hand calibration failed with exception:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
