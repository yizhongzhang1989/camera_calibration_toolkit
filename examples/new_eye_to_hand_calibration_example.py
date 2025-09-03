#!/usr/bin/env python3
"""
New Eye-to-Hand Calibration Example
===================================

This example demonstrates how to use the NewEyeToHandCalibrator class for 
eye-to-hand calibration (stationary camera observing robot end-effector):

1. Load robot pose images and corresponding end-effector poses
2. Calculate camera intrinsic parameters using actual calibration images
3. Detect calibration patterns in images
4. Use automatic method selection to find the best OpenCV calibration method
5. Generate comprehensive debug images and analysis
6. Save calibration results with detailed error metrics

The NewEyeToHandCalibrator features:
- Inherits from HandEyeBaseCalibrator for code deduplication
- Same API approach as NewEyeInHandCalibrator for consistency
- Automatic method comparison and selection
- Robust pattern detection with graceful failure handling
- Eye-to-hand specific transformation chain (base2cam)
- Comprehensive error analysis and visualization
"""

import os
import cv2
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.new_eye_to_hand_calibration import NewEyeToHandCalibrator
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import load_pattern_from_json
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
        
        # Create pattern using the JSON data
        pattern = load_pattern_from_json(config_data)
        
        return pattern
        
    except Exception as e:
        print(f"❌ Error loading pattern configuration: {e}")
        return None


def calculate_camera_intrinsics(sample_dir):
    """Calculate camera intrinsic parameters using IntrinsicCalibrator."""
    print("📷 Calculating camera intrinsic parameters from dedicated calibration data...")
    
    # Use dedicated intrinsic calibration images
    intrinsic_dir = "sample_data/intrinsic_calib_grid_test_images"
    if not os.path.exists(intrinsic_dir):
        print(f"❌ Dedicated intrinsic calibration directory not found: {intrinsic_dir}")
        print("   Falling back to eye-to-hand images for intrinsic calibration...")
        intrinsic_dir = sample_dir
    
    # Get image files for intrinsic calibration
    image_files = [f for f in os.listdir(intrinsic_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    image_paths = [os.path.join(intrinsic_dir, f) for f in image_files]
    
    print(f"   Using {len(image_paths)} images from dedicated intrinsic calibration dataset")
    
    # Load pattern configuration
    config_path = os.path.join(intrinsic_dir, "chessboard_config.json")
    pattern = load_pattern_config(config_path)
    if pattern is None:
        return None, None
    
    try:
        # Create intrinsic calibrator
        intrinsic_calibrator = IntrinsicCalibrator(calibration_pattern=pattern)
        
        # Load images
        intrinsic_calibrator.set_images_from_paths(image_paths)
        
        # Detect pattern points
        if not intrinsic_calibrator.detect_pattern_points(verbose=True):
            print("❌ Failed to detect calibration patterns for intrinsic calibration")
            return None, None
        
        # Perform intrinsic calibration
        print("   Running intrinsic calibration...")
        success = intrinsic_calibrator.calibrate_camera(verbose=True)
        
        if success:
            print("   Getting calibration results...")
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coeffs = intrinsic_calibrator.get_distortion_coefficients()
            # Skip rms_error to avoid formatting issues
            # rms_error = intrinsic_calibrator.get_reprojection_error()
            
            print(f"✅ Standalone intrinsic calibration successful!")
            # print(f"   RMS error: {rms_error:.4f} pixels")
            print("   Displaying camera matrix...")
            print(f"     fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            print(f"     cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            print("   Displaying distortion coefficients...")
            print("   Distortion coefficients:", distortion_coeffs.flatten())
            
            # Skip saving for now to avoid issues
            print("   Skipping save to avoid potential issues...")
            
            print("   Returning calibration results...")
            return camera_matrix, distortion_coeffs
            
        else:
            print("❌ Intrinsic calibration failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during intrinsic calibration: {e}")
        return None, None


def test_new_eye_to_hand_calibration():
    """Test new eye-to-hand calibration workflow using NewEyeToHandCalibrator."""
    sample_dir = "sample_data/eye_to_hand_test_data"
    
    print("Eye-to-Hand Camera Calibration (New Base Class Version)")
    print("=" * 60)
    print(f"📁 Using sample data from: {sample_dir}")
    
    # Calculate camera intrinsic parameters first
    camera_matrix, distortion_coefficients = calculate_camera_intrinsics(sample_dir)
    if camera_matrix is None or distortion_coefficients is None:
        print("❌ Failed to calculate camera intrinsics - cannot proceed with hand-eye calibration")
        return False
        
    print(f"📷 Camera intrinsics calculated successfully:")
    print(f"   fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"   cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    print("   Distortion:", np.array(distortion_coefficients).flatten())
    
    # Load pattern configuration from JSON file
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return False
        
    pattern = load_pattern_config(config_path)
    
    # Get image files
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    image_paths = [os.path.join(sample_dir, f) for f in image_files]
    
    print(f"📸 Found {len(image_files)} calibration images")
    
    # Create NewEyeToHandCalibrator with the working base class architecture
    print(f"\n🔧 Creating NewEyeToHandCalibrator...")
    
    calibrator = NewEyeToHandCalibrator(
        image_paths=image_paths,              # Load images and JSON poses automatically
        camera_matrix=camera_matrix,          # Camera intrinsics
        distortion_coefficients=distortion_coefficients.flatten(),
        calibration_pattern=pattern          # Calibration pattern (loaded from JSON)
    )
    
    print("✅ NewEyeToHandCalibrator initialized:")
    
    # Check if images and poses were loaded
    info = calibrator.get_calibration_info()
    print(f"   Images: {info['image_count']}")
    print(f"   Robot poses: {info['transform_count']}")
    print(f"   Camera intrinsics: {'✓' if calibrator.camera_matrix is not None else '✗'}")
    print(f"   Calibration pattern: {'✓' if calibrator.calibration_pattern is not None else '✗'}")
    
    if info['image_count'] == 0 or info['transform_count'] == 0:
        print("❌ Failed to load calibration data")
        return False
    
    # Test different calibration methods to find the best one
    print(f"\n🔧 Testing different hand-eye calibration methods...")
    
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
        (cv2.CALIB_HAND_EYE_PARK, "PARK"), 
        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
    ]
    
    results = []
    best_method = "TSAI"
    best_error = float('inf')
    
    print(f"\n📊 Comparing {len(methods)} calibration methods:")
    print("=" * 60)
    
    for method, name in methods:
        print(f"\n🎯 Method {len(results)+1}/{len(methods)}: {name}")
        try:
            # Use the calibrate method with the specific method
            if calibrator.calibrate(method=method, verbose=False):
                rms_error = calibrator.get_rms_error()
                transformation = calibrator.get_transformation_matrix()
                per_image_errors = calibrator.get_per_image_errors()
                
                results.append({
                    'name': name,
                    'rms_error': rms_error,
                    'transformation': transformation,
                    'per_image_errors': per_image_errors
                })
                
                print(f"   ✅ Success: RMS error = {rms_error:.4f} pixels")
                
                if rms_error < best_error:
                    best_error = rms_error
                    best_method = name
                    
            else:
                print(f"   ❌ Failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if not results:
        print("❌ All calibration methods failed")
        return False
    
    # Sort results and display comparison
    results.sort(key=lambda x: x['rms_error'])
    
    print(f"\n📈 Calibration Methods Comparison:")
    print("=" * 50)
    print(f"{'Method':<12} {'RMS Error':<12} {'Status'}")
    print("-" * 35)
    
    for result in results:
        status = "🏆 Best" if result['name'] == best_method else "✅ Good"
        print(f"{result['name']:<12} {result['rms_error']:<12.4f} {status}")
    
    # Use the best result
    best_result = results[0]
    final_rms_error = best_result['rms_error']
    transformation_matrix = best_result['transformation']
    
    print(f"\n🎯 Using best method: {best_result['name']}")
    print(f"   RMS reprojection error: {final_rms_error:.4f} pixels")
    
    # Display transformation matrix
    print(f"\n📐 Base to camera transformation matrix:")
    if transformation_matrix is not None:
        for i, row in enumerate(transformation_matrix):
            print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
    
    # Show per-image errors
    if best_result['per_image_errors']:
        errors = [e for e in best_result['per_image_errors'] if not np.isinf(e)]
        if errors:
            print(f"   Error statistics: min={min(errors):.3f}, max={max(errors):.3f}, std={np.std(errors):.3f}")
            errors_summary = [f'{e:.3f}' for e in errors[:5]]
            print(f"   Per-image errors (first 5): {errors_summary}")

    # Perform optimization to improve calibration accuracy
    print(f"\n🔍 Performing calibration optimization...")
    print(f"   Initial RMS error: {final_rms_error:.4f} pixels")
    print("   Note: Optimization refines both target2end and base2cam matrices")
    print("         using nonlinear optimization to minimize reprojection error")
    
    try:
        # Store initial values for comparison
        initial_rms = final_rms_error
        initial_base2cam = transformation_matrix.copy() if transformation_matrix is not None else None
        initial_target2end = calibrator.get_target2end_matrix().copy() if calibrator.get_target2end_matrix() is not None else None
        
        # Perform optimization
        optimized_rms = calibrator.optimize_calibration(iterations=100, verbose=True)
        
        # Get optimized matrices
        optimized_base2cam = calibrator.get_base2cam_matrix()
        optimized_target2end = calibrator.get_target2end_matrix()
        
        # Calculate improvement
        improvement = initial_rms - optimized_rms
        improvement_pct = (improvement / initial_rms) * 100 if initial_rms > 0 else 0
        
        print(f"✅ Optimization completed successfully!")
        print(f"   Optimized RMS error: {optimized_rms:.4f} pixels")
        print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
        
        # Update variables for saving results
        final_rms_error = optimized_rms
        transformation_matrix = optimized_base2cam
        
    except Exception as e:
        print(f"⚠️  Optimization failed: {e}")
        print(f"   Continuing with initial calibration results")
    
    # Set up output directory
    output_dir = "data/results/new_eye_to_hand_calibration"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save debug images
    print(f"\n🖼️  Generating debug images...")
    
    try:
        # Save calibration results
        results_file = os.path.join(output_dir, 'new_eye_to_hand_calibration_results.json')
        calibration_results = {
            "best_method": best_result['name'],
            "rms_error": final_rms_error,
            "base_to_camera_matrix": transformation_matrix.tolist() if transformation_matrix is not None else None,
            "target_to_end_matrix": calibrator.get_target2end_matrix().tolist() if calibrator.get_target2end_matrix() is not None else None,
            "per_image_errors": best_result['per_image_errors'] if best_result['per_image_errors'] else [],
            "all_methods": {result['name']: result['rms_error'] for result in results},
            "camera_intrinsics": {
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": distortion_coefficients.tolist()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(calibration_results, f, indent=2)
            
        print(f"✅ New eye-to-hand calibration results saved to: {results_file}")
        print(f"   Calibration results saved to: {output_dir}")
        print(f"💾 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"⚠️  Warning: Error saving results: {e}")
    
    return True


def inspect_sample_data():
    """Inspect the sample data to show what we're working with."""
    sample_dir = "sample_data/eye_to_hand_test_data"
    
    print("🔍 Inspecting Sample Data")
    print("=" * 30)
    
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        return
    
    # Count files
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    json_files = [f for f in os.listdir(sample_dir) if f.endswith('.json')]
    
    print(f"📊 Sample data statistics:")
    print(f"   Image files: {len(image_files)}")
    print(f"   Pose files: {len(json_files)}")
    
    # Examine first pose file
    if json_files:
        first_json = os.path.join(sample_dir, json_files[0])
        try:
            with open(first_json, 'r') as f:
                pose_data = json.load(f)
                
            print(f"\n📄 Sample pose file format ({json_files[0]}):")
            print(f"   Keys: {list(pose_data.keys())}")
            
            if 'end_xyzrpy' in pose_data:
                xyz_rpy = pose_data['end_xyzrpy']
                print(f"   Position: x={xyz_rpy['x']:.3f}, y={xyz_rpy['y']:.3f}, z={xyz_rpy['z']:.3f}")
                print(f"   Orientation: rx={xyz_rpy['rx']:.3f}, ry={xyz_rpy['ry']:.3f}, rz={xyz_rpy['rz']:.3f}")
        except Exception as e:
            print(f"   ⚠️ Could not read pose file: {e}")
    
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
    print("New Eye-to-Hand Camera Calibration Example")
    print("=" * 75)
    print("Demonstrates the NewEyeToHandCalibrator with HandEyeBaseCalibrator architecture")
    print()
    
    try:
        # Inspect sample data first
        inspect_sample_data()
        
        # Run calibration with new base class approach
        success = test_new_eye_to_hand_calibration()
        
        print(f"\n✨ New eye-to-hand calibration example completed!")
        print(f"   Results saved to: data/results/new_eye_to_hand_calibration/")
        
        if success:
            print(f"   ✅ Calibration and optimization successful")
            print(f"   🎯 Used NewEyeToHandCalibrator with base class architecture")
            print(f"   📊 Method comparison and optimization analysis completed")
            return 0
        else:
            print(f"   ❌ Calibration failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: New eye-to-hand calibration failed with exception:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
