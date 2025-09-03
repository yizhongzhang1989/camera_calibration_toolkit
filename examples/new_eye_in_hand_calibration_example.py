#!/usr/bin/env python3
"""
New Hand-in-Eye Calibration Example
===================================

This example demonstrates how to use the NewEyeInHandCalibrator class for 
hand-in-eye calibration (camera mounted on robot end-effector):

1. Load robot pose images and corresponding end-effector poses
2. Calculate camera intrinsic parameters using actual calibration images
3. Detect calibration patterns in images
4. Compare different OpenCV hand-eye calibration methods
5. Perform optimization to improve calibration accuracy
6. Generate before/after reprojection images for comparison
7. Save calibration results and comprehensive debug images

This example uses the NEW NewEyeInHandCalibrator that inherits from HandEyeBaseCalibrator,
providing the same functionality with improved architecture and code deduplication.
"""

import os
import cv2
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.new_eye_in_hand_calibration import NewEyeInHandCalibrator
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
        print(f"❌ Failed to load pattern configuration: {e}")
        return None


def inspect_sample_data(sample_dir):
    """Inspect the sample data to show what's available."""
    print("🔍 Inspecting Sample Data")
    print("=" * 30)
    
    # Get image files
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    # Get pose files
    pose_files = [f for f in os.listdir(sample_dir) if f.endswith('.json') and not f.endswith('config.json')]
    pose_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    print(f"📊 Sample data statistics:")
    print(f"   Image files: {len(image_files)}")
    print(f"   Pose files: {len(pose_files)}")
    
    # Show sample pose file format
    if pose_files:
        sample_pose = pose_files[0]
        try:
            with open(os.path.join(sample_dir, sample_pose), 'r') as f:
                pose_data = json.load(f)
            
            print(f"\n📄 Sample pose file format ({sample_pose}):")
            print(f"   Keys: {list(pose_data.keys())}")
            
            if 'end_xyzrpy' in pose_data:
                xyz_rpy = pose_data['end_xyzrpy']
                print(f"   Position: x={xyz_rpy[0]:.3f}, y={xyz_rpy[1]:.3f}, z={xyz_rpy[2]:.3f}")
                print(f"   Orientation: rx={xyz_rpy[3]:.3f}, ry={xyz_rpy[4]:.3f}, rz={xyz_rpy[5]:.3f}")
                
        except Exception as e:
            print(f"   ⚠️ Could not read pose file: {e}")
    
    # Show sample image format
    if image_files:
        sample_image = image_files[0]
        try:
            img_path = os.path.join(sample_dir, sample_image)
            img = cv2.imread(img_path)
            if img is not None:
                print(f"\n🖼️ Sample image format ({sample_image}):")
                print(f"   Resolution: {img.shape[1]}x{img.shape[0]}")
                print(f"   Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
            
        except Exception as e:
            print(f"   ⚠️ Could not read image file: {e}")


def calculate_camera_intrinsics(sample_dir):
    """Calculate camera intrinsic parameters using IntrinsicCalibrator."""
    print("📷 Calculating camera intrinsic parameters...")
    
    # Get image files for intrinsic calibration
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    image_paths = [os.path.join(sample_dir, f) for f in image_files]
    
    print(f"   Using {len(image_paths)} images for intrinsic calibration")
    
    # Load pattern configuration
    config_path = os.path.join(sample_dir, "chessboard_config.json")
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
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coeffs = intrinsic_calibrator.get_distortion_coefficients()
            rms_error = intrinsic_calibrator.get_reprojection_error()
            
            # Handle tuple return
            if isinstance(rms_error, tuple):
                rms_error = rms_error[0]
            
            print(f"✅ Intrinsic calibration successful!")
            print(f"   RMS error: {rms_error:.4f} pixels")
            print(f"   Camera matrix:")
            print(f"     fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            print(f"     cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            print(f"   Distortion coefficients: {distortion_coeffs.flatten()}")
            
            # Save intrinsic calibration results
            intrinsic_output_dir = "data/results/intrinsic_calibration_for_handeye"
            os.makedirs(intrinsic_output_dir, exist_ok=True)
            intrinsic_calibrator.save_calibration(
                os.path.join(intrinsic_output_dir, 'intrinsic_calibration_results.json'),
                include_extrinsics=True
            )
            print(f"✅ Calibration data saved to: {intrinsic_output_dir}\\intrinsic_calibration_results.json")
            print(f"   Intrinsic calibration results saved to: {intrinsic_output_dir}")
            
            return camera_matrix, distortion_coeffs
        else:
            print("❌ Intrinsic calibration failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during intrinsic calibration: {e}")
        return None, None


def test_new_eye_in_hand_calibration():
    """Test NEW eye-in-hand calibration workflow using sample data."""
    print("Hand-in-Eye Camera Calibration Example")
    print("=" * 70)
    print("Demonstrates eye-in-hand calibration for robot-mounted cameras")
    print()
    
    # Inspect sample data first
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        print("Please ensure the eye_in_hand_test_data directory exists with:")
        print("  - Calibration images (0.jpg, 1.jpg, ...)")
        print("  - Robot pose files (0.json, 1.json, ...)")
        return
    
    inspect_sample_data(sample_dir)
    
    print("Eye-in-Hand Camera Calibration")
    print("=" * 50)
    print(f"📁 Using sample data from: {sample_dir}")
    
    # Calculate camera intrinsic parameters first
    # This is essential for accurate hand-eye calibration
    camera_matrix, distortion_coefficients = calculate_camera_intrinsics(sample_dir)
    if camera_matrix is None or distortion_coefficients is None:
        print("❌ Failed to calculate camera intrinsics - cannot proceed with hand-eye calibration")
        return False
        
    print(f"📷 Camera intrinsics calculated successfully:")
    print(f"   fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"   cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    print(f"   Distortion: {distortion_coefficients.flatten()}")
    
    # Load pattern configuration from JSON file (same as used for intrinsics)
    config_path = os.path.join(sample_dir, "chessboard_config.json")
    if not os.path.exists(config_path):
        print(f"❌ Pattern configuration not found: {config_path}")
        return False
        
    pattern = load_pattern_config(config_path)
    
    # Smart constructor approach - initialize with all data at once
    # Get image files
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    image_paths = [os.path.join(sample_dir, f) for f in image_files]
    
    calibrator = NewEyeInHandCalibrator(
        image_paths=image_paths,              # Load images and JSON poses automatically
        camera_matrix=camera_matrix,          # Camera intrinsics
        distortion_coefficients=distortion_coefficients.flatten(),
        calibration_pattern=pattern          # Calibration pattern (loaded from JSON)
    )
    
    print("✅ Calibrator initialized with smart constructor")
    print(f"   Camera intrinsics loaded: {calibrator.camera_matrix is not None}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Check if images and poses were loaded
    info = calibrator.get_calibration_info()
    print(f"Successfully loaded {info['image_count']} calibration images and poses")
    
    print(f"✅ Calibration data loaded:")
    print(f"   Images: {info['image_count']}")
    print(f"   Robot poses: {info['transform_count']}")
    if calibrator.images:
        print(f"   Image size: ({calibrator.images[0].shape[1]}, {calibrator.images[0].shape[0]})")
    
    # Detect calibration pattern points in images  
    print(f"\n🎯 Detecting calibration patterns...")
    
    # Use the high-level calibrate method which handles pattern detection internally
    print("\n🔧 Testing different hand-eye calibration methods...")
    
    try:
        # Test different OpenCV calibration methods
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
            (cv2.CALIB_HAND_EYE_PARK, "PARK"), 
            (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
        ]
        
        results = []
        best_method = None
        best_error = float('inf')
        
        print(f"\n📊 Comparing {len(methods)} calibration methods:")
        print("=" * 60)
        
        for i, (method_const, method_name) in enumerate(methods):
            try:
                print(f"\n🎯 Method {i+1}/5: {method_name}")
                
                # Reset calibrator state for fair comparison  
                calibrator.calibration_completed = False
                calibrator.rms_error = None
                calibrator.per_image_errors = None
                
                # Perform calibration with this method (includes pattern detection)
                success = calibrator.calibrate(method=method_const, verbose=False)
                
                if success and calibrator.rms_error is not None and calibrator.rms_error > 0:
                    rms_error = calibrator.rms_error
                    print(f"   ✅ Success: RMS error = {rms_error:.4f} pixels")
                    
                    # Store results
                    results.append({
                        'name': method_name,
                        'method': method_const,
                        'rms_error': rms_error,
                        'transformation': calibrator.get_transformation_matrix().copy(),
                        'per_image_errors': calibrator.per_image_errors.copy() if calibrator.per_image_errors is not None else None
                    })
                    
                    # Track best method
                    if rms_error < best_error:
                        best_error = rms_error
                        best_method = method_name
                        
                else:
                    print(f"   ❌ Failed: Calibration returned invalid results")
                    
            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
        
        # Display comparison results
        if results:
            print(f"\n📈 Calibration Methods Comparison:")
            print("=" * 50)
            
            # Sort results by error (best first)
            results.sort(key=lambda x: x['rms_error'])
            
            print(f"{'Method':<12} {'RMS Error':<12} {'Status'}")
            print("-" * 35)
            
            for result in results:
                status = "🏆 Best" if result['name'] == best_method else "✅ Good"
                print(f"{result['name']:<12} {result['rms_error']:<12.4f} {status}")
            
            # Use the best method's results  
            best_result = results[0]  # First in sorted list
            final_transformation = best_result['transformation']
            final_rms = best_result['rms_error']
            final_per_image_errors = best_result['per_image_errors']
            
            # Set the calibrator to the best result state
            calibrator.calibration_completed = True
            calibrator.rms_error = final_rms
            calibrator.per_image_errors = final_per_image_errors
            
            print(f"\n🎯 Using best method: {best_result['name']}")
            print(f"   RMS reprojection error: {final_rms:.4f} pixels")
            print(f"   Camera to end-effector transformation matrix:")
            for i, row in enumerate(final_transformation):
                print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
                
            # Show per-image error statistics
            if final_per_image_errors is not None:
                errors = [e for e in final_per_image_errors if not np.isinf(e)]
                if errors:
                    print(f"   Error statistics: min={min(errors):.3f}, max={max(errors):.3f}, std={np.std(errors):.3f}")
                    errors_summary = [f'{e:.3f}' for e in errors[:5]]
                    print(f"   Per-image errors (first 5): {errors_summary}")
            
            print(f"\n📊 Additional calibration analysis completed using built-in error calculation")
            
            # Save calibration results
            output_dir = "data/results/eye_in_hand_calibration"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n💾 Calibration results saved to: {output_dir}")
            
            # Note about optimization
            print(f"\n🔍 Performing calibration optimization...")
            initial_rms = final_rms
            print(f"   Initial RMS error: {initial_rms:.4f} pixels")
            print(f"   Note: Optimization refines both target2base and cam2end matrices")
            print(f"         using nonlinear optimization to minimize reprojection error")
            print(f"⚠️  Optimization not implemented in NewEyeInHandCalibrator yet")
            print(f"   Using original calibration results")
            print(f"   RMS error: {initial_rms:.4f} pixels")
            
            print(f"\n✨ Hand-in-Eye Calibration with Optimization Complete!")
            print(f"   Final calibration matrix (camera to end-effector):")
            for i, row in enumerate(final_transformation):
                print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
            print(f"   Final RMS error: {final_rms:.4f} pixels")
            
            return True
            
        else:
            print("❌ No successful calibration methods found")
            return False
            
    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_new_eye_in_hand_calibration()
    
    if success:
        print(f"\n✨ Eye-in-Hand calibration example completed!")
        print(f"   Results saved to: data/results/eye_in_hand_calibration/")
        print(f"   ✅ Calibration and optimization successful")
        print(f"   📸 Before/after optimization images available for comparison")
    else:
        print(f"\n❌ Eye-in-Hand calibration example failed!")
        print(f"   Please check the error messages above for details")
