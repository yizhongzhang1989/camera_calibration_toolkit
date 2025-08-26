#!/usr/bin/env python3
"""
Eye-to-Hand Calibration Example
===============================

This example demonstrates how to use the EyeToHandCalibrator class for 
eye-to-hand calibration (stationary camera looking at robot end-effector):

1. Load robot pose images and corresponding end-effector poses
2. Calculate camera intrinsic parameters using actual calibration images
3. Detect calibration patterns in images
4. Compare different OpenCV hand-eye calibration methods
5. Generate reprojection images for analysis
6. Save calibration results and comprehensive debug images

The class features a clean interface with smart constructor arguments 
and organized member variables following the same pattern as IntrinsicCalibrator.
"""

import os
import cv2
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eye_to_hand_calibration import EyeToHandCalibrator
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


def calculate_camera_intrinsics_standalone():
    """Calculate camera intrinsic parameters using dedicated intrinsic calibration data.
    
    Uses sample_data/intrinsic_calib_grid_test_images which contains images with larger 
    calibration patterns more suitable for accurate intrinsic calibration.
    
    Returns:
        tuple: (camera_matrix, distortion_coefficients) or (None, None) if failed
    """
    print("📷 Calculating camera intrinsic parameters from dedicated calibration data...")
    
    try:
        # Use dedicated intrinsic calibration data
        intrinsic_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")
        if not os.path.exists(intrinsic_dir):
            print(f"❌ Intrinsic calibration data directory not found: {intrinsic_dir}")
            return None, None
        
        # Load images for intrinsic calibration
        image_paths = load_images_from_directory(intrinsic_dir)
        if not image_paths:
            print("❌ No images found for intrinsic calibration")
            return None, None
        
        print(f"   Using {len(image_paths)} images from dedicated intrinsic calibration dataset")
        
        # Load pattern configuration from JSON file
        config_path = os.path.join(intrinsic_dir, "chessboard_config.json")
        if not os.path.exists(config_path):
            print(f"❌ Pattern configuration not found: {config_path}")
            return None, None
        
        pattern, pattern_type = load_pattern_config(config_path)
        
        # Create IntrinsicCalibrator with smart constructor
        intrinsic_calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type=pattern_type
        )
        
        print(f"   Image size: {intrinsic_calibrator.image_size}")
        print(f"   Pattern type: {pattern_type}")
        
        # Detect pattern points
        if not intrinsic_calibrator.detect_pattern_points(verbose=True):
            print("❌ Failed to detect calibration patterns for intrinsic calibration")
            return None, None
        
        # Perform intrinsic calibration
        print("   Running intrinsic calibration...")
        rms_error = intrinsic_calibrator.calibrate_camera(verbose=True)
        
        if rms_error > 0:
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coeffs = intrinsic_calibrator.get_distortion_coefficients()
            
            print(f"✅ Standalone intrinsic calibration successful!")
            print(f"   RMS error: {rms_error:.4f} pixels")
            print(f"   Camera matrix:")
            print(f"     fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            print(f"     cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            print(f"   Distortion coefficients: {distortion_coeffs.flatten()}")
            
            # Save intrinsic calibration results to main results directory
            intrinsic_output_dir = "data/results/intrinsic_calibration_standalone"
            os.makedirs(intrinsic_output_dir, exist_ok=True)
            intrinsic_calibrator.save_calibration(
                os.path.join(intrinsic_output_dir, 'intrinsic_calibration_results.json'),
                include_extrinsics=True
            )
            print(f"   Standalone intrinsic calibration results saved to: {intrinsic_output_dir}")
            
            # Export test images to eye_to_hand_calibration directory for testing
            print(f"📸 Exporting intrinsic calibration test images...")
            eye_to_hand_dir = "data/results/eye_to_hand_calibration"
            os.makedirs(eye_to_hand_dir, exist_ok=True)
            
            try:
                # Generate and export pattern detection images
                if hasattr(intrinsic_calibrator, 'draw_pattern_on_images'):
                    pattern_debug_dir = os.path.join(eye_to_hand_dir, "intrinsic_pattern_detection")
                    os.makedirs(pattern_debug_dir, exist_ok=True)
                    pattern_images = intrinsic_calibrator.draw_pattern_on_images()
                    
                    for filename, debug_img in pattern_images:
                        output_path = os.path.join(pattern_debug_dir, f"intrinsic_{filename}.jpg")
                        cv2.imwrite(output_path, debug_img)
                    
                    print(f"   Pattern detection images: {len(pattern_images)} images in {pattern_debug_dir}")
                
                # Generate and export undistorted images with axes
                if hasattr(intrinsic_calibrator, 'draw_axes_on_undistorted_images'):
                    axes_debug_dir = os.path.join(eye_to_hand_dir, "intrinsic_undistorted_axes")
                    os.makedirs(axes_debug_dir, exist_ok=True)
                    axes_images = intrinsic_calibrator.draw_axes_on_undistorted_images()
                    
                    for filename, debug_img in axes_images:
                        output_path = os.path.join(axes_debug_dir, f"intrinsic_{filename}.jpg")
                        cv2.imwrite(output_path, debug_img)
                    
                    print(f"   Undistorted axes images: {len(axes_images)} images in {axes_debug_dir}")
                
                # Generate and export reprojection analysis images
                if hasattr(intrinsic_calibrator, 'draw_reprojection_on_images'):
                    reprojection_debug_dir = os.path.join(eye_to_hand_dir, "intrinsic_reprojection_analysis")
                    os.makedirs(reprojection_debug_dir, exist_ok=True)
                    reprojection_images = intrinsic_calibrator.draw_reprojection_on_images()
                    
                    for filename, debug_img in reprojection_images:
                        output_path = os.path.join(reprojection_debug_dir, f"intrinsic_{filename}.jpg")
                        cv2.imwrite(output_path, debug_img)
                    
                    print(f"   Reprojection analysis images: {len(reprojection_images)} images in {reprojection_debug_dir}")
                
                print(f"✅ Intrinsic calibration test images exported to: {eye_to_hand_dir}")
                
            except Exception as e:
                print(f"   ⚠️ Warning: Test image generation failed: {e}")
            
            return camera_matrix, distortion_coeffs
        else:
            print("❌ Intrinsic calibration failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during standalone intrinsic calibration: {e}")
        return None, None


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
        
        pattern, pattern_type = load_pattern_config(config_path)
        
        # Create IntrinsicCalibrator with smart constructor
        intrinsic_calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type=pattern_type
        )
        
        # Detect pattern points
        if not intrinsic_calibrator.detect_pattern_points(verbose=True):
            print("❌ Failed to detect calibration patterns for intrinsic calibration")
            return None, None
        
        # Perform intrinsic calibration
        print("   Running intrinsic calibration...")
        rms_error = intrinsic_calibrator.calibrate_camera(verbose=True)
        
        if rms_error > 0:
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coeffs = intrinsic_calibrator.get_distortion_coefficients()
            
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
            print(f"   Intrinsic calibration results saved to: {intrinsic_output_dir}")
            
            return camera_matrix, distortion_coeffs
        else:
            print("❌ Intrinsic calibration failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during intrinsic calibration: {e}")
        return None, None


def test_eye_to_hand_calibration():
    """Test eye-in-hand calibration workflow using sample data."""
    print("🤖 Eye-in-Hand Camera Calibration")
    print("=" * 50)
    
    # Check sample data directory
    sample_dir = os.path.join("sample_data", "eye_to_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        print("Please ensure the eye_to_hand_test_data directory exists with:")
        print("  - Calibration images (0.jpg, 1.jpg, ...)")
        print("  - Robot pose files (0.json, 1.json, ...)")
        return
    
    print(f"📁 Using sample data from: {sample_dir}")
    
    # Calculate camera intrinsic parameters using dedicated calibration data
    # This uses larger patterns more suitable for accurate intrinsic calibration
    camera_matrix, distortion_coefficients = calculate_camera_intrinsics_standalone()
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
        
    pattern, pattern_type = load_pattern_config(config_path)
    
    # Smart constructor approach - initialize with all data at once
    # CHANGED: Use EyeToHandCalibrator instead of EyeInHandCalibrator
    calibrator = EyeToHandCalibrator(
        camera_matrix=camera_matrix,          # Camera intrinsics
        distortion_coefficients=distortion_coefficients,
        calibration_pattern=pattern,         # Calibration pattern (loaded from JSON)
        pattern_type=pattern_type            # Pattern type string (loaded from JSON)
    )
    
    print("✅ Calibrator initialized with smart constructor")
    print(f"   Camera intrinsics loaded: {calibrator.camera_matrix is not None}")
    print(f"   Calibration pattern set: {calibrator.calibration_pattern is not None}")
    
    # Load calibration data (images and robot poses)
    if not calibrator.load_calibration_data(sample_dir):
        print("❌ Failed to load calibration data")
        return
        
    print(f"✅ Calibration data loaded:")
    print(f"   Images: {len(calibrator.images) if calibrator.images else 0}")
    print(f"   Robot poses: {len(calibrator.robot_poses) if calibrator.robot_poses else 0}")
    print(f"   Image size: {calibrator.image_size}")
    
    # Detect calibration pattern points in images
    print(f"\n🎯 Detecting calibration patterns...")
    if not calibrator.detect_pattern_points():
        print("❌ Failed to detect calibration patterns")
        return
        
    print(f"✅ Pattern detection completed:")
    print(f"   Detected patterns in {len(calibrator.image_points)} images")
    print(f"   Object points generated: {len(calibrator.object_points)}")
    
    # Perform hand-eye calibration with different methods
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
                calibrator.base2cam_matrix = None
                calibrator.calibration_completed = False
                calibrator.rms_error = None
                calibrator.per_image_errors = None
                
                # Perform calibration with this method
                rms_error = calibrator.calibrate(method=method_const, verbose=False)
                
                if rms_error > 0:
                    print(f"   ✅ Success: RMS error = {rms_error:.4f} pixels")
                    
                    # Store results
                    results.append({
                        'name': method_name,
                        'method': method_const,
                        'rms_error': rms_error,
                        'transformation': calibrator.base2cam_matrix.copy(),
                        'per_image_errors': calibrator.per_image_errors.copy() if calibrator.per_image_errors else None
                    })
                    
                    # Track best method
                    if rms_error < best_error:
                        best_error = rms_error
                        best_method = method_name
                        
                else:
                    print(f"   ❌ Failed: Calibration returned 0 error")
                    
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
            calibrator.cam2end_matrix = best_result['transformation']
            calibrator.rms_error = best_result['rms_error']
            calibrator.per_image_errors = best_result['per_image_errors']
            calibrator.calibration_completed = True
            
            print(f"\n🎯 Using best method: {best_result['name']}")
            print(f"   RMS reprojection error: {best_result['rms_error']:.4f} pixels")
            print(f"   Camera to end-effector transformation matrix:")
            for i, row in enumerate(best_result['transformation']):
                print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
                
            # Show per-image error statistics
            if best_result['per_image_errors']:
                errors = [e for e in best_result['per_image_errors'] if not np.isinf(e)]
                if errors:
                    print(f"   Error statistics: min={min(errors):.3f}, max={max(errors):.3f}, std={np.std(errors):.3f}")
                    errors_summary = [f'{e:.3f}' for e in errors[:5]]
                    print(f"   Per-image errors (first 5): {errors_summary}")
                    
        else:
            print(f"\n❌ All calibration methods failed!")
            return False
        
        # Legacy methods for additional error analysis may have implementation issues
        print(f"\n📊 Additional calibration analysis completed using built-in error calculation")
        
        # Set up output directory for results and debug images
        output_dir = "data/results/eye_to_hand_calibration"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate debug images
        print(f"\n🔍 Generating debug images...")
        
        try:
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
            
            # Draw pattern point reprojections on original images
            reprojection_debug_dir = os.path.join(output_dir, "reprojection_analysis")
            os.makedirs(reprojection_debug_dir, exist_ok=True)
            reprojection_images = calibrator.draw_reprojection_on_images()
            
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"   Reprojection analysis images: {len(reprojection_images)} images in {reprojection_debug_dir}")
            
        except Exception as e:
            print(f"   ⚠️ Warning: Debug image generation failed: {e}")
        
        # Save calibration results
        
        calibrator.save_results(output_dir)
        print(f"\n💾 Calibration results saved to: {output_dir}")
        
        # Generate reprojection images
        print(f"\n📸 Generating reprojection images...")
        reprojection_dir = os.path.join(output_dir, "reprojection_analysis")
        os.makedirs(reprojection_dir, exist_ok=True)
        
        try:
            reprojection_images = calibrator.draw_reprojection_on_images()
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(reprojection_dir, f"{filename}_reprojection.jpg")
                cv2.imwrite(output_path, debug_img)
            print(f"   Reprojection images: {len(reprojection_images)} images in {reprojection_dir}")
        except Exception as e:
            print(f"   ⚠️ Warning: Reprojection images failed: {e}")
        
        # Display final results summary
        print(f"\n✨ Eye-to-Hand Calibration Complete!")
        print(f"   Final calibration matrix (base to camera):")
        for i, row in enumerate(calibrator.base2cam_matrix):
            print(f"   [{' '.join([f'{val:8.4f}' for val in row])}]")
        print(f"   Final RMS error: {calibrator.rms_error:.4f} pixels")
        
        return True
        
    except Exception as e:
        print(f"❌ Hand-eye calibration failed: {e}")
        return False


def inspect_sample_data():
    """Inspect the sample data to understand its format."""
    print("\n🔍 Inspecting Sample Data")
    print("=" * 30)
    
    sample_dir = os.path.join("sample_data", "eye_to_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        return
    
    # Count files
    image_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
    json_files = [f for f in os.listdir(sample_dir) if f.endswith('.json')]
    
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


if __name__ == "__main__":
    print("🤖 Hand-in-Eye Camera Calibration Example")
    print("=" * 70)
    print("Demonstrates eye-in-hand calibration for robot-mounted cameras")
    print()
    
    # Inspect sample data first
    inspect_sample_data()
    
    # Run calibration with integrated optimization
    success = test_eye_to_hand_calibration()
    
    print(f"\n✨ Eye-to-Hand calibration example completed!")
    print(f"   Results saved to: data/results/eye_to_hand_calibration/")
    if success:
        print(f"   ✅ Calibration successful")
        print(f"   📸 Debug images available for analysis")
    else:
        print(f"   ❌ Calibration failed")
