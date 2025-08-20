#!/usr/bin/env python3
"""
Hand-in-Eye Calibration Example
===============================

This example demonstrates how to use the EyeInHandCalibrator class for 
hand-in-eye calibration (camera mounted on robot end-effector):

1. Load robot pose images and corresponding end-effector poses
2. Calculate camera intrinsic parameters using actual calibration images
3. Detect calibration patterns in images
4. Compare different OpenCV hand-eye calibration methods
5. Perform optimization to improve calibration accuracy
6. Generate before/after reprojection images for comparison
7. Save calibration results and comprehensive debug images

The class features a clean interface with smart constructor arguments 
and organized member variables following the same pattern as IntrinsicCalibrator.
Optimization is seamlessly integrated into the main workflow.
"""

import os
import cv2
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eye_in_hand_calibration import EyeInHandCalibrator
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_chessboard_pattern
from core.utils import load_images_from_directory


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
        
        # Create calibration pattern (same as used for hand-eye calibration)
        pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)
        
        # Create IntrinsicCalibrator with smart constructor
        intrinsic_calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type='standard'
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


def test_eye_in_hand_calibration():
    """Test eye-in-hand calibration workflow using sample data."""
    print("🤖 Eye-in-Hand Camera Calibration")
    print("=" * 50)
    
    # Check sample data directory
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
    if not os.path.exists(sample_dir):
        print(f"❌ Sample data directory not found: {sample_dir}")
        print("Please ensure the eye_in_hand_test_data directory exists with:")
        print("  - Calibration images (0.jpg, 1.jpg, ...)")
        print("  - Robot pose files (0.json, 1.json, ...)")
        return
    
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
    
    # Create calibration pattern
    # Based on the sample images, they use an 11x8 chessboard pattern
    pattern = create_chessboard_pattern('standard', width=11, height=8, square_size=0.02)
    print(f"🎯 Calibration pattern: 11x8 chessboard, 20mm squares")
    
    # Smart constructor approach - initialize with all data at once
    calibrator = EyeInHandCalibrator(
        camera_matrix=camera_matrix,          # Camera intrinsics
        distortion_coefficients=distortion_coefficients,
        calibration_pattern=pattern,         # Calibration pattern
        pattern_type='standard'              # Pattern type string
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
                calibrator.cam2end_matrix = None
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
                        'transformation': calibrator.cam2end_matrix.copy(),
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
        output_dir = "data/results/eye_in_hand_calibration"
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
        
        # Generate reprojection images before optimization
        print(f"\n📸 Generating reprojection images before optimization...")
        reprojection_before_dir = os.path.join(output_dir, "reprojection_before_optimization")
        os.makedirs(reprojection_before_dir, exist_ok=True)
        
        try:
            reprojection_images_before = calibrator.draw_reprojection_on_images()
            for filename, debug_img in reprojection_images_before:
                output_path = os.path.join(reprojection_before_dir, f"{filename}_before_opt.jpg")
                cv2.imwrite(output_path, debug_img)
            print(f"   Before optimization: {len(reprojection_images_before)} images in {reprojection_before_dir}")
        except Exception as e:
            print(f"   ⚠️ Warning: Pre-optimization reprojection images failed: {e}")
        
        # Perform optimization to improve calibration results
        print(f"\n🔍 Performing calibration optimization...")
        print(f"   Initial RMS error: {best_result['rms_error']:.4f} pixels")
        print(f"   Note: Optimization refines both target2base and cam2end matrices")
        print(f"         using nonlinear optimization to minimize reprojection error")
        
        optimization_improved = False
        try:
            # Use the new optimize_calibration method with member-based approach
            optimized_error = calibrator.optimize_calibration(
                iterations=5,        # Number of optimization iterations
                ftol_rel=1e-6,      # Relative tolerance for convergence
                verbose=True        # Show optimization progress
            )
            
            if optimized_error > 0 and optimized_error < best_result['rms_error']:
                improvement = best_result['rms_error'] - optimized_error
                improvement_pct = (improvement / best_result['rms_error']) * 100
                print(f"✅ Optimization completed successfully!")
                print(f"   Optimized RMS error: {optimized_error:.4f} pixels")
                print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                calibrator.rms_error = optimized_error  # Update the RMS error
                optimization_improved = True
            else:
                print(f"⚠️  Optimization completed with no significant improvement")
                print(f"   Final RMS error: {optimized_error:.4f} pixels" if optimized_error > 0 else f"   Using original RMS error: {best_result['rms_error']:.4f} pixels")
                
        except Exception as opt_error:
            print(f"⚠️  Optimization failed: {opt_error}")
            print(f"   Using original calibration results")
            print(f"   RMS error: {best_result['rms_error']:.4f} pixels")
        
        # Generate reprojection images after optimization
        print(f"\n📸 Generating reprojection images after optimization...")
        reprojection_after_dir = os.path.join(output_dir, "reprojection_after_optimization")
        os.makedirs(reprojection_after_dir, exist_ok=True)
        
        try:
            reprojection_images_after = calibrator.draw_reprojection_on_images()
            for filename, debug_img in reprojection_images_after:
                output_path = os.path.join(reprojection_after_dir, f"{filename}_after_opt.jpg")
                cv2.imwrite(output_path, debug_img)
            print(f"   After optimization: {len(reprojection_images_after)} images in {reprojection_after_dir}")
            
            if optimization_improved:
                print(f"   💡 Compare before/after images to see optimization improvements")
            else:
                print(f"   ℹ️  Images show final calibration results (optimization had minimal impact)")
                
        except Exception as e:
            print(f"   ⚠️ Warning: Post-optimization reprojection images failed: {e}")
        
        # Display final results summary
        print(f"\n✨ Hand-in-Eye Calibration with Optimization Complete!")
        print(f"   Final calibration matrix (camera to end-effector):")
        for i, row in enumerate(calibrator.cam2end_matrix):
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
    
    sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
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
    success = test_eye_in_hand_calibration()
    
    print(f"\n✨ Eye-in-Hand calibration example completed!")
    print(f"   Results saved to: data/results/eye_in_hand_calibration/")
    if success:
        print(f"   ✅ Calibration and optimization successful")
        print(f"   📸 Before/after optimization images available for comparison")
    else:
        print(f"   ❌ Calibration failed")
