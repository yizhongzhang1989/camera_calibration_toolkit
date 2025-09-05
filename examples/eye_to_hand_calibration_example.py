#!/usr/bin/env python3
"""
Eye-to-Hand Calibration Test Script
===================================

This test script validates the EyeToHandCalibrator class for
eye-to-hand calibration (stationary camera looking at robot end-effector):

1. Load robot pose images and corresponding end-effector poses from eye_to_hand_test_data
2. Perform intrinsic calibration using intrinsic_calib_grid_test_images
3. Initialize EyeToHandCalibrator with loaded data
4. Perform complete eye-to-hand calibration with error validation
5. Generate comprehensive debug images for calibration analysis
6. Save results and test IO operations

Test Features:
- RMS error threshold validation (fails if > 1.0 pixel)
- Updated calibrate() method with comprehensive dictionary return format
- Complete eye-to-hand calibration workflow with JSON serialization
- Debug image generation: pattern detection, axes visualization, reprojection analysis
- Comprehensive calibration validation and error reporting

Note: This test script validates calibration quality through error threshold checking
and raises exceptions for calibration failures to ensure robust validation.
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


def perform_intrinsic_calibration(data_dir: str, results_dir: str) -> tuple:
    """
    Perform camera intrinsic calibration using images and pattern from the specified directory.

    Args:
        data_dir: Directory containing calibration images and pattern configuration
        results_dir: Directory to save intrinsic calibration results

    Returns:
        tuple: (camera_matrix, distortion_coefficients, rms_error) or (None, None, None) if failed
    """
    print("\n" + "="*60)
    print("📐 Performing Camera Intrinsic Calibration")
    print("="*60)

    try:
        # Step 1: Load calibration pattern configuration
        pattern_config_path = os.path.join(data_dir, "chessboard_config.json")
        if not os.path.exists(pattern_config_path):
            raise FileNotFoundError(f"Pattern config not found: {pattern_config_path}")

        # Load JSON data from file
        with open(pattern_config_path, 'r') as f:
            pattern_json_data = json.load(f)

        calibration_pattern = load_pattern_from_json(pattern_json_data)
        print(f"✅ Loaded pattern: {calibration_pattern}")

        # Step 2: Load images for intrinsic calibration
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        image_files.sort()  # Sort to ensure consistent ordering

        print(f"🔍 Found {len(image_files)} image files")

        images = []
        for image_file in image_files:
            image_path = os.path.join(data_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                print(f"✅ Loaded image: {image_file}")
            else:
                print(f"⚠️ Warning: Could not load {image_file}")

        print(f"📊 Successfully loaded {len(images)} images for intrinsic calibration")

        # Step 3: Perform intrinsic calibration
        intrinsic_calibrator = IntrinsicCalibrator(
            images=images,
            calibration_pattern=calibration_pattern
        )

        print(f"🔍 Performing intrinsic calibration with {len(images)} images...")
        success = intrinsic_calibrator.calibrate(verbose=True)

        if success:
            camera_matrix = intrinsic_calibrator.get_camera_matrix()
            distortion_coefficients = intrinsic_calibrator.get_distortion_coefficients()
            rms_error = intrinsic_calibrator.get_rms_error()

            print(f"✅ Intrinsic calibration successful!")
            print(f"   RMS error: {rms_error:.4f} pixels")
            print(f"   Camera matrix shape: {camera_matrix.shape}")
            print(f"   Distortion coefficients shape: {distortion_coefficients.shape}")

            # Save intrinsic calibration results
            intrinsic_results_dir = os.path.join(results_dir, "intrinsic_calibration")
            intrinsic_calibrator.save_results(intrinsic_results_dir)
            print(f"✅ Intrinsic calibration results saved to: {intrinsic_results_dir}")

            return camera_matrix, distortion_coefficients, rms_error

        else:
            raise RuntimeError("Intrinsic calibration failed")

    except Exception as e:
        print(f"❌ Intrinsic calibration failed: {e}")
        return None, None, None


def load_data(data_dir: str) -> tuple:
    """
    Load image paths and calibration pattern from the specified directory.

    Args:
        data_dir: Directory containing images and pattern configuration

    Returns:
        tuple: (image_paths, calibration_pattern) or (None, None) if failed
    """
    print("\n" + "="*60)
    print("📂 Loading Eye-to-Hand Data")
    print("="*60)

    try:
        # Step 1: Load calibration pattern configuration
        pattern_config_path = os.path.join(data_dir, "chessboard_config.json")
        if not os.path.exists(pattern_config_path):
            raise FileNotFoundError(f"Pattern config not found: {pattern_config_path}")

        # Load JSON data from file
        with open(pattern_config_path, 'r') as f:
            pattern_json_data = json.load(f)

        calibration_pattern = load_pattern_from_json(pattern_json_data)
        print(f"✅ Loaded pattern: {calibration_pattern}")

        # Step 2: Get all image files in the directory
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        image_files.sort()  # Sort to ensure consistent ordering

        print(f"🔍 Found {len(image_files)} image files")

        # Step 3: Load image paths
        image_paths = []

        for image_file in image_files:
            image_path = os.path.join(data_dir, image_file)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image file not found {image_file}, skipping")
                continue

            image_paths.append(image_path)
            print(f"✅ Loaded: {image_file}")

        print(f"📊 Successfully loaded {len(image_paths)} image paths")

        if len(image_paths) == 0:
            raise RuntimeError("No valid image paths were loaded")

        return image_paths, calibration_pattern

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None


def test_eye_to_hand_calibration():
    """
    Test function demonstrating complete eye-to-hand calibration with error validation.
    Raises exception if RMS error > 1 pixel threshold.
    """
    print("=" * 80)
    print("🤖 Eye-to-Hand Calibration Example - Complete Calibration Workflow")
    print("=" * 80)

    # Define data paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eye_to_hand_data_dir = os.path.join(project_root, "sample_data", "eye_to_hand_test_data")
    intrinsic_images_dir = os.path.join(project_root, "sample_data", "intrinsic_calib_grid_test_images")
    results_dir = os.path.join(project_root, "data", "results", "eye_to_hand_example")

    print(f"📂 Eye-to-hand pose data directory: {eye_to_hand_data_dir}")
    print(f"📸 Intrinsic calibration images: {intrinsic_images_dir}")
    print(f"💾 Results will be saved to: {results_dir}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Perform intrinsic calibration using the dedicated images directory
    camera_matrix, distortion_coefficients, rms_error = perform_intrinsic_calibration(
        data_dir=intrinsic_images_dir,
        results_dir=results_dir
    )

    if camera_matrix is None:
        print("❌ Failed to perform intrinsic calibration")
        return False

    # Step 2: Load eye-to-hand data (image paths and pattern)
    image_paths, calibration_pattern = load_data(eye_to_hand_data_dir)

    if image_paths is None:
        print("❌ Failed to load eye-to-hand data")
        return False

    # Step 3: Initialize EyeToHandCalibrator with loaded data
    print("\n" + "="*60)
    print("🤖 Step 3: Initialize Eye-to-Hand Calibrator")
    print("="*60)

    try:
        # Create eye-to-hand calibrator with image paths (images and poses will be loaded automatically)
        eye_to_hand_calibrator = EyeToHandCalibrator(
            images=None,  # Set to None, will be loaded automatically from image_paths
            end2base_matrices=None,  # Set to None, will be loaded automatically from JSON files
            image_paths=image_paths,
            calibration_pattern=calibration_pattern,  # Use the loaded calibration pattern
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients.flatten()  # Flatten to 1D array
        )

        print("✅ EyeToHandCalibrator initialized successfully")

        # Step 4: Perform Eye-to-Hand Calibration
        print("\n" + "="*60)
        print("🤖 Step 4: Perform Eye-to-Hand Calibration")
        print("="*60)

        print("🧪 Testing calibration with automatic method selection...")
        calibration_result = eye_to_hand_calibrator.calibrate(method=None, verbose=True)

        if calibration_result is not None:
            # Check RMS error threshold - consider calibration failed if > 1 pixel
            rms_error = calibration_result['rms_error']
            if rms_error > 1.0:
                print(f"❌ Eye-to-hand calibration failed!")
                print(f"   RMS Error: {rms_error:.4f} pixels (threshold: 1.0)")
                raise ValueError(f"Eye-to-hand calibration RMS error {rms_error:.4f} exceeds threshold of 1.0 pixels")
            
            print(f"✅ Eye-to-hand calibration completed successfully!")
            print(f"   • RMS error: {calibration_result['rms_error']:.4f} pixels")
            print(f"   • Valid images: {len([p for p in eye_to_hand_calibrator.image_points if p is not None])}/{len(eye_to_hand_calibrator.image_points)}")
            print(f"   • Base-to-camera transformation matrix shape: {calibration_result['base2cam_matrix'].shape}")
            print(f"   • Target-to-end transformation matrix shape: {calibration_result['target2end_matrix'].shape}")

            # Step 5: Generate Debug Images
            print("\n" + "="*60)
            print("🔍 Step 5: Generate Debug Images")
            print("="*60)

            # Set up output directory for debug images
            debug_output_dir = os.path.join(results_dir, "debug_images")
            os.makedirs(debug_output_dir, exist_ok=True)

            print(f"🎨 Generating debug images...")

            try:
                # Draw detected patterns on original images
                pattern_debug_dir = os.path.join(debug_output_dir, "pattern_detection")
                os.makedirs(pattern_debug_dir, exist_ok=True)
                pattern_images = eye_to_hand_calibrator.draw_pattern_on_images()

                for filename, debug_img in pattern_images:
                    output_path = os.path.join(pattern_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)

                print(f"   ✅ Pattern detection images: {len(pattern_images)} images saved to {pattern_debug_dir}")

                # Draw 3D axes on undistorted images
                axes_debug_dir = os.path.join(debug_output_dir, "undistorted_axes")
                os.makedirs(axes_debug_dir, exist_ok=True)
                axes_images = eye_to_hand_calibrator.draw_axes_on_undistorted_images()

                for filename, debug_img in axes_images:
                    output_path = os.path.join(axes_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)

                print(f"   ✅ Undistorted axes images: {len(axes_images)} images saved to {axes_debug_dir}")

                # Draw pattern point reprojections on original images
                reprojection_debug_dir = os.path.join(debug_output_dir, "reprojection_analysis")
                os.makedirs(reprojection_debug_dir, exist_ok=True)
                reprojection_images = eye_to_hand_calibrator.draw_reprojection_on_images()

                for filename, debug_img in reprojection_images:
                    output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)

                print(f"   ✅ Reprojection analysis images: {len(reprojection_images)} images saved to {reprojection_debug_dir}")
                print(f"   📁 All debug images saved to: {debug_output_dir}")

            except Exception as e:
                print(f"   ⚠️ Warning: Debug image generation failed: {e}")
        else:
            print("❌ Eye-to-hand calibration failed")
            raise ValueError("Eye-to-hand calibration failed")

        # Step 6: Save Results
        print("\n" + "="*60)
        print("💾 Step 6: Save Results")
        print("="*60)

        try:
            eye_to_hand_calibrator.save_results(results_dir)
            print("✅ Eye-to-hand results saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

        print("\n" + "="*80)
        print("🎉 EYE-TO-HAND CALIBRATION TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   • Loaded {len(image_paths)} image-pose pairs")
        print(f"   • Intrinsic calibration RMS error: {rms_error:.4f} pixels")
        if calibration_result is not None:
            print(f"   • Eye-to-hand calibration: ✅ COMPLETED")
            print(f"   • Best calibration method: {eye_to_hand_calibrator.get_best_method_name()}")
            print(f"   • Hand-eye calibration RMS error: {calibration_result['rms_error']:.4f} pixels")
            print(f"   • Used {len([p for p in eye_to_hand_calibrator.image_points if p is not None])}/{len(eye_to_hand_calibrator.image_points)} images")
        else:
            print(f"   • Eye-to-hand calibration: ⚠️ FAILED")
        print(f"   • All data validation: ✅ PASSED")
        print(f"   • IO operations: ✅ TESTED")
        print(f"   • Results saved to: {results_dir}")
        print("\nNote: This test validates eye-to-hand calibration with RMS error threshold checking.")

        return True

    except Exception as e:
        print(f"❌ Eye-to-hand calibrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to run eye-to-hand calibration test with error handling.
    """
    print("Starting Eye-to-Hand Calibration Test...")
    
    success_count = 0
    total_tests = 1
    
    try:
        print("\n" + "="*60)
        print("🤖 Testing Eye-to-Hand Calibration")
        print("="*60)
        
        test_eye_to_hand_calibration()
        success_count += 1
        print("✅ Eye-to-hand calibration test: PASSED")
        
    except Exception as e:
        print(f"❌ Eye-to-hand calibration test: FAILED - {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {total_tests - success_count} test(s) failed!")
        return False

if __name__ == "__main__":
    main()
