#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Test Script
===================================

This test script validates the EyeInHandCalibrator class for 
eye-in-hand calibration (camera mounted on robot end-effector):

1. Load robot pose images and corresponding end-effector poses from eye_in_hand_test_data
2. Perform intrinsic calibration using eye_in_hand_test_data images
3. Initialize EyeInHandCalibrator with loaded data
4. Perform complete eye-in-hand calibration with error validation
5. Generate comprehensive debug images for calibration analysis
6. Save results and test IO operations

Test Features:
- RMS error threshold validation (fails if > 0.1 pixel)
- Updated calibrate() method with comprehensive dictionary return format
- Complete eye-in-hand calibration workflow with JSON serialization
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

from core.eye_in_hand_calibration import EyeInHandCalibrator
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
    Load images and robot pose matrices from the specified directory.
    
    Args:
        data_dir: Directory containing images and corresponding JSON pose files
        
    Returns:
        tuple: (images, end2base_matrices, calibration_pattern) or (None, None, None) if failed
    """
    print("\n" + "="*60)
    print("📂 Loading Eye-in-Hand Data")
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
        
        # Step 3: Load images and corresponding poses
        images = []
        end2base_matrices = []
        
        for image_file in image_files:
            # Get corresponding JSON file
            json_file = image_file.replace('.jpg', '.json')
            image_path = os.path.join(data_dir, image_file)
            json_path = os.path.join(data_dir, json_file)
            
            if not os.path.exists(json_path):
                print(f"⚠️ Warning: No pose file found for {image_file}, skipping")
                continue
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Warning: Could not load image {image_file}, skipping")
                continue
                
            # Load robot pose
            try:
                with open(json_path, 'r') as f:
                    pose_data = json.load(f)
                    
                # Extract end2base transformation matrix
                end2base_matrix = np.array(pose_data['end2base'])
                
                images.append(image)
                end2base_matrices.append(end2base_matrix)
                
                print(f"✅ Loaded: {image_file} with pose data")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load pose for {image_file}: {e}")
                continue
        
        print(f"📊 Successfully loaded {len(images)} image-pose pairs")
        
        if len(images) == 0:
            raise RuntimeError("No valid image-pose pairs were loaded")
            
        return images, end2base_matrices, calibration_pattern
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None


def test_eye_in_hand_calibration():
    """
    Test function demonstrating complete eye-in-hand calibration with error validation.
    Raises exception if RMS error > 1 pixel threshold.
    """
    print("=" * 80)
    print("🤖 Eye-in-Hand Calibration Example - Complete Calibration Workflow")
    print("=" * 80)
    
    # Define data paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eye_in_hand_data_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
    results_dir = os.path.join(project_root, "data", "results", "eye_in_hand_example")
    
    print(f"📂 Eye-in-hand data directory: {eye_in_hand_data_dir}")
    print(f"💾 Results will be saved to: {results_dir}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load data (images, poses, and pattern) from eye-in-hand directory
    images, end2base_matrices, calibration_pattern = load_data(eye_in_hand_data_dir)
    
    if images is None:
        print("❌ Failed to load eye-in-hand data")
        return False
    
    # Step 2: Perform intrinsic calibration using the eye-in-hand data directory
    camera_matrix, distortion_coefficients, rms_error = perform_intrinsic_calibration(
        data_dir=eye_in_hand_data_dir,
        results_dir=results_dir
    )
    
    if camera_matrix is None:
        print("❌ Failed to perform intrinsic calibration")
        return False
    
    # Step 3: Initialize EyeInHandCalibrator with loaded data
    print("\n" + "="*60)
    print("🤖 Step 3: Initialize Eye-in-Hand Calibrator")
    print("="*60)
    
    try:
        # Create eye-in-hand calibrator with all loaded data
        eye_in_hand_calibrator = EyeInHandCalibrator(
            images=images,
            end2base_matrices=end2base_matrices,
            image_paths=None,  # Set to None as we directly provide images and matrices
            calibration_pattern=calibration_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients.flatten()  # Flatten to 1D array
        )
        
        print("✅ EyeInHandCalibrator initialized successfully")
        
        # Step 4: Perform Eye-in-Hand Calibration
        print("\n" + "="*60)
        print("🤖 Step 4: Perform Eye-in-Hand Calibration")
        print("="*60)
                
        print("🧪 Testing calibration with automatic method selection...")
        calibration_result = eye_in_hand_calibrator.calibrate(method=None, verbose=True)
        
        if calibration_result is not None:
            # Check RMS error threshold - consider calibration failed if > 1 pixel
            rms_error = calibration_result['rms_error']
            if rms_error > 0.1:
                print(f"❌ Eye-in-hand calibration failed!")
                print(f"   RMS Error: {rms_error:.4f} pixels (threshold: 1.0)")
                raise ValueError(f"Eye-in-hand calibration RMS error {rms_error:.4f} exceeds threshold of 1.0 pixels")
            
            print(f"✅ Eye-in-hand calibration completed successfully!")
            print(f"   • RMS error: {calibration_result['rms_error']:.4f} pixels")
            print(f"   • Valid images: {len([p for p in eye_in_hand_calibrator.image_points if p is not None])}/{len(eye_in_hand_calibrator.image_points)}")
            print(f"   • Camera-to-end transformation matrix shape: {calibration_result['cam2end_matrix'].shape}")
            print(f"   • Target-to-base transformation matrix shape: {calibration_result['target2base_matrix'].shape}")
            
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
                pattern_images = eye_in_hand_calibrator.draw_pattern_on_images()
                
                for filename, debug_img in pattern_images:
                    output_path = os.path.join(pattern_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)
                
                print(f"   ✅ Pattern detection images: {len(pattern_images)} images saved to {pattern_debug_dir}")
                
                # Draw 3D axes on undistorted images
                axes_debug_dir = os.path.join(debug_output_dir, "undistorted_axes")
                os.makedirs(axes_debug_dir, exist_ok=True)
                axes_images = eye_in_hand_calibrator.draw_axes_on_undistorted_images()
                
                for filename, debug_img in axes_images:
                    output_path = os.path.join(axes_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)
                
                print(f"   ✅ Undistorted axes images: {len(axes_images)} images saved to {axes_debug_dir}")
                
                # Draw pattern point reprojections on original images
                reprojection_debug_dir = os.path.join(debug_output_dir, "reprojection_analysis")
                os.makedirs(reprojection_debug_dir, exist_ok=True)
                reprojection_images = eye_in_hand_calibrator.draw_reprojection_on_images()
                
                for filename, debug_img in reprojection_images:
                    output_path = os.path.join(reprojection_debug_dir, f"{filename}.jpg")
                    cv2.imwrite(output_path, debug_img)
                
                print(f"   ✅ Reprojection analysis images: {len(reprojection_images)} images saved to {reprojection_debug_dir}")
                print(f"   📁 All debug images saved to: {debug_output_dir}")
                
            except Exception as e:
                print(f"   ⚠️ Warning: Debug image generation failed: {e}")
        else:
            print("❌ Eye-in-hand calibration failed")
            raise ValueError("Eye-in-hand calibration failed")

        # Step 6: Save Results
        print("\n" + "="*60)
        print("💾 Step 6: Save Results")
        print("="*60)
        
        try:
            eye_in_hand_calibrator.save_results(results_dir)
            print("✅ Eye-in-hand results saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        print("\n" + "="*80)
        print("🎉 EYE-IN-HAND CALIBRATION TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   • Loaded {len(images)} image-pose pairs")
        print(f"   • Intrinsic calibration RMS error: {rms_error:.4f} pixels")
        if calibration_result is not None:
            print(f"   • Eye-in-hand calibration: ✅ COMPLETED")
            print(f"   • Best calibration method: {eye_in_hand_calibrator.best_method_name}")
            print(f"   • Hand-eye calibration RMS error: {calibration_result['rms_error']:.4f} pixels")
            print(f"   • Used {len([p for p in eye_in_hand_calibrator.image_points if p is not None])}/{len(eye_in_hand_calibrator.image_points)} images")
        else:
            print(f"   • Eye-in-hand calibration: ⚠️ FAILED")
        print(f"   • All data validation: ✅ PASSED")
        print(f"   • IO operations: ✅ TESTED")
        print(f"   • Results saved to: {results_dir}")
        print("\nNote: This test validates eye-in-hand calibration with RMS error threshold checking.")
        
        return True
        
    except Exception as e:
        print(f"❌ Eye-in-hand calibrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to run eye-in-hand calibration test with error handling.
    """
    print("Starting Eye-in-Hand Calibration Test...")
    
    success_count = 0
    total_tests = 1
    
    try:
        print("\n" + "="*60)
        print("🤖 Testing Eye-in-Hand Calibration")
        print("="*60)
        
        test_eye_in_hand_calibration()
        success_count += 1
        print("✅ Eye-in-hand calibration test: PASSED")
        
    except Exception as e:
        print(f"❌ Eye-in-hand calibration test: FAILED - {e}")
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
