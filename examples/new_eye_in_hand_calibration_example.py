#!/usr/bin/env python3
"""
New Eye-in-Hand Calibration Example
===================================

This example demonstrates how to use the NewEyeInHandCalibrator class for 
eye-in-hand calibration (camera mounted on robot end-effector):

1. Load robot pose images and corresponding end-effector poses from eye_in_hand_test_data
2. Calculate camera intrinsic parameters using eye_in_hand_test_data images
3. Load and validate all calibration data
4. Test the new IO-only architecture

Note: Calibration algorithms have been removed from NewEyeInHandCalibrator.
This example focuses on data loading and intrinsic calibration only.
Future calibration functionality will be added via dedicated calibration modules.
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


def main():
    """
    Main function demonstrating new eye-in-hand calibration data loading and intrinsic calibration.
    """
    print("=" * 80)
    print("🤖 New Eye-in-Hand Calibration Example - Data Loading & Intrinsic Calibration")
    print("=" * 80)
    
    # Define data paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eye_in_hand_data_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
    results_dir = os.path.join(project_root, "data", "results", "new_eye_in_hand_example")
    
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
    
    # Step 3: Initialize NewEyeInHandCalibrator with loaded data
    print("\n" + "="*60)
    print("🤖 Step 3: Initialize New Eye-in-Hand Calibrator")
    print("="*60)
    
    try:
        # Create eye-in-hand calibrator with all loaded data
        eye_in_hand_calibrator = NewEyeInHandCalibrator(
            images=images,
            end2base_matrices=end2base_matrices,
            image_paths=None,  # Set to None as we directly provide images and matrices
            calibration_pattern=calibration_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients.flatten()  # Flatten to 1D array
        )
        
        print("✅ NewEyeInHandCalibrator initialized successfully")
        
        # Step 4: Test data validation
        print("\n" + "="*60)
        print("✅ Step 4: Validate Eye-in-Hand Data")
        print("="*60)
        
        is_valid = eye_in_hand_calibrator.validate_eye_in_hand_data()
        
        if is_valid:
            print("✅ All eye-in-hand calibration data is valid!")
        else:
            print("❌ Eye-in-hand calibration data validation failed")
            return False
        
        # Step 5: Display calibration information
        print("\n" + "="*60)
        print("📊 Step 5: Eye-in-Hand Calibration Information")
        print("="*60)
        
        calib_info = eye_in_hand_calibrator.get_calibration_info()
        print(f"📋 Calibration Info:")
        print(f"   • Images loaded: {calib_info['image_count']}")
        print(f"   • Robot poses: {calib_info['transform_count']}")
        print(f"   • Pattern: {calib_info['pattern_type']}")
        print(f"   • Has intrinsics: {calib_info['has_intrinsics']}")
        print(f"   • Has extrinsics: {calib_info['has_extrinsics']}")
        print(f"   • Calibration completed: {calib_info['calibration_completed']}")
        
        # Step 6: Test IO methods
        print("\n" + "="*60)
        print("💾 Step 6: Test IO Methods")
        print("="*60)
        
        # Test cam2end matrix operations (should be None initially)
        print(f"📄 Initial cam2end_matrix: {eye_in_hand_calibrator.get_cam2end_matrix()}")
        
        # Test setting a dummy cam2end matrix
        dummy_cam2end = np.eye(4)
        eye_in_hand_calibrator.set_cam2end_matrix(dummy_cam2end)
        print(f"✅ Set dummy cam2end_matrix")
        print(f"📄 Retrieved cam2end_matrix shape: {eye_in_hand_calibrator.get_cam2end_matrix().shape}")
        
        # Test getting calibration results
        results = eye_in_hand_calibrator.get_calibration_results()
        print(f"📊 Calibration results keys: {list(results.keys())}")
        
        # Step 7: Save results
        print("\n" + "="*60)
        print("💾 Step 7: Save Results")
        print("="*60)
        
        try:
            eye_in_hand_calibrator.save_eye_in_hand_results(results_dir)
            print("✅ Eye-in-hand results saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        print("\n" + "="*80)
        print("🎉 NEW EYE-IN-HAND CALIBRATION EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   • Loaded {len(images)} image-pose pairs")
        print(f"   • Intrinsic calibration RMS error: {rms_error:.4f} pixels")
        print(f"   • All data validation: ✅ PASSED")
        print(f"   • IO operations: ✅ TESTED")
        print(f"   • Results saved to: {results_dir}")
        print("\nNote: Calibration algorithms have been moved to dedicated modules.")
        print("This example demonstrates the new IO-only architecture.")
        
        return True
        
    except Exception as e:
        print(f"❌ Eye-in-hand calibrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
