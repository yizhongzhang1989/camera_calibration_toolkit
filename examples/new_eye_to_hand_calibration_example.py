#!/usr/bin/env python3
"""
New Eye-to-Hand Calibration Example
====================================

This example demonstrates how to use the NewEyeToHandCalibrator class for 
eye-to-hand calibration (c    # Step 4: Initialize NewEyeToHandCalibrator with loaded data
    print("\n" + "="*60)        print("📊 Summary:")
        print(f"   • Performed intrinsic calibration")
        print(f"   • Loaded {len(image_paths)} image-pose pairs for eye-to-hand")
        print(f"   • Intrinsic calibration RMS error: {rms_error:.4f} pixels")
        print(f"   • All data validation: ✅ PASSED")
        print(f"   • IO operations: ✅ TESTED")
        print(f"   • Results saved to: {results_dir}")
        print("\nNote: Calibration algorithms have been moved to dedicated modules.")
        print("This example demonstrates the new IO-only architecture.")t("👁️ Step 4: Initialize New Eye-to-Hand Calibrator")
    print("="*60)
    
    try:
        # Create eye-to-hand calibrator with image paths (images will be loaded automatically)
        eye_to_hand_calibrator = NewEyeToHandCalibrator(
            images=None,  # Set to None, will be loaded automatically from image_paths
            end2base_matrices=end2base_matrices,
            image_paths=image_paths,
            calibration_pattern=eye_to_hand_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients.flatten()  # Flatten to 1D array
        )tatically, looking at robot):

1. Load robot pose images from eye_to_hand_test_data 
2. Calculate camera intrinsic parameters using intrinsic_calib_grid_test_images
3. Load and validate all calibration data
4. Test the new IO-only architecture

Note: Calibration algorithms have been removed from NewEyeToHandCalibrator.
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

from core.new_eye_to_hand_calibration import NewEyeToHandCalibrator
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


def main():
    """
    Main function demonstrating new eye-to-hand calibration data loading and intrinsic calibration.
    """
    print("=" * 80)
    print("👁️ New Eye-to-Hand Calibration Example - Data Loading & Intrinsic Calibration")
    print("=" * 80)
    
    # Define data paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eye_to_hand_data_dir = os.path.join(project_root, "sample_data", "eye_to_hand_test_data")
    intrinsic_images_dir = os.path.join(project_root, "sample_data", "intrinsic_calib_grid_test_images")
    results_dir = os.path.join(project_root, "data", "results", "new_eye_to_hand_example")
    
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
    
    # Step 2: Load eye-to-hand pattern configuration
    print("\n" + "="*60)
    print("📋 Step 2: Load Eye-to-Hand Pattern")
    print("="*60)
    
    eye_to_hand_pattern_config = os.path.join(eye_to_hand_data_dir, "chessboard_config.json")
    if not os.path.exists(eye_to_hand_pattern_config):
        raise FileNotFoundError(f"Eye-to-hand pattern config not found: {eye_to_hand_pattern_config}")
    
    # Load JSON data from file
    with open(eye_to_hand_pattern_config, 'r') as f:
        eye_to_hand_pattern_json_data = json.load(f)
    
    eye_to_hand_pattern = load_pattern_from_json(eye_to_hand_pattern_json_data)
    print(f"✅ Loaded eye-to-hand pattern: {eye_to_hand_pattern}")
    
    # Step 4: Load images and robot poses for eye-to-hand calibration
    print("\n" + "="*60)
    print("📸 Step 3: Load Eye-to-Hand Images and Robot Poses")
    print("="*60)
    
    # Get all image files in the eye-to-hand directory
    eye_to_hand_image_files = [f for f in os.listdir(eye_to_hand_data_dir) if f.endswith('.jpg')]
    eye_to_hand_image_files.sort()  # Sort to ensure consistent ordering
    
    print(f"🔍 Found {len(eye_to_hand_image_files)} eye-to-hand image files")
    
    # Load image paths (matrices will be loaded automatically from JSON files)
    image_paths = []
    
    for image_file in eye_to_hand_image_files:
        # Get corresponding JSON file
        json_file = image_file.replace('.jpg', '.json')
        image_path = os.path.join(eye_to_hand_data_dir, image_file)
        json_path = os.path.join(eye_to_hand_data_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"⚠️ Warning: No pose file found for {image_file}, skipping")
            continue
            
        # Check if image exists (but don't load it)
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Image file not found {image_file}, skipping")
            continue
            
        # Add image path (matrices will be loaded automatically from JSON)
        image_paths.append(image_path)
        print(f"✅ Added: {image_file} (matrices will be loaded automatically)")
    
    print(f"📊 Successfully prepared {len(image_paths)} image paths for automatic loading")
    
    # Step 4: Initialize NewEyeToHandCalibrator with loaded data
    print("\n" + "="*60)
    print("👁️ Step 4: Initialize New Eye-to-Hand Calibrator")
    print("="*60)
    
    try:
        # Create eye-to-hand calibrator with image paths (images and matrices loaded automatically)
        eye_to_hand_calibrator = NewEyeToHandCalibrator(
            images=None,  # Set to None, will be loaded automatically from image_paths
            end2base_matrices=None,  # Set to None, will be loaded automatically from JSON files
            image_paths=image_paths,
            calibration_pattern=eye_to_hand_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients.flatten()  # Flatten to 1D array
        )
        
        print("✅ NewEyeToHandCalibrator initialized successfully")
        
        # Step 5: Test data validation
        print("\n" + "="*60)
        print("✅ Step 5: Validate Eye-to-Hand Data")
        print("="*60)
        
        is_valid = eye_to_hand_calibrator.validate_eye_to_hand_data()
        
        if is_valid:
            print("✅ All eye-to-hand calibration data is valid!")
        else:
            print("❌ Eye-to-hand calibration data validation failed")
            return False
        
        # Step 5.5: Calculate target-to-camera matrices
        print("\n" + "="*60)
        print("🎯 Step 5.5: Calculate Target-to-Camera Matrices")
        print("="*60)
        
        try:
            # First detect calibration patterns in all images
            eye_to_hand_calibrator.detect_pattern_points(verbose=True)
            
            # Calculate target2cam matrices for all detected patterns
            eye_to_hand_calibrator._calculate_target2cam_matrices(verbose=True)
            
            # Count successful calculations
            successful_matrices = sum(1 for matrix in eye_to_hand_calibrator.target2cam_matrices if matrix is not None)
            print(f"✅ Successfully calculated {successful_matrices} target2cam matrices")
            
        except Exception as e:
            print(f"⚠️ Target2cam matrix calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 6: Display calibration information
        print("\n" + "="*60)
        print("📊 Step 6: Eye-to-Hand Calibration Information")
        print("="*60)
        
        calib_info = eye_to_hand_calibrator.get_calibration_info()
        print(f"📋 Calibration Info:")
        print(f"   • Images loaded: {calib_info['image_count']}")
        print(f"   • Robot poses: {calib_info['transform_count']}")
        print(f"   • Pattern: {calib_info['pattern_type']}")
        print(f"   • Has intrinsics: {calib_info['has_intrinsics']}")
        print(f"   • Has extrinsics: {calib_info['has_extrinsics']}")
        print(f"   • Calibration completed: {calib_info['calibration_completed']}")
        
        # Step 7: Test IO methods
        print("\n" + "="*60)
        print("💾 Step 7: Test IO Methods")
        print("="*60)
        
        # Test base2cam matrix operations (should be None initially)
        print(f"📄 Initial base2cam_matrix: {eye_to_hand_calibrator.get_base2cam_matrix()}")
        
        # Test setting a dummy base2cam matrix
        dummy_base2cam = np.eye(4)
        eye_to_hand_calibrator.set_base2cam_matrix(dummy_base2cam)
        print(f"✅ Set dummy base2cam_matrix")
        print(f"📄 Retrieved base2cam_matrix shape: {eye_to_hand_calibrator.get_base2cam_matrix().shape}")
        
        # Test getting calibration results
        results = eye_to_hand_calibrator.get_calibration_results()
        print(f"📊 Calibration results keys: {list(results.keys())}")
        
        # Step 8: Save results
        print("\n" + "="*60)
        print("💾 Step 8: Save Results")
        print("="*60)
        
        try:
            eye_to_hand_calibrator.save_eye_to_hand_results(results_dir)
            print("✅ Eye-to-hand results saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        print("\n" + "="*80)
        print("🎉 NEW EYE-TO-HAND CALIBRATION EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Summary:")
        print(f"   • Performed intrinsic calibration")
        print(f"   • Loaded {len(image_paths)} image-pose pairs for eye-to-hand")
        print(f"   • Intrinsic calibration RMS error: {rms_error:.4f} pixels")
        print(f"   • All data validation: ✅ PASSED")
        print(f"   • IO operations: ✅ TESTED")
        print(f"   • Results saved to: {results_dir}")
        print("\nNote: Calibration algorithms have been moved to dedicated modules.")
        print("This example demonstrates the new IO-only architecture.")
        
        # Demonstrate the new calibrate interface
        print("\n" + "="*60)
        print("🧪 Demonstrating New Calibration Interface")
        print("="*60)
        
        print("📋 Available calibration methods:")
        methods = eye_to_hand_calibrator.get_available_methods()
        for method_id, method_name in methods.items():
            print(f"   • {method_name}: {method_id}")
        
        print(f"\n🔍 Testing placeholder calibrate() method:")
        result = eye_to_hand_calibrator.calibrate(method=None, verbose=True)
        print(f"   Calibration result: {result} (expected: False for IO-only class)")
        
        return True
        
    except Exception as e:
        print(f"❌ Eye-to-hand calibrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
