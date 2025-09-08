#!/usr/bin/env python3
"""
Eye-to-Hand Calibration Example
==============================

Simple example demonstrating eye-to-hand calibration workflow:
1. Perform intrinsic calibration
2. Load robot poses and images
3. Perform eye-to-hand calibration
4. Generate results

Camera is stationary, looking at the robot end-effector.
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


def main():
    """Main eye-to-hand calibration example."""
    print("Eye-to-Hand Calibration Example")
    print("=" * 40)
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "sample_data", "eye_to_hand_test_data")
    intrinsic_dir = os.path.join(project_root, "sample_data", "intrinsic_calib_grid_test_images")
    results_dir = os.path.join(project_root, "data", "results", "eye_to_hand_example")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Intrinsic calibration directory: {intrinsic_dir}")
    print(f"Results directory: {results_dir}")
    
    # Step 1: Load calibration pattern for intrinsic calibration
    with open(os.path.join(intrinsic_dir, "chessboard_config.json"), 'r') as f:
        intrinsic_pattern_data = json.load(f)
    intrinsic_pattern = load_pattern_from_json(intrinsic_pattern_data)
    print(f"Loaded intrinsic pattern: {intrinsic_pattern}")
    
    # Step 2: Get image paths for intrinsic calibration
    intrinsic_image_files = sorted([f for f in os.listdir(intrinsic_dir) if f.endswith('.jpg')])
    intrinsic_image_paths = [os.path.join(intrinsic_dir, img_file) for img_file in intrinsic_image_files]
    print(f"Found {len(intrinsic_image_paths)} intrinsic calibration image paths")
    
    # Step 3: Perform intrinsic calibration
    print("\nPerforming intrinsic calibration...")
    intrinsic_calibrator = IntrinsicCalibrator(image_paths=intrinsic_image_paths, calibration_pattern=intrinsic_pattern)
    intrinsic_results = intrinsic_calibrator.calibrate()
    
    if not intrinsic_results:
        print("Intrinsic calibration failed!")
        return 1
    
    camera_matrix = intrinsic_results['camera_matrix']
    distortion_coeffs = intrinsic_results['distortion_coefficients']
    rms_error = intrinsic_results['rms_error']
    print(f"Intrinsic calibration completed - RMS: {rms_error:.4f} pixels")
    print("=" * 40)
    
    # Step 4: Load hand-eye calibration pattern
    with open(os.path.join(data_dir, "chessboard_config.json"), 'r') as f:
        handeye_pattern_data = json.load(f)
    handeye_pattern = load_pattern_from_json(handeye_pattern_data)
    print(f"Loaded hand-eye pattern: {handeye_pattern}")
    
    # Step 5: Get image paths for hand-eye calibration
    print("\nPreparing image paths for hand-eye calibration...")
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
    image_paths = []
    
    for img_file in image_files:
        json_file = img_file.replace('.jpg', '.json')
        json_path = os.path.join(data_dir, json_file)
        
        if os.path.exists(json_path):
            image_path = os.path.join(data_dir, img_file)
            image_paths.append(image_path)
    
    print(f"Found {len(image_paths)} image paths with corresponding pose files")
    
    # Step 6: Perform eye-to-hand calibration
    print("\nPerforming eye-to-hand calibration...")
    calibrator = EyeToHandCalibrator(
        image_paths=image_paths,
        calibration_pattern=handeye_pattern,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coeffs.flatten(),
        verbose=True
    )
    
    result = calibrator.calibrate(verbose=True)
    
    if result is not None:
        print(f"Eye-to-hand calibration completed - RMS: {result['rms_error']:.4f} pixels")
        
        # Step 7: Generate report
        print("\nGenerating calibration report...")
        report_files = calibrator.generate_calibration_report(results_dir)
        
        if report_files:
            print(f"Report generated: {report_files['html_report']}")
            print(f"JSON data: {report_files['json_data']}")
        
        print("\nCalibration completed successfully!")
        print(f"Base-to-camera transformation matrix:")
        print(result['base2cam_matrix'])
        print(f"Target-to-end transformation matrix:")
        print(result['target2end_matrix'])
        
        # Return 0 for success if RMS < 1, 1 for failure if RMS >= 1
        if result['rms_error'] < 1.0:
            return 0
        else:
            print(f"Warning: RMS error {result['rms_error']:.4f} >= 1.0 pixels")
            return 1
        
    else:
        print("Eye-to-hand calibration failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)