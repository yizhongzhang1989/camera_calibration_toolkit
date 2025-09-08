#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Example
==============================

Simple example demonstrating eye-in-hand calibration workflow:
1. Perform intrinsic calibration
2. Load robot poses and images
3. Perform eye-in-hand calibration
4. Generate results

Camera is mounted on the robot end-effector.
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


def main():
    """Main eye-in-hand calibration example."""
    print("Eye-in-Hand Calibration Example")
    print("=" * 40)
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
    results_dir = os.path.join(project_root, "data", "results", "eye_in_hand_example")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    
    # Step 1: Load calibration pattern
    with open(os.path.join(data_dir, "chessboard_config.json"), 'r') as f:
        pattern_data = json.load(f)
    calibration_pattern = load_pattern_from_json(pattern_data)
    print(f"Loaded pattern: {calibration_pattern}")
    
    # Step 2: Load images and robot poses together
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
    images = []
    end2base_matrices = []
    
    for img_file in image_files:
        # Load image
        img = cv2.imread(os.path.join(data_dir, img_file))
        if img is not None:
            images.append(img)
            
            # Load corresponding pose matrix
            json_file = img_file.replace('.jpg', '.json')
            json_path = os.path.join(data_dir, json_file)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    pose_data = json.load(f)
                end2base_matrix = np.array(pose_data['end2base'])
                end2base_matrices.append(end2base_matrix)
            else:
                print(f"Warning: No pose file found for {img_file}")
                
    print(f"Loaded {len(images)} images and {len(end2base_matrices)} robot poses")
    
    # Step 3: Perform intrinsic calibration
    print("\nPerforming intrinsic calibration...")
    intrinsic_calibrator = IntrinsicCalibrator(images=images, calibration_pattern=calibration_pattern)
    intrinsic_results = intrinsic_calibrator.calibrate(verbose=True)
    
    if not intrinsic_results:
        print("Intrinsic calibration failed!")
        return 1
    
    camera_matrix = intrinsic_results['camera_matrix']
    distortion_coeffs = intrinsic_results['distortion_coefficients']
    rms_error = intrinsic_results['rms_error']
    print(f"Intrinsic calibration completed - RMS: {rms_error:.4f} pixels")
    
    # Step 4: Load robot poses and images (already loaded in step 2)
    print(f"\nUsing {len(end2base_matrices)} robot poses loaded with images")
    
    # Step 5: Perform eye-in-hand calibration
    print("\nPerforming eye-in-hand calibration...")
    calibrator = EyeInHandCalibrator(
        images=images,
        end2base_matrices=end2base_matrices,
        calibration_pattern=calibration_pattern,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coeffs.flatten(),
        verbose=True
    )
    
    result = calibrator.calibrate(verbose=True)
    
    if result is not None:
        print(f"Eye-in-hand calibration completed - RMS: {result['rms_error']:.4f} pixels")
        
        # Step 6: Generate report
        print("\nGenerating calibration report...")
        report_files = calibrator.generate_calibration_report(results_dir)
        
        if report_files:
            print(f"Report generated: {report_files['html_report']}")
            print(f"JSON data: {report_files['json_data']}")
        
        print("\nCalibration completed successfully!")
        print(f"Camera-to-end transformation matrix:")
        print(result['cam2end_matrix'])
        print(f"Target-to-base transformation matrix:")
        print(result['target2base_matrix'])
        
        # Return 0 for success if RMS < 1, 1 for failure if RMS >= 1
        if result['rms_error'] < 1.0:
            return 0
        else:
            print(f"Warning: RMS error {result['rms_error']:.4f} >= 1.0 pixels")
            return 1
        
    else:
        print("Eye-in-hand calibration failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
