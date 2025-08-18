#!/usr/bin/env python3
"""
Legacy Compatibility Script
===========================

This script provides a compatibility layer to run the original 
duco_camera_calibrateOPT_cmd.py functionality using the new modular structure.
"""

import sys
import os
import argparse
import cv2

# Add the new toolkit to the path
toolkit_path = os.path.join(os.path.dirname(__file__), 'camera_calibration_toolkit')
sys.path.append(toolkit_path)

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator


def main(CALIBRATION_DATA_DIR, CALIBRATION_PARAMETER_OUTPUT_DIR, XX, YY, L, REPROJECTION_RESULT_OUTPUT_DIR=None):
    """
    Main function that replicates the functionality of the original script
    using the new modular components.
    """
    
    print(f"Calibration images folder: {CALIBRATION_DATA_DIR}")
    print(f"Calibration params output folder: {CALIBRATION_PARAMETER_OUTPUT_DIR}")
    print(f"Chessboard size: {XX} x {YY}, square size: {L} m")
    print(f"Visualize reprojection: {REPROJECTION_RESULT_OUTPUT_DIR is not None}")
    
    # Ensure output directories exist
    os.makedirs(CALIBRATION_PARAMETER_OUTPUT_DIR, exist_ok=True)
    if REPROJECTION_RESULT_OUTPUT_DIR:
        os.makedirs(REPROJECTION_RESULT_OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Intrinsic calibration
        print("Starting intrinsic calibration...")
        
        # Create calibration pattern
        from core.calibration_patterns import create_chessboard_pattern
        from core.utils import load_images_from_directory
        
        pattern = create_chessboard_pattern(
            pattern_type='standard',
            width=XX,
            height=YY,
            square_size=L
        )
        
        # Load images from directory
        image_paths = load_images_from_directory(CALIBRATION_DATA_DIR)
        if not image_paths:
            print(f"Error: No valid images found in {CALIBRATION_DATA_DIR}")
            return
            
        intrinsic_cal = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        # Run calibration
        if not intrinsic_cal.detect_pattern_points(verbose=True):
            print("Error: Pattern detection failed!")
            return
            
        rms_error = intrinsic_cal.calibrate_camera(verbose=True)
        
        if rms_error > 0:
            # Get calibration results
            camera_matrix = intrinsic_cal.get_camera_matrix()
            dist_coeffs = intrinsic_cal.get_distortion_coefficients()
            
            intrinsic_cal.save_calibration(
                os.path.join(CALIBRATION_PARAMETER_OUTPUT_DIR, 'calibration_results.json'),
                include_extrinsics=True
            )
            print("Intrinsic calibration completed successfully")
            print(f"RMS Error: {rms_error:.4f} pixels")
        else:
            print("Error: Intrinsic calibration failed!")
            return
        
        # Step 2: Eye-in-hand calibration
        print("Starting eye-in-hand calibration...")
        
        # Create calibration pattern
        from core.calibration_patterns import create_chessboard_pattern
        pattern = create_chessboard_pattern('standard', width=XX, height=YY, square_size=L)
        
        # Initialize with modern API
        eye_in_hand_cal = EyeInHandCalibrator(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        # Load calibration data using modern API
        if not eye_in_hand_cal.load_calibration_data(CALIBRATION_DATA_DIR):
            print("Error: Failed to load calibration data!")
            return
        
        # Detect pattern points
        if not eye_in_hand_cal.detect_pattern_points():
            print("Error: Failed to detect calibration patterns!")
            return
        
        # Perform calibration
        rms_error = eye_in_hand_cal.calibrate(verbose=True)
        
        if rms_error <= 0:
            print("Error: Eye-in-hand calibration failed!")
            return
        
        print(f"Initial RMS reprojection error: {rms_error:.6f} pixels")
        
        # Step 3: Optimization using modern API
        print("Starting calibration optimization...")
        optimized_error = eye_in_hand_cal.optimize_calibration(
            iterations=5, ftol_rel=1e-6, verbose=True)
        
        if optimized_error > 0:
            print(f"Optimized RMS reprojection error: {optimized_error:.6f} pixels")
            improvement = (rms_error - optimized_error) / rms_error * 100
            print(f"Improvement: {improvement:.1f}%")
        else:
            print("Optimization completed with no significant improvement")
        
        # Save final results using modern API
        eye_in_hand_cal.save_results(CALIBRATION_PARAMETER_OUTPUT_DIR)
        
        # Generate visualizations if requested
        if REPROJECTION_RESULT_OUTPUT_DIR:
            print("Generating reprojection visualizations...")
            
            # Generate reprojection images using modern API
            reprojection_images = eye_in_hand_cal.draw_reprojection_on_images()
            
            for filename, debug_img in reprojection_images:
                output_path = os.path.join(REPROJECTION_RESULT_OUTPUT_DIR, f"{filename}_reprojection.jpg")
                cv2.imwrite(output_path, debug_img)
            
            print(f"Reprojection visualizations saved to: {REPROJECTION_RESULT_OUTPUT_DIR}")
        print(f"Calibration completed successfully!")
        print(f"Results saved to: {CALIBRATION_PARAMETER_OUTPUT_DIR}")
        
        if REPROJECTION_RESULT_OUTPUT_DIR:
            print(f"Visualizations saved to: {REPROJECTION_RESULT_OUTPUT_DIR}")
        
        return 0
        
    except Exception as e:
        print(f"Calibration failed with error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye-in-Hand Calibration (Legacy Compatibility)")

    parser.add_argument("--calib_data_dir", required=True, help="Path to calibration images folder")
    parser.add_argument("--xx", type=int, required=True, help="Number of corners along chessboard X axis")
    parser.add_argument("--yy", type=int, required=True, help="Number of corners along chessboard Y axis")
    parser.add_argument("--square_size", type=float, required=True, help="Size of one chessboard square in meters")
    parser.add_argument("--calib_out_dir", default="data/results", help="Path to output folder for calibration parameters")
    parser.add_argument("--reproj_out_dir", default="data/results", help="(Optional) Output folder for reprojection visualization")

    args = parser.parse_args()

    exit_code = main(
        args.calib_data_dir, 
        args.calib_out_dir, 
        args.xx, 
        args.yy, 
        args.square_size, 
        args.reproj_out_dir
    )
    
    sys.exit(exit_code)
