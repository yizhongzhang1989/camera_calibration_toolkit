#!/usr/bin/env python3
"""
Camera Calibration Toolkit - Main Entry Point
==============================================

This script provides a command-line interface to the camera calibration toolkit
for compatibility with existing workflows.
"""

import argparse
import sys
import os
import cv2
import sys
import os

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Toolkit")
    
    parser.add_argument("--mode", choices=['web', 'intrinsic', 'eye_in_hand'], 
                       default='web', help="Calibration mode")
    parser.add_argument("--calib_data_dir", help="Path to calibration images folder")
    parser.add_argument("--xx", type=int, help="Number of corners along chessboard X axis")
    parser.add_argument("--yy", type=int, help="Number of corners along chessboard Y axis") 
    parser.add_argument("--square_size", type=float, help="Size of one chessboard square in meters")
    parser.add_argument("--calib_out_dir", default="data/results", help="Path to output folder for calibration parameters")
    parser.add_argument("--reproj_out_dir", default="data/results", help="Output folder for reprojection visualization")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (web mode only)")
    parser.add_argument("--host", default='localhost', help="Web server host (web mode only)")
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        # Launch web interface
        print("Starting Camera Calibration Toolkit Web Interface...")
        print(f"Server will be available at http://{args.host}:{args.port}")
        
        from web.app import app
        app.run(debug=True, host=args.host, port=args.port)
        
    elif args.mode == 'intrinsic':
        # Run intrinsic calibration
        if not all([args.calib_data_dir, args.xx, args.yy, args.square_size]):
            print("Error: Missing required arguments for intrinsic calibration")
            print("Required: --calib_data_dir, --xx, --yy, --square_size")
            print("Optional: --calib_out_dir (default: data/results)")
            return 1
            
        # Create calibration pattern
        from core.calibration_patterns import create_chessboard_pattern
        from core.utils import load_images_from_directory
        
        pattern = create_chessboard_pattern(
            pattern_type='standard',
            width=args.xx,
            height=args.yy,
            square_size=args.square_size
        )
        
        # Load images from directory
        image_paths = load_images_from_directory(args.calib_data_dir)
        if not image_paths:
            print(f"Error: No valid images found in {args.calib_data_dir}")
            return 1
            
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        # Run calibration
        if not calibrator.detect_pattern_points(verbose=True):
            print("Error: Pattern detection failed!")
            return 1
            
        rms_error = calibrator.calibrate_camera(verbose=True)
        
        if rms_error > 0:
            calibrator.save_calibration(
                os.path.join(args.calib_out_dir, 'calibration_results.json'),
                include_extrinsics=True
            )
            print(f"Intrinsic calibration completed successfully!")
            print(f"RMS Error: {rms_error:.4f} pixels")
            print(f"Results saved to: {args.calib_out_dir}")
        else:
            print("Intrinsic calibration failed!")
            return 1
            
    elif args.mode == 'eye_in_hand':
        # Run eye-in-hand calibration  
        if not all([args.calib_data_dir, args.xx, args.yy, args.square_size]):
            print("Error: Missing required arguments for eye-in-hand calibration")
            print("Required: --calib_data_dir, --xx, --yy, --square_size")
            print("Optional: --calib_out_dir (default: data/results)")
            return 1
            
        # First run intrinsic calibration
        print("Running intrinsic calibration...")
        
        # Create calibration pattern
        pattern = create_chessboard_pattern(
            pattern_type='standard',
            width=args.xx,
            height=args.yy,
            square_size=args.square_size
        )
        
        # Load images from directory
        image_paths = load_images_from_directory(args.calib_data_dir)
        if not image_paths:
            print(f"Error: No valid images found in {args.calib_data_dir}")
            return 1
            
        intrinsic_cal = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        # Run calibration
        if not intrinsic_cal.detect_pattern_points(verbose=True):
            print("Error: Pattern detection failed!")
            return 1
            
        rms_error = intrinsic_cal.calibrate_camera(verbose=True)
        
        if rms_error <= 0:
            print("Intrinsic calibration failed!")
            return 1
            
        # Get calibration results
        camera_matrix = intrinsic_cal.get_camera_matrix()
        dist_coeffs = intrinsic_cal.get_distortion_coefficients()
        
        intrinsic_cal.save_calibration(
            os.path.join(args.calib_out_dir, 'calibration_results.json'),
            include_extrinsics=True
        )
        
        # Run eye-in-hand calibration
        print("Running eye-in-hand calibration...")
        
        # Create calibration pattern
        from core.calibration_patterns import create_chessboard_pattern
        pattern = create_chessboard_pattern('standard', width=args.xx, height=args.yy, square_size=args.square_size)
        
        # Initialize with modern API
        eye_in_hand_cal = EyeInHandCalibrator(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            calibration_pattern=pattern,
            pattern_type='standard'
        )
        
        try:
            # Load calibration data using modern API
            if not eye_in_hand_cal.load_calibration_data(args.calib_data_dir):
                print("Failed to load calibration data")
                return 1
            
            # Detect pattern points
            if not eye_in_hand_cal.detect_pattern_points():
                print("Failed to detect calibration patterns")
                return 1
            
            # Perform calibration
            rms_error = eye_in_hand_cal.calibrate(verbose=True)
            
            if rms_error <= 0:
                print("Eye-in-hand calibration failed")
                return 1
            
            # Save results using modern API
            eye_in_hand_cal.save_results(args.calib_out_dir)
            
            print(f"Eye-in-hand calibration completed successfully!")
            print(f"RMS reprojection error: {rms_error:.4f} pixels")
            print(f"Results saved to: {args.calib_out_dir}")
            
            # Generate visualizations if output directory specified
            if args.reproj_out_dir:
                print("Generating reprojection visualizations...")
                os.makedirs(args.reproj_out_dir, exist_ok=True)
                
                # Generate reprojection images using modern API
                reprojection_images = eye_in_hand_cal.draw_reprojection_on_images()
                
                for filename, debug_img in reprojection_images:
                    output_path = os.path.join(args.reproj_out_dir, f"{filename}_reprojection.jpg")
                    cv2.imwrite(output_path, debug_img)
                
                print(f"Visualizations saved to: {args.reproj_out_dir}")
                
        except Exception as e:
            print(f"Eye-in-hand calibration failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
