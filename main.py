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

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    parser.add_argument("--calib_out_dir", help="Path to output folder for calibration parameters")
    parser.add_argument("--reproj_out_dir", help="Output folder for reprojection visualization")
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
        if not all([args.calib_data_dir, args.xx, args.yy, args.square_size, args.calib_out_dir]):
            print("Error: Missing required arguments for intrinsic calibration")
            print("Required: --calib_data_dir, --xx, --yy, --square_size, --calib_out_dir")
            return 1
            
        calibrator = IntrinsicCalibrator()
        success, camera_matrix, dist_coeffs = calibrator.calibrate_from_directory(
            args.calib_data_dir, args.xx, args.yy, args.square_size, verbose=True)
        
        if success:
            calibrator.save_parameters(args.calib_out_dir)
            print(f"Intrinsic calibration completed successfully!")
            print(f"Results saved to: {args.calib_out_dir}")
        else:
            print("Intrinsic calibration failed!")
            return 1
            
    elif args.mode == 'eye_in_hand':
        # Run eye-in-hand calibration  
        if not all([args.calib_data_dir, args.xx, args.yy, args.square_size, args.calib_out_dir]):
            print("Error: Missing required arguments for eye-in-hand calibration")
            print("Required: --calib_data_dir, --xx, --yy, --square_size, --calib_out_dir")
            return 1
            
        # First run intrinsic calibration
        intrinsic_cal = IntrinsicCalibrator()
        print("Running intrinsic calibration...")
        success, camera_matrix, dist_coeffs = intrinsic_cal.calibrate_from_directory(
            args.calib_data_dir, args.xx, args.yy, args.square_size, verbose=True)
        
        if not success:
            print("Intrinsic calibration failed!")
            return 1
            
        intrinsic_cal.save_parameters(args.calib_out_dir)
        
        # Run eye-in-hand calibration
        print("Running eye-in-hand calibration...")
        eye_in_hand_cal = EyeInHandCalibrator()
        eye_in_hand_cal.load_camera_intrinsics(camera_matrix, dist_coeffs)
        
        try:
            image_paths, base2end_matrices, end2base_matrices = eye_in_hand_cal.load_calibration_data(
                args.calib_data_dir)
            
            cam2end_R, cam2end_t, cam2end_4x4, rvecs, tvecs = eye_in_hand_cal.calibrate(
                image_paths, end2base_matrices, args.xx, args.yy, args.square_size, verbose=True)
            
            # Calculate reprojection errors
            errors, target2base_matrices = eye_in_hand_cal.calculate_reprojection_errors(
                image_paths, base2end_matrices, end2base_matrices, 
                rvecs, tvecs, args.xx, args.yy, args.square_size, 
                vis=(args.reproj_out_dir is not None), 
                save_dir=args.reproj_out_dir)
            
            eye_in_hand_cal.save_results(args.calib_out_dir)
            
            print(f"Eye-in-hand calibration completed successfully!")
            print(f"Mean reprojection error: {errors.mean():.4f} pixels")
            print(f"Results saved to: {args.calib_out_dir}")
            
            if args.reproj_out_dir:
                print(f"Visualizations saved to: {args.reproj_out_dir}")
                
        except Exception as e:
            print(f"Eye-in-hand calibration failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
