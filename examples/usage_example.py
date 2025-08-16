"""
Example: Using the Camera Calibration Toolkit Core Modules
==========================================================

This example demonstrates how to use the core calibration modules
independently of the web interface.
"""

import sys
import os
import numpy as np

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator


def example_intrinsic_calibration():
    """Example of intrinsic camera calibration."""
    print("=== Intrinsic Camera Calibration Example ===")
    
    # Initialize calibrator
    calibrator = IntrinsicCalibrator()
    
    # Calibration parameters
    chessboard_x = 11  # corners along x-axis
    chessboard_y = 8   # corners along y-axis
    square_size = 0.02  # meters per square
    
    # Path to calibration images
    images_directory = "path/to/intrinsic/calibration/images"
    
    try:
        # Run calibration
        success, camera_matrix, dist_coeffs = calibrator.calibrate_from_directory(
            images_directory, chessboard_x, chessboard_y, square_size, verbose=True)
        
        if success:
            print("Calibration successful!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs}")
            
            # Save results
            calibrator.save_parameters("./results/intrinsic")
            print("Results saved to ./results/intrinsic")
            
            # Example: undistort an image
            undistorted = calibrator.undistort_image(
                "path/to/test/image.jpg", 
                "path/to/undistorted/image.jpg"
            )
            
        else:
            print("Calibration failed!")
            
    except Exception as e:
        print(f"Error during intrinsic calibration: {e}")


def example_eye_in_hand_calibration():
    """Example of eye-in-hand calibration."""
    print("\n=== Eye-in-Hand Calibration Example ===")
    
    # Initialize calibrators
    intrinsic_cal = IntrinsicCalibrator()
    eye_in_hand_cal = EyeInHandCalibrator()
    
    # Calibration parameters
    chessboard_x = 11
    chessboard_y = 8
    square_size = 0.02
    
    # Paths
    calibration_data_dir = "path/to/eye_in_hand/data"  # Contains images and pose JSON files
    
    try:
        # Step 1: Load or perform intrinsic calibration
        try:
            # Try to load existing intrinsic parameters
            camera_matrix, dist_coeffs = intrinsic_cal.load_parameters("./results/intrinsic")
            print("Loaded existing intrinsic parameters")
        except:
            # Perform intrinsic calibration
            print("Performing intrinsic calibration...")
            success, camera_matrix, dist_coeffs = intrinsic_cal.calibrate_from_directory(
                calibration_data_dir, chessboard_x, chessboard_y, square_size)
            
            if not success:
                print("Intrinsic calibration failed!")
                return
        
        # Step 2: Load camera intrinsics into eye-in-hand calibrator
        eye_in_hand_cal.load_camera_intrinsics(camera_matrix, dist_coeffs)
        
        # Step 3: Load calibration data (images and robot poses)
        image_paths, base2end_matrices, end2base_matrices = eye_in_hand_cal.load_calibration_data(
            calibration_data_dir)
        
        print(f"Loaded {len(image_paths)} calibration images")
        
        # Step 4: Perform eye-in-hand calibration
        print("Running eye-in-hand calibration...")
        cam2end_R, cam2end_t, cam2end_4x4, rvecs, tvecs = eye_in_hand_cal.calibrate(
            image_paths, end2base_matrices, chessboard_x, chessboard_y, square_size, verbose=True)
        
        print("Camera to end-effector transformation matrix:")
        print(cam2end_4x4)
        
        # Step 5: Calculate reprojection errors
        print("Calculating reprojection errors...")
        errors, target2base_matrices = eye_in_hand_cal.calculate_reprojection_errors(
            image_paths, base2end_matrices, end2base_matrices, 
            rvecs, tvecs, chessboard_x, chessboard_y, square_size,
            vis=True, save_dir="./results/visualizations")
        
        print(f"Mean reprojection error: {np.mean(errors):.4f} pixels")
        print(f"Error statistics: min={np.min(errors):.4f}, max={np.max(errors):.4f}, std={np.std(errors):.4f}")
        
        # Step 6: Optional optimization
        print("Running optimization...")
        optimized_cam2end, optimized_target2base = eye_in_hand_cal.optimize_calibration(
            image_paths, rvecs, tvecs, end2base_matrices, base2end_matrices,
            chessboard_x, chessboard_y, square_size, iterations=3)
        
        print("Optimized camera to end-effector transformation matrix:")
        print(optimized_cam2end)
        
        # Step 7: Save results
        eye_in_hand_cal.save_results("./results/eye_in_hand")
        print("Results saved to ./results/eye_in_hand")
        
    except Exception as e:
        print(f"Error during eye-in-hand calibration: {e}")


def example_using_calibration_results():
    """Example of how to use calibration results for other applications."""
    print("\n=== Using Calibration Results Example ===")
    
    try:
        # Load calibration results
        eye_in_hand_cal = EyeInHandCalibrator()
        eye_in_hand_cal.load_results("./results/eye_in_hand/eye_in_hand_calibration_results.json")
        
        # Get the transformation matrix
        cam2end_matrix = eye_in_hand_cal.get_transformation_matrix()
        
        # Example: Transform a point from camera coordinates to end-effector coordinates
        point_in_camera = np.array([0.1, 0.05, 0.5, 1.0])  # homogeneous coordinates
        point_in_end_effector = cam2end_matrix @ point_in_camera
        
        print(f"Point in camera frame: {point_in_camera[:3]}")
        print(f"Point in end-effector frame: {point_in_end_effector[:3]}")
        
        # Example: Transform from end-effector to camera
        end2cam_matrix = np.linalg.inv(cam2end_matrix)
        point_back_in_camera = end2cam_matrix @ point_in_end_effector
        
        print(f"Point transformed back to camera frame: {point_back_in_camera[:3]}")
        
    except Exception as e:
        print(f"Error using calibration results: {e}")


def create_example_pose_file():
    """Create an example robot pose file."""
    import json
    
    example_pose = {
        "end_xyzrpy": {
            "x": 0.5,      # X position in meters
            "y": 0.2,      # Y position in meters  
            "z": 0.3,      # Z position in meters
            "rx": 0.0,     # Roll in radians
            "ry": 0.0,     # Pitch in radians
            "rz": 1.57     # Yaw in radians (90 degrees)
        }
    }
    
    with open("example_pose.json", "w") as f:
        json.dump(example_pose, f, indent=4)
    
    print("Example pose file created: example_pose.json")


if __name__ == "__main__":
    print("Camera Calibration Toolkit - Core Modules Examples")
    print("=" * 50)
    
    # Create example pose file
    create_example_pose_file()
    
    # Note: You'll need to update the paths to your actual data directories
    print("\nNote: Please update the file paths in this example to point to your actual calibration data.")
    print("The examples will not run without proper calibration images and pose files.")
    
    # Uncomment the lines below once you have proper calibration data:
    
    # example_intrinsic_calibration()
    # example_eye_in_hand_calibration() 
    # example_using_calibration_results()
