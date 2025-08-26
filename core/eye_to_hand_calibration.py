"""
Eye-to-Hand Calibration Module
==============================

This module handles eye-to-hand calibration for stationary cameras looking at robot end-effectors.
It calibrates the transformation between the camera and robot base coordinate frame.

Key Difference from Eye-in-Hand:
-------------------------------
- Eye-in-hand: cv2.calibrateHandEye(end2base_Rs, end2base_ts, target2cam_Rs, target2cam_ts)
- Eye-to-hand: cv2.calibrateHandEye(base2end_Rs, base2end_ts, target2cam_Rs, target2cam_ts)

This module inherits from EyeInHandCalibrator and overrides only the calibration function
to use base2end transformations instead of end2base transformations.
"""

import numpy as np
import cv2
import os
from typing import Optional, List, Tuple

from .eye_in_hand_calibration import EyeInHandCalibrator


class EyeToHandCalibrator(EyeInHandCalibrator):
    """
    Eye-to-hand calibration class for stationary cameras observing robot end-effectors.
    
    This class calibrates the transformation between a stationary camera and the robot
    base coordinate frame. The target/calibration pattern is mounted on the robot end-effector.
    
    Inherits from EyeInHandCalibrator but overrides the calibration method to use
    base2end transformations instead of end2base transformations as required by
    OpenCV's calibrateHandEye for eye-to-hand configurations.
    
    Key differences:
    - Uses base2end_Rs and base2end_ts (instead of end2base_Rs and end2base_ts)
    - Calibrates base2cam_matrix (instead of cam2end_matrix)
    - Target is attached directly to end-effector (not held by gripper)
    """
    
    def __init__(self, images=None, image_paths=None, robot_poses=None, 
                 camera_matrix=None, distortion_coefficients=None, 
                 calibration_pattern=None, pattern_type=None):
        """
        Initialize EyeToHandCalibrator with the same interface as EyeInHandCalibrator.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            robot_poses: List of robot poses (transformation matrices or dicts) or None
            camera_matrix: Camera intrinsic matrix or None
            distortion_coefficients: Distortion coefficients array or None
            calibration_pattern: CalibrationPattern instance or None
            pattern_type: Pattern type string for backwards compatibility or None
        """
        # Initialize parent class
        super().__init__(images, image_paths, robot_poses, camera_matrix, 
                        distortion_coefficients, calibration_pattern, pattern_type)
        
        # Override result variable names for eye-to-hand
        self.base2cam_matrix = None          # Base to camera transformation matrix (instead of cam2end_matrix)
        
        print("✅ Eye-to-hand calibrator initialized")
        print("   📋 Pattern: {}".format(self.calibration_pattern.__class__.__name__ if self.calibration_pattern else "None"))
        print("   📷 Camera matrix provided: {}".format(self.camera_matrix is not None))
        print("   🎯 Target mounting: Direct attachment")
    
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_HORAUD, verbose: bool = False) -> float:
        """
        Perform eye-to-hand calibration using the correct OpenCV calibrateHandEye arguments.
        
        Key difference from eye-in-hand: Uses base2end transformations instead of end2base.
        
        Args:
            method: Hand-eye calibration method. Available options:
                - cv2.CALIB_HAND_EYE_TSAI
                - cv2.CALIB_HAND_EYE_PARK  
                - cv2.CALIB_HAND_EYE_HORAUD (default)
                - cv2.CALIB_HAND_EYE_ANDREFF
                - cv2.CALIB_HAND_EYE_DANIILIDIS
            verbose: Whether to print detailed information
            
        Returns:
            float: RMS reprojection error (0.0 if calibration failed)
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters have not been loaded")
            
        if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
            raise ValueError("Robot poses have not been set")
            
        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not detected. Call detect_pattern_points() first.")
            
        if len(self.image_points) != len(self.end2base_matrices):
            raise ValueError(f"Mismatch: {len(self.image_points)} detected patterns vs {len(self.end2base_matrices)} robot poses")
            
        if len(self.image_points) < 3:
            raise ValueError(f"Insufficient data: need at least 3 image-pose pairs, got {len(self.image_points)}")
        
        try:
            if verbose:
                print(f"Running eye-to-hand calibration with {len(self.image_points)} image-pose pairs")
                print(f"Using method: {method}")
                print("Key difference: Using base2end transformations (not end2base)")
        
            # Calculate target to camera transformations from detected pattern points (same as parent)
            self.rvecs = []
            self.tvecs = []
            self.target2cam_matrices = []
            
            for i in range(len(self.image_points)):
                ret, rvec, tvec = cv2.solvePnP(self.object_points[i], self.image_points[i], 
                                              self.camera_matrix, self.distortion_coefficients)
                if ret:
                    self.rvecs.append(rvec)
                    self.tvecs.append(tvec)
                    
                    # Create target to camera transformation matrix
                    target2cam_matrix = np.eye(4)
                    target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                    target2cam_matrix[:3, 3] = tvec[:, 0]
                    self.target2cam_matrices.append(target2cam_matrix)
                else:
                    raise ValueError(f"Could not solve PnP for image {i}")
            
            # CRITICAL DIFFERENCE: Use base2end transformations for eye-to-hand calibration
            base2end_Rs = np.array([matrix[:3, :3] for matrix in self.base2end_matrices])
            base2end_ts = np.array([matrix[:3, 3] for matrix in self.base2end_matrices])
            rvecs_array = np.array([rvec for rvec in self.rvecs])
            tvecs_array = np.array([tvec for tvec in self.tvecs])
            
            if verbose:
                print("Using base2end transformations (eye-to-hand configuration)")
                print(f"base2end_Rs shape: {base2end_Rs.shape}")
                print(f"base2end_ts shape: {base2end_ts.shape}")
                print(f"target2cam rvecs shape: {rvecs_array.shape}")
                print(f"target2cam tvecs shape: {tvecs_array.shape}")
            
            # Perform eye-to-hand calibration using OpenCV with base2end transformations
            # Output is cam2base_R, cam2base_t (camera to base transformation)
            cam2base_R, cam2base_t = cv2.calibrateHandEye(
                base2end_Rs, base2end_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2base 4x4 transformation matrix
            cam2base_4x4 = np.eye(4)
            cam2base_4x4[:3, :3] = cam2base_R
            cam2base_4x4[:3, 3] = cam2base_t[:, 0]
            
            # For eye-to-hand, we need base2cam matrix, so invert the result
            base2cam_4x4 = np.linalg.inv(cam2base_4x4)
            
            # Store results - use base2cam_matrix for eye-to-hand calibration
            self.base2cam_matrix = base2cam_4x4
            self.calibration_completed = True

            # Use the first frame directly for target2end matrix
            cam2base_matrix = np.linalg.inv(base2cam_4x4)
            self.target2end_matrix = self.base2end_matrices[0] @ cam2base_matrix @ self.target2cam_matrices[0]
            
            # Calculate reprojection errors using the eye-to-hand transformation chain
            self.per_image_errors = []
            total_error = 0.0
            valid_images = 0
            
            for i in range(len(self.image_points)):
                try:
                    # Eye-to-hand transformation chain: target2cam = base2cam * base2end * target2end
                    eyetohand_target2cam = base2cam_4x4 @ self.end2base_matrices[i] @ self.target2end_matrix
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyetohand_target2cam[:3, :3], 
                        eyetohand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    self.per_image_errors.append(error)
                    total_error += error * error
                    valid_images += 1
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not calculate reprojection error for image {i}: {e}")
                    self.per_image_errors.append(float('inf'))
            
            # Calculate RMS error
            if valid_images > 0:
                self.rms_error = np.sqrt(total_error / valid_images)
            else:
                self.rms_error = float('inf')
            
            if verbose:
                print("✅ Eye-to-hand calibration completed successfully!")
                print(f"RMS reprojection error: {self.rms_error:.4f} pixels")
                print(f"Base to camera transformation matrix:")
                print(f"{base2cam_4x4}")
                print(f"Per-image errors: {[f'{err:.4f}' for err in self.per_image_errors if not np.isinf(err)]}")

            return self.rms_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Eye-to-hand calibration failed: {e}")
            self.calibration_completed = False
            return 0.0
        
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get the base to camera transformation matrix (eye-to-hand result)."""
        return self.base2cam_matrix
    
    def save_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to JSON file.
        
        Args:
            save_directory: Directory to save the results
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        import os
        import json
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        results = {
            "camera_intrinsics": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist()
            },
            "eye_to_hand_calibration": {
                "base2cam_matrix": self.base2cam_matrix.tolist(),
            }
        }
        
        if hasattr(self, 'target2end_matrix') and self.target2end_matrix is not None:
            results["eye_to_hand_calibration"]["target2end_matrix"] = self.target2end_matrix.tolist()
        
        # Also save the transformation separately for easy loading
        transform_file_path = os.path.join(save_directory, "base2cam_transformation.json")
        transform_data = {
            "base2cam_matrix": self.base2cam_matrix.tolist()
        }
        with open(transform_file_path, 'w', encoding='utf-8') as f:
            json.dump(transform_data, f, indent=4, ensure_ascii=False)
        
        json_file_path = os.path.join(save_directory, "eye_to_hand_calibration_results.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Eye-to-hand calibration results saved to {save_directory}")

    def draw_reprojection_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw pattern point reprojections on original images using eye-to-hand calibration results.
        
        This shows the accuracy of the eye-to-hand calibration by comparing detected pattern points
        with points reprojected using the calibrated transformation matrix.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
        if not self.target2cam_matrices or not self.end2base_matrices:
            raise ValueError("No transformation matrices available. Ensure calibration completed successfully.")
        
        debug_images = []
        
        for i, (img, objp, detected_corners) in enumerate(zip(
            self.images, self.object_points, self.image_points
        )):
            # Use original (distorted) image
            original_img = img.copy()
            
            # Calculate reprojected points using eye-to-hand calibration
            try:
                # Method 1: Direct target to camera from calibration
                target2cam_direct = self.target2cam_matrices[i]
                reprojected_direct, _ = cv2.projectPoints(
                    objp, target2cam_direct[:3, :3], target2cam_direct[:3, 3],
                    self.camera_matrix, self.distortion_coefficients  # Include distortion for original image
                )
                
                # Method 2: Target to camera via eye-to-hand calibration chain
                # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                eyetohand_target2cam = self.base2cam_matrix @ self.end2base_matrices[i] @ self.target2end_matrix
                
                reprojected_eyetohand, _ = cv2.projectPoints(
                    objp, eyetohand_target2cam[:3, :3], eyetohand_target2cam[:3, 3],
                    self.camera_matrix, self.distortion_coefficients  # Include distortion for original image
                )
                
                # Draw detected corners in green (ground truth)
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(original_img, tuple(corner), 8, (0, 255, 0), 2)
                
                # Draw direct reprojection in blue (from direct PnP)
                direct_2d = reprojected_direct.reshape(-1, 2).astype(int)
                for corner in direct_2d:
                    cv2.drawMarker(original_img, tuple(corner), (255, 0, 0), 
                                 cv2.MARKER_CROSS, 12, 2)
                
                # Draw eye-to-hand reprojection in red (from eye-to-hand calibration)
                eyetohand_2d = reprojected_eyetohand.reshape(-1, 2).astype(int)
                for corner in eyetohand_2d:
                    cv2.drawMarker(original_img, tuple(corner), (0, 0, 255), 
                                 cv2.MARKER_TRIANGLE_UP, 12, 2)
                
                # Add legend
                cv2.putText(original_img, "Green: Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(original_img, "Blue: Direct PnP", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(original_img, "Red: Eye-to-Hand", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display error
                if self.per_image_errors and i < len(self.per_image_errors):
                    error_text = f"RMS Error: {self.per_image_errors[i]:.3f} px"
                    cv2.putText(original_img, error_text, (10, original_img.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Warning: Could not generate reprojection for image {i}: {e}")
                # Just draw detected corners if reprojection fails
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(original_img, tuple(corner), 8, (0, 255, 0), 2)
            
            # Get original filename without path and extension
            if hasattr(self, 'image_paths') and self.image_paths and i < len(self.image_paths):
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            else:
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, original_img))
        
        return debug_images
