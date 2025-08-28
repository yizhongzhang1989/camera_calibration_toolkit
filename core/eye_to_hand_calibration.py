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

# Import nlopt for optimization (same as eye-in-hand)
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    nlopt = None
    HAS_NLOPT = False
    print("Warning: nlopt not available. Optimization methods will be disabled.")

# Import utility functions from parent module
from .utils import (
    xyz_rpy_to_matrix,
    matrix_to_xyz_rpy,
)

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
                 calibration_pattern=None):
        """
        Initialize EyeToHandCalibrator with the same interface as EyeInHandCalibrator.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            robot_poses: List of robot poses (transformation matrices or dicts) or None
            camera_matrix: Camera intrinsic matrix or None
            distortion_coefficients: Distortion coefficients array or None
            calibration_pattern: CalibrationPattern instance or None
        """
        # Initialize parent class
        super().__init__(images, image_paths, robot_poses, camera_matrix, 
                        distortion_coefficients, calibration_pattern)
        
        # Override result variable names for eye-to-hand
        self.base2cam_matrix = None          # Base to camera transformation matrix (instead of cam2end_matrix)
        
        print("✅ Eye-to-hand calibrator initialized")
        print("   📋 Pattern: {}".format(self.calibration_pattern.__class__.__name__ if self.calibration_pattern else "None"))
        print("   📷 Camera matrix provided: {}".format(self.camera_matrix is not None))
        print("   🎯 Target mounting: Direct attachment")
    
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_HORAUD, verbose: bool = False) -> bool:
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
            bool: True if calibration succeeded, False if failed
            
        Note:
            After successful calibration, use getter methods to access results:
            - get_rms_error(): Overall RMS reprojection error
            - get_transformation_matrix(): Base to camera transform
            - get_per_image_errors(): Per-image reprojection errors
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

            # Calculate the optimal target2end matrix that minimizes overall reprojection error
            self.target2end_matrix = self._calculate_optimal_target2end_matrix(base2cam_4x4, verbose)
            
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

            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Eye-to-hand calibration failed: {e}")
            self.calibration_completed = False
            return False
        
    def get_rms_error(self) -> Optional[float]:
        """Get overall RMS reprojection error (lower is better)."""
        return self.rms_error
    
    def get_per_image_errors(self) -> Optional[List[float]]:
        """Get per-image reprojection errors."""
        return self.per_image_errors
        
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

    def _calculate_optimal_target2end_matrix(self, base2cam_4x4: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate the single target2end matrix that minimizes overall reprojection error.
        
        This method finds the target2end transformation that best explains all the
        observed target positions across all images, rather than calculating a
        separate target2end for each image which would result in zero error.
        
        Args:
            base2cam_4x4: The base to camera transformation matrix from calibration
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to end-effector transformation matrix
        """
        if verbose:
            print("Calculating optimal target2end matrix...")
        
        best_error = float('inf')
        best_target2end = None
        
        # Try using each image's target2cam transformation to estimate target2end
        # Then find the one that gives the smallest overall reprojection error
        candidate_target2end_matrices = []
        cam2base_matrix = np.linalg.inv(base2cam_4x4)
        
        for i in range(len(self.target2cam_matrices)):
            # For eye-to-hand: target2end = base2end * cam2base * target2cam
            # where base2end = (end2base)^-1 and cam2base = (base2cam)^-1
            try:
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                candidate_target2end = base2end_matrix @ cam2base_matrix @ self.target2cam_matrices[i]
                candidate_target2end_matrices.append(candidate_target2end)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not calculate candidate target2end for image {i}: {e}")
                continue
        
        if not candidate_target2end_matrices:
            if verbose:
                print("❌ No valid candidate target2end matrices found")
            return np.eye(4)
        
        # Test each candidate to find the one with minimum overall reprojection error
        for candidate_idx, candidate_target2end in enumerate(candidate_target2end_matrices):
            total_error = 0.0
            valid_images = 0
            
            for i in range(len(self.image_points)):
                try:
                    # Calculate target2cam using the candidate target2end matrix
                    # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                    eyetohand_target2cam = base2cam_4x4 @ self.end2base_matrices[i] @ candidate_target2end
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyetohand_target2cam[:3, :3], 
                        eyetohand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    total_error += error * error
                    valid_images += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not test candidate {candidate_idx} for image {i}: {e}")
                    continue
            
            # Calculate RMS error for this candidate
            if valid_images > 0:
                rms_error = np.sqrt(total_error / valid_images)
                if rms_error < best_error:
                    best_error = rms_error
                    best_target2end = candidate_target2end.copy()
                    if verbose:
                        print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f} (best so far)")
                elif verbose:
                    print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f}")
        
        if best_target2end is not None:
            if verbose:
                print(f"✅ Optimal target2end matrix found with RMS error: {best_error:.4f}")
                print("Target2end transformation matrix:")
                print(best_target2end)
        else:
            if verbose:
                print("⚠️ Could not find optimal target2end matrix, using first candidate")
            best_target2end = candidate_target2end_matrices[0]
        
        return best_target2end

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

    def optimize_calibration(self, iterations: int = 5, ftol_rel: float = 1e-6, verbose: bool = False) -> float:
        """
        Optimize eye-to-hand calibration using nonlinear optimization.
        
        This method refines the calibration results obtained from the initial calibration
        by minimizing the overall reprojection error across all images.
        
        Args:
            iterations: Number of optimization iterations (default: 5)
            ftol_rel: Relative tolerance for optimization (default: 1e-6)
            verbose: Whether to print detailed optimization information
            
        Returns:
            float: Final RMS reprojection error after optimization
            
        Note:
            This method requires the initial calibration to be completed first.
            All required data (images, poses, etc.) should already be loaded as member variables.
        """
        if not self.calibration_completed:
            raise ValueError("Initial calibration must be completed before optimization")
            
        if not HAS_NLOPT:
            if verbose:
                print("Warning: nlopt not available. Returning current calibration without optimization.")
            return self.rms_error
            
        if verbose:
            print("🔧 Starting calibration optimization...")
            print(f"   Initial RMS error: {self.rms_error:.4f} pixels")
            print(f"   Optimization iterations: {iterations}")
            print(f"   Convergence tolerance: {ftol_rel}")
        
        # Get pattern parameters from calibration pattern
        if self.calibration_pattern is None:
            raise ValueError("Optimization requires a calibration pattern to be set")
        
        try:
            # Find the image with minimum reprojection error as optimization starting point
            min_error_idx = np.argmin([e for e in self.per_image_errors if not np.isinf(e)])
            if verbose:
                print(f"   Using image {min_error_idx} (error: {self.per_image_errors[min_error_idx]:.4f}) as starting point")
            
            # Initial values for optimization (eye-to-hand uses different matrices)
            initial_base2cam = self.base2cam_matrix.copy()
            initial_target2end = self.target2end_matrix.copy()
            initial_error = self.rms_error
            
            # Joint optimization of both matrices simultaneously
            optimized_base2cam, optimized_target2end = self._optimize_matrices_jointly(
                initial_base2cam, initial_target2end, ftol_rel, verbose)
            
            # Update calibration results with optimized values
            self.base2cam_matrix = optimized_base2cam
            self.target2end_matrix = optimized_target2end
            
            # Recalculate reprojection errors with optimized parameters
            self._recalculate_reprojection_errors()
            
            self.optimization_completed = True
            
            if verbose:
                improvement = initial_error - self.rms_error
                print(f"✅ Optimization completed!")
                print(f"   Final RMS error: {self.rms_error:.4f} pixels")
                print(f"   Improvement: {improvement:.4f} pixels")
            
            return self.rms_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Optimization failed: {e}")
            return self.rms_error
    
    def _calculate_optimization_error(self, base2cam_matrix: np.ndarray, target2end_matrix: np.ndarray) -> float:
        """
        Calculate RMS reprojection error for given transformation matrices.
        
        Args:
            base2cam_matrix: Base to camera transformation matrix
            target2end_matrix: Target to end-effector transformation matrix
            
        Returns:
            float: RMS reprojection error
        """
        total_error = 0.0
        valid_images = 0
        
        for i in range(len(self.image_points)):
            try:
                # Calculate target to camera using eye-to-hand calibration chain
                # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                target2cam = base2cam_matrix @ self.end2base_matrices[i] @ target2end_matrix
                
                # Project 3D points to image
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], 
                    target2cam[:3, :3], 
                    target2cam[:3, 3], 
                    self.camera_matrix, 
                    self.distortion_coefficients)
                
                # Calculate reprojection error
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error * error
                valid_images += 1
                
            except Exception:
                continue
        
        if valid_images > 0:
            return np.sqrt(total_error / valid_images)
        else:
            return float('inf')

    def _optimize_target2end_matrix(self, initial_target2end: np.ndarray, base2cam_matrix: np.ndarray,
                                   ftol_rel: float, verbose: bool) -> np.ndarray:
        """
        Optimize the target2end matrix using NLopt.
        
        Args:
            initial_target2end: Initial target2end transformation matrix
            base2cam_matrix: Fixed base to camera transformation matrix
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: Optimized target2end transformation matrix
        """
        if not HAS_NLOPT:
            return initial_target2end
            
        try:
            import nlopt
            
            # Extract initial pose from matrix
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(initial_target2end)
            
            # Setup optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
            
            def objective(params, grad):
                x, y, z, roll, pitch, yaw = params
                target2end_matrix = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
                return self._calculate_optimization_error(base2cam_matrix, target2end_matrix)
            
            opt.set_min_objective(objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize
            try:
                opt_params = opt.optimize([x, y, z, roll, pitch, yaw])
                optimized_matrix = xyz_rpy_to_matrix(opt_params)
                
                if verbose:
                    initial_error = objective([x, y, z, roll, pitch, yaw], None)
                    final_error = objective(opt_params, None)
                    print(f"       Target2end: {initial_error:.4f} -> {final_error:.4f} pixels")
                
                return optimized_matrix
                
            except Exception as opt_e:
                if verbose:
                    print(f"       Target2end optimization failed: {opt_e}")
                return initial_target2end
                
        except ImportError:
            return initial_target2end

    def _optimize_base2cam_matrix(self, initial_base2cam: np.ndarray, target2end_matrix: np.ndarray,
                               ftol_rel: float, verbose: bool) -> np.ndarray:
        """
        Optimize the base2cam matrix using NLopt.
        
        Args:
            initial_base2cam: Initial base to camera transformation matrix
            target2end_matrix: Fixed target to end-effector transformation matrix
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: Optimized base to camera transformation matrix
        """
        if not HAS_NLOPT:
            return initial_base2cam
            
        try:
            import nlopt
            
            # Extract initial pose from matrix
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(initial_base2cam)
            
            # Setup optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
            
            def objective(params, grad):
                x, y, z, roll, pitch, yaw = params
                base2cam_matrix = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
                return self._calculate_optimization_error(base2cam_matrix, target2end_matrix)
            
            opt.set_min_objective(objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize
            try:
                opt_params = opt.optimize([x, y, z, roll, pitch, yaw])
                optimized_matrix = xyz_rpy_to_matrix(opt_params)
                
                if verbose:
                    initial_error = objective([x, y, z, roll, pitch, yaw], None)
                    final_error = objective(opt_params, None)
                    print(f"       Base2cam: {initial_error:.4f} -> {final_error:.4f} pixels")
                
                return optimized_matrix
                
            except Exception as opt_e:
                if verbose:
                    print(f"       Base2cam optimization failed: {opt_e}")
                return initial_base2cam
                
        except ImportError:
            return initial_base2cam

    def _recalculate_reprojection_errors(self):
        """Recalculate per-image errors and overall RMS error with current matrices."""
        self.per_image_errors = []
        total_error = 0.0
        valid_images = 0
        
        for i in range(len(self.image_points)):
            try:
                # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                target2cam = self.base2cam_matrix @ self.end2base_matrices[i] @ self.target2end_matrix
                
                # Project 3D points to image
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i],
                    target2cam[:3, :3],
                    target2cam[:3, 3],
                    self.camera_matrix,
                    self.distortion_coefficients)
                
                # Calculate reprojection error for this image
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                self.per_image_errors.append(error)
                total_error += error * error
                valid_images += 1
            except Exception:
                self.per_image_errors.append(float('inf'))
        
        # Update RMS error
        if valid_images > 0:
            self.rms_error = np.sqrt(total_error / valid_images)
        else:
            self.rms_error = float('inf')

    def _optimize_matrices_jointly(self, initial_base2cam, initial_target2end, ftol_rel, verbose):
        """
        Optimize both base2cam and target2end matrices simultaneously.
        This should converge faster than iterative optimization.
        
        Args:
            initial_base2cam: Initial base-to-camera transformation matrix
            initial_target2end: Initial target-to-end-effector transformation matrix  
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print optimization progress
            
        Returns:
            tuple: (optimized_base2cam, optimized_target2end) matrices
        """
        try:
            import nlopt
            
            # Convert initial matrices to parameter vectors
            base2cam_params = matrix_to_xyz_rpy(initial_base2cam)  # [x, y, z, roll, pitch, yaw]
            target2end_params = matrix_to_xyz_rpy(initial_target2end)  # [x, y, z, roll, pitch, yaw]
            
            # Combined parameter vector: [base2cam_params, target2end_params] (12 total)
            initial_params = np.concatenate([base2cam_params, target2end_params])
            
            # Setup joint optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 12)  # 12 parameters total
            
            def joint_objective(params, grad):
                """Objective function that optimizes both matrices simultaneously."""
                # Split parameters back into two matrices
                base2cam_params = params[:6]  # First 6 parameters for base2cam
                target2end_params = params[6:]  # Last 6 parameters for target2end
                
                # Convert to matrices
                base2cam_matrix = xyz_rpy_to_matrix(base2cam_params)
                target2end_matrix = xyz_rpy_to_matrix(target2end_params)
                
                # Calculate error using both matrices
                return self._calculate_optimization_error(base2cam_matrix, target2end_matrix)
            
            opt.set_min_objective(joint_objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize both matrices jointly
            try:
                optimized_params = opt.optimize(initial_params)
                
                # Split optimized parameters back into matrices
                optimized_base2cam = xyz_rpy_to_matrix(optimized_params[:6])
                optimized_target2end = xyz_rpy_to_matrix(optimized_params[6:])
                
                if verbose:
                    initial_error = joint_objective(initial_params, None)
                    final_error = joint_objective(optimized_params, None)
                    print(f"   Joint optimization: {initial_error:.4f} -> {final_error:.4f} pixels")
                    improvement = (initial_error - final_error) / initial_error * 100
                    print(f"   Improvement: {improvement:.1f}%")
                
                return optimized_base2cam, optimized_target2end
                
            except Exception as opt_e:
                if verbose:
                    print(f"   Joint optimization failed: {opt_e}")
                return initial_base2cam, initial_target2end
                
        except ImportError:
            if verbose:
                print("   nlopt not available, skipping joint optimization")
            return initial_base2cam, initial_target2end
