"""
New Eye-to-Hand Calibration Module
==================================

This module provides the NewEyeToHandCalibrator class that inherits from HandEyeBaseCalibrator.
It follows the same interface and structure as NewEyeInHandCalibrator but implements 
eye-to-hand specific matrix calculations.

Key Features:
- Inherits all common functionality from HandEyeBaseCalibrator
- Provides eye-to-hand specific data structures
- Complete calibration functionality with all OpenCV methods
- Automatic method selection with error comparison

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/IO)
       -> NewEyeToHandCalibrator (eye-to-hand specific calibration)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any

from .hand_eye_base_calibration import HandEyeBaseCalibrator
from .calibration_patterns import CalibrationPattern


class NewEyeToHandCalibrator(HandEyeBaseCalibrator):
    """
    Complete Eye-to-Hand Calibration implementation.
    
    This class provides full calibration functionality for eye-to-hand configurations where:
    - Camera is stationary (fixed in workspace) 
    - Target/calibration pattern is mounted on robot end-effector
    - Calibrates base2cam_matrix (robot base to camera transformation)
    
    Key Data Structures:
    - base2cam_matrix: Transformation from robot base to camera
    - target2end_matrix: Transformation from calibration target to end-effector
    
    Supports all OpenCV hand-eye calibration methods with automatic method selection.
    """
    
    
    def __init__(self, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None,
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize NewEyeToHandCalibrator.
        
        Args:
            images: List of image arrays or None
            end2base_matrices: List of 4x4 end-effector to base transformation matrices
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None
            distortion_coefficients: Camera distortion coefficients or None
        """
        # Initialize base class with common functionality
        super().__init__(images, end2base_matrices, image_paths, calibration_pattern, 
                        camera_matrix, distortion_coefficients)
        
        # Eye-to-hand specific transformation matrices
        self.base2cam_matrix = None           # Robot base to camera transformation (primary result)
        self.target2end_matrix = None         # Target to end-effector transformation (secondary result)
    
    def calibrate(self, method: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform eye-to-hand calibration using the specified method or find the best method.
        
        This method uses the data stored in class members (images, robot poses, camera intrinsics)
        to perform hand-eye calibration. If method is None, all available methods will be tested
        and the one with the lowest reprojection error will be selected.
        
        Args:
            method: Optional OpenCV calibration method constant. If None, all methods will be 
                   tested and the best one selected. Available options:
                   - cv2.CALIB_HAND_EYE_TSAI
                   - cv2.CALIB_HAND_EYE_PARK
                   - cv2.CALIB_HAND_EYE_HORAUD
                   - cv2.CALIB_HAND_EYE_ANDREFF
                   - cv2.CALIB_HAND_EYE_DANIILIDIS
            verbose: Whether to print detailed calibration progress and results
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing calibration results if successful, None if failed.
            Result dictionary contains:
            - 'success': bool - True if calibration succeeded
            - 'method': int - OpenCV method constant used
            - 'method_name': str - Human-readable method name
            - 'base2cam_matrix': np.ndarray - Base to camera transformation matrix
            - 'target2end_matrix': np.ndarray - Target to end-effector transformation matrix
            - 'rms_error': float - Overall RMS reprojection error
            - 'per_image_errors': List[float] - Per-image reprojection errors
            - 'valid_images': int - Number of valid images used in calibration
            - 'total_images': int - Total number of images processed
            
        Note:
            Before calling this method, ensure that:
            1. Images and robot poses are loaded
            2. Camera intrinsic parameters are available
            3. Calibration patterns are detected (call detect_pattern_points())
            4. Target-to-camera matrices are calculated (call _calculate_target2cam_matrices())
        """
        try:
            # detect pattern points
            self.detect_pattern_points()

            # Calculate target2cam matrices
            self._calculate_target2cam_matrices()

            # Validate prerequisites
            self._validate_calibration_prerequisites()
            
            valid_images = len([p for p in self.image_points if p is not None])
            total_images = len(self.image_points) if self.image_points else 0
            
            if verbose:
                print(f"🤖 Running eye-to-hand calibration with {valid_images} image-pose pairs")
            
            # If no method specified, try all methods and find the best
            if method is None:
                if verbose:
                    print("🔍 No method specified, testing all available methods...")
                
                best_method = None
                best_method_name = None
                best_rms_error = float('inf')
                best_base2cam = None
                best_target2end = None
                best_per_image_errors = None
                
                methods_to_try = self.get_available_methods()
                
                for test_method, method_name in methods_to_try.items():
                    if verbose:
                        print(f"\n🧪 Testing method: {method_name} ({test_method})")
                    
                    try:
                        # Perform calibration with this method
                        success, base2cam_matrix, target2end_matrix, rms_error, per_image_errors = self._perform_single_calibration(test_method, verbose)
                        
                        if success and rms_error < best_rms_error:
                            best_method = test_method
                            best_method_name = method_name
                            best_rms_error = rms_error
                            best_base2cam = base2cam_matrix.copy()
                            best_target2end = target2end_matrix.copy()
                            best_per_image_errors = per_image_errors.copy()
                            
                            if verbose:
                                print(f"   ✅ New best method: {method_name} with RMS error {rms_error:.4f}")
                        elif success:
                            if verbose:
                                print(f"   ✅ Method {method_name} succeeded with RMS error {rms_error:.4f}")
                        else:
                            if verbose:
                                print(f"   ❌ Method {method_name} failed")
                                
                    except Exception as e:
                        if verbose:
                            print(f"   ❌ Method {method_name} failed with error: {e}")
                        continue
                
                if best_method is not None:
                    # Store the best results
                    self.base2cam_matrix = best_base2cam
                    self.target2end_matrix = best_target2end
                    self.rms_error = best_rms_error
                    self.per_image_errors = best_per_image_errors
                    self.best_method = best_method
                    self.best_method_name = best_method_name
                    self.calibration_completed = True
                    
                    if verbose:
                        print(f"\n🎉 Best method selected: {best_method_name} with RMS error {best_rms_error:.4f}")
                        print(f"Base to camera transformation matrix:")
                        print(f"{best_base2cam}")
                    
                    # Return calibration results as dictionary
                    return {
                        'success': True,
                        'method': best_method,
                        'method_name': best_method_name,
                        'base2cam_matrix': best_base2cam,
                        'target2end_matrix': best_target2end,
                        'rms_error': best_rms_error,
                        'per_image_errors': best_per_image_errors,
                        'valid_images': valid_images,
                        'total_images': total_images
                    }
                else:
                    if verbose:
                        print("❌ All calibration methods failed")
                    return None
            
            else:
                # Use the specified method
                method_name = self.get_method_name(method)
                if verbose:
                    print(f"🎯 Using specified method: {method_name} ({method})")
                
                success, base2cam_matrix, target2end_matrix, rms_error, per_image_errors = self._perform_single_calibration(method, verbose)
                
                if success:
                    # Store results
                    self.base2cam_matrix = base2cam_matrix
                    self.target2end_matrix = target2end_matrix
                    self.rms_error = rms_error
                    self.per_image_errors = per_image_errors
                    self.best_method = method
                    self.best_method_name = method_name
                    self.calibration_completed = True
                    
                    if verbose:
                        print(f"✅ Eye-to-hand calibration completed successfully!")
                        print(f"RMS reprojection error: {rms_error:.4f} pixels")
                        print(f"Base to camera transformation matrix:")
                        print(f"{base2cam_matrix}")
                    
                    # Return calibration results as dictionary
                    return {
                        'success': True,
                        'method': method,
                        'method_name': method_name,
                        'base2cam_matrix': base2cam_matrix,
                        'target2end_matrix': target2end_matrix,
                        'rms_error': rms_error,
                        'per_image_errors': per_image_errors,
                        'valid_images': valid_images,
                        'total_images': total_images
                    }
                else:
                    if verbose:
                        print(f"❌ Eye-to-hand calibration failed with method {method_name}")
                    return None
                    
        except Exception as e:
            if verbose:
                print(f"❌ Eye-to-hand calibration failed: {e}")
            self.calibration_completed = False
            return None

    # ============================================================================
    # Eye-to-Hand Specific IO Methods
    # ============================================================================
    
    def set_base2cam_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the robot base to camera transformation matrix.
        
        Args:
            matrix: 4x4 transformation matrix from robot base to camera
            
        Raises:
            ValueError: If matrix has invalid format
        """
        if matrix is None:
            self.base2cam_matrix = None
            return
            
        if not isinstance(matrix, np.ndarray):
            raise ValueError("base2cam_matrix must be a numpy array")
        
        if matrix.shape != (4, 4):
            raise ValueError(f"base2cam_matrix must be 4x4, got shape {matrix.shape}")
            
        self.base2cam_matrix = matrix.copy()
    
    def set_target2end_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the target to end-effector transformation matrix.
        
        Args:
            matrix: 4x4 transformation matrix from target to end-effector
            
        Raises:
            ValueError: If matrix has invalid format
        """
        if matrix is None:
            self.target2end_matrix = None
            return
            
        if not isinstance(matrix, np.ndarray):
            raise ValueError("target2end_matrix must be a numpy array")
        
        if matrix.shape != (4, 4):
            raise ValueError(f"target2end_matrix must be 4x4, got shape {matrix.shape}")
            
        self.target2end_matrix = matrix.copy()
    
    def get_base2cam_matrix(self) -> Optional[np.ndarray]:
        """
        Get the robot base to camera transformation matrix.
        
        Returns:
            np.ndarray or None: 4x4 transformation matrix if available
        """
        return self.base2cam_matrix
    
    def get_target2end_matrix(self) -> Optional[np.ndarray]:
        """
        Get the target to end-effector transformation matrix.
        
        Returns:
            np.ndarray or None: 4x4 transformation matrix if available
        """
        return self.target2end_matrix
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the main transformation matrix (base2cam_matrix for eye-to-hand).
        
        Returns:
            np.ndarray or None: Robot base to camera transformation matrix
        """
        return self.base2cam_matrix
    
    # ============================================================================
    # Eye-to-Hand Specific Result Access Methods
    # ============================================================================
    
    def get_calibration_results(self) -> Dict[str, Any]:
        """
        Get eye-to-hand specific calibration results.
        
        Returns:
            dict: Dictionary containing eye-to-hand calibration results
        """
        base_info = self.get_calibration_info()
        
        eye_to_hand_info = {
            "calibration_type": "eye_to_hand",
            "base2cam_matrix": self.base2cam_matrix.tolist() if self.base2cam_matrix is not None else None,
            "target2end_matrix": self.target2end_matrix.tolist() if self.target2end_matrix is not None else None,
            "has_base2cam": self.base2cam_matrix is not None,
            "has_target2end": self.target2end_matrix is not None
        }
        
        # Merge base info with eye-to-hand specific info
        base_info.update(eye_to_hand_info)
        return base_info
    
    def save_eye_to_hand_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results
        """
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            # Get all calibration results
            results = self.get_calibration_results()
            
            # Save main results file
            results_file = os.path.join(save_directory, "eye_to_hand_calibration_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Eye-to-hand calibration results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save eye-to-hand results: {e}")
    
    # ============================================================================
    # Validation Methods
    # ============================================================================
    
    def validate_eye_to_hand_data(self) -> bool:
        """
        Validate that all required data for eye-to-hand calibration is available.
        
        Returns:
            bool: True if all required data is present and valid
        """
        # Check base class data
        if not self.images or len(self.images) == 0:
            print("❌ No images loaded")
            return False
            
        if not self.end2base_matrices or len(self.end2base_matrices) == 0:
            print("❌ No end-effector to base transformation matrices")
            return False
            
        if len(self.images) != len(self.end2base_matrices):
            print(f"❌ Mismatch: {len(self.images)} images vs {len(self.end2base_matrices)} transformation matrices")
            return False
            
        if self.camera_matrix is None:
            print("❌ Camera intrinsic matrix not set")
            return False
            
        if self.distortion_coefficients is None:
            print("❌ Camera distortion coefficients not set")
            return False
            
        if self.calibration_pattern is None:
            print("❌ Calibration pattern not set")
            return False
            
        print("✅ All required data for eye-to-hand calibration is available")
        return True
    
    def is_eye_to_hand_calibrated(self) -> bool:
        """
        Check if eye-to-hand calibration has been completed successfully.
        
        Returns:
            bool: True if eye-to-hand calibration is complete
        """
        return (self.is_calibrated() and 
                self.base2cam_matrix is not None)

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
        
        for i in range(len(self.target2cam_matrices)):
            if self.target2cam_matrices[i] is not None:
                # For eye-to-hand: target2cam = base2cam * end2base * target2end
                # So: target2end = inv(end2base) * inv(base2cam) * target2cam
                cam2base_matrix = np.linalg.inv(base2cam_4x4)
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                candidate_target2end = base2end_matrix @ cam2base_matrix @ self.target2cam_matrices[i]
                candidate_target2end_matrices.append(candidate_target2end)
        
        # Test each candidate to find the one with minimum overall reprojection error
        for candidate_idx, candidate_target2end in enumerate(candidate_target2end_matrices):
            # Use the separate reprojection error function for each candidate
            rms_error, _ = self._calculate_reprojection_errors(base2cam_4x4, candidate_target2end, verbose=False)
            
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
            best_target2end = candidate_target2end_matrices[0] if candidate_target2end_matrices else np.eye(4)
        
        return best_target2end

    def calculate_reprojection_errors(self, base2cam_matrix: Optional[np.ndarray] = None, target2end_matrix: Optional[np.ndarray] = None, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Calculate reprojection errors using hand-eye calibration results.
        
        This is a public method that can be used to calculate reprojection errors
        for given transformation matrices or the stored calibration results.
        
        Args:
            base2cam_matrix: 4x4 base-to-camera transformation matrix. 
                           If None, uses self.base2cam_matrix
            target2end_matrix: 4x4 target-to-end-effector transformation matrix.
                             If None, uses self.target2end_matrix
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple of (rms_error, per_image_errors):
            - rms_error: Overall RMS reprojection error across all valid images
            - per_image_errors: List of reprojection errors for each image (inf for invalid images)
            
        Raises:
            ValueError: If required matrices or data are not available
        """
        # Use provided matrices or stored calibration results
        if base2cam_matrix is None:
            if self.base2cam_matrix is None:
                raise ValueError("No base2cam_matrix provided and no calibration results stored")
            base2cam_matrix = self.base2cam_matrix
            
        if target2end_matrix is None:
            if self.target2end_matrix is None:
                raise ValueError("No target2end_matrix provided and no calibration results stored")
            target2end_matrix = self.target2end_matrix
        
        # Validate prerequisites for reprojection error calculation
        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not detected. Run detect_pattern_points() first.")
            
        if self.end2base_matrices is None:
            raise ValueError("Robot poses not loaded")
            
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters not available")
        
        return self._calculate_reprojection_errors(base2cam_matrix, target2end_matrix, verbose)

    def _calculate_reprojection_errors(self, base2cam_matrix: np.ndarray, target2end_matrix: np.ndarray, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Calculate reprojection errors using hand-eye calibration results.
        
        This method projects 3D calibration points to 2D image points using the calibrated
        hand-eye transformation and compares with detected pattern points to compute
        reprojection errors.
        
        Args:
            base2cam_matrix: 4x4 base-to-camera transformation matrix
            target2end_matrix: 4x4 target-to-end-effector transformation matrix
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple of (rms_error, per_image_errors):
            - rms_error: Overall RMS reprojection error across all valid images
            - per_image_errors: List of reprojection errors for each image (inf for invalid images)
        """
        per_image_errors = []
        total_error = 0.0
        valid_error_count = 0
        
        for i in range(len(self.image_points)):
            if (self.image_points[i] is not None and 
                self.object_points[i] is not None and 
                self.end2base_matrices[i] is not None):
                try:
                    # Calculate reprojected points using hand-eye calibration result
                    # For eye-to-hand: target2cam = base2cam * end2base * target2end
                    eyetohand_target2cam = base2cam_matrix @ self.end2base_matrices[i] @ target2end_matrix
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyetohand_target2cam[:3, :3], 
                        eyetohand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    per_image_errors.append(error)
                    total_error += error * error
                    valid_error_count += 1
                    
                    if verbose:
                        print(f"   Image {i}: Reprojection error = {error:.4f} pixels")
                        
                except Exception as e:
                    if verbose:
                        print(f"   Warning: Could not calculate reprojection error for image {i}: {e}")
                    per_image_errors.append(float('inf'))
            else:
                if verbose:
                    print(f"   Image {i}: Skipped (missing data)")
                per_image_errors.append(float('inf'))
        
        # Calculate RMS error
        if valid_error_count > 0:
            rms_error = np.sqrt(total_error / valid_error_count)
            if verbose:
                print(f"   Overall RMS reprojection error: {rms_error:.4f} pixels ({valid_error_count} valid images)")
        else:
            rms_error = float('inf')
            if verbose:
                print("   No valid images for reprojection error calculation")
        
        return rms_error, per_image_errors

    def _perform_single_calibration(self, method: int, verbose: bool = False) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float, Optional[List[float]]]:
        """
        Perform calibration with a single method.
        
        Args:
            method: OpenCV calibration method constant
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (success, base2cam_matrix, target2end_matrix, rms_error, per_image_errors)
        """
        try:
            # Filter valid data points
            valid_indices = []
            for i in range(len(self.image_points)):
                if (self.image_points[i] is not None and 
                    self.object_points[i] is not None and 
                    self.end2base_matrices[i] is not None and
                    self.target2cam_matrices[i] is not None):
                    valid_indices.append(i)
            
            if len(valid_indices) < 3:
                if verbose:
                    print(f"   Insufficient valid data: {len(valid_indices)} points (need at least 3)")
                return False, None, None, float('inf'), None
            
            # Prepare data for OpenCV calibration
            # For eye-to-hand, we need base2end transformations (inverses of end2base)
            base2end_Rs = []
            base2end_ts = []
            
            for i in valid_indices:
                # Convert end2base to base2end by taking the inverse
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                base2end_Rs.append(base2end_matrix[:3, :3])
                base2end_ts.append(base2end_matrix[:3, 3])
            
            base2end_Rs = np.array(base2end_Rs)
            base2end_ts = np.array(base2end_ts)
            
            # Use rvecs and tvecs from target2cam matrices
            target2cam_Rs = np.array([self.target2cam_matrices[i][:3, :3] for i in valid_indices])
            target2cam_ts = np.array([self.target2cam_matrices[i][:3, 3] for i in valid_indices])
            
            # Convert rotation matrices to rotation vectors
            rvecs_array = np.array([cv2.Rodrigues(R)[0] for R in target2cam_Rs])
            tvecs_array = target2cam_ts.reshape(-1, 3, 1)
            
            # Perform eye-to-hand calibration using OpenCV
            # For eye-to-hand: cv2.calibrateHandEye(base2end_Rs, base2end_ts, target2cam_Rs, target2cam_ts)
            # Returns cam2base_R, cam2base_t (camera to base transformation)
            cam2base_R, cam2base_t = cv2.calibrateHandEye(
                base2end_Rs, base2end_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2base 4x4 transformation matrix
            cam2base_4x4 = np.eye(4)
            cam2base_4x4[:3, :3] = cam2base_R
            cam2base_4x4[:3, 3] = cam2base_t[:, 0]
            
            # For eye-to-hand, we need base2cam matrix, so invert the result
            base2cam_4x4 = np.linalg.inv(cam2base_4x4)
            
            # Calculate the optimal target2end matrix
            target2end_matrix = self._calculate_optimal_target2end_matrix(base2cam_4x4, verbose)
            
            # Calculate reprojection errors using the separate function
            rms_error, per_image_errors = self._calculate_reprojection_errors(base2cam_4x4, target2end_matrix, verbose)
            
            return True, base2cam_4x4, target2end_matrix, rms_error, per_image_errors
            
        except Exception as e:
            if verbose:
                print(f"   Calibration failed: {e}")
            return False, None, None, float('inf'), None

    def save_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results to
        """
        self.save_eye_to_hand_results(save_directory)

    def save_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results to
        """
        self.save_eye_to_hand_results(save_directory)
