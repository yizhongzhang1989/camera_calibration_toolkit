"""
Eye-in-Hand Calibration Module
==============================

This module provides the EyeInHandCalibrator class that inherits from HandEyeBaseCalibrator.
It contains only IO functionality for eye-in-hand calibration data handling.

Key Features:
- Inherits all common IO functionality from HandEyeBaseCalibrator
- Provides eye-in-hand specific data structures
- Maintains API compatibility for data handling
- Separates IO from calibration logic

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/IO)
       -> EyeInHandCalibrator (eye-in-hand specific IO)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any

# Optional import for optimization
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    nlopt = None
    HAS_NLOPT = False

from .hand_eye_base_calibration import HandEyeBaseCalibrator
from .calibration_patterns import CalibrationPattern
from .utils import xyz_rpy_to_matrix, matrix_to_xyz_rpy
from .calibration_patterns import CalibrationPattern


class EyeInHandCalibrator(HandEyeBaseCalibrator):
    """
    Eye-in-Hand Calibration class for IO operations only.
    
    This class provides data handling for eye-in-hand calibration where the camera
    is mounted on the robot end-effector. Contains only IO and data management functions.
    
    Key Data Structures:
    - cam2end_matrix: Transformation from camera to end-effector
    - target2base_matrix: Transformation from calibration target to robot base
    
    The calibration logic is separated and should be implemented elsewhere.
    """
    
    def __init__(self, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None, 
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None,
                 verbose: bool = False):
        """
        Initialize EyeInHandCalibrator for IO operations.
        
        Args:
            images: List of image arrays or None
            end2base_matrices: List of 4x4 end-effector to base transformation matrices
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None
            distortion_coefficients: Camera distortion coefficients or None
            verbose: Whether to print progress information during initialization (default: False)
        """
        # Initialize base class with common functionality
        super().__init__(images, end2base_matrices, image_paths, calibration_pattern, 
                        camera_matrix, distortion_coefficients, verbose=verbose)
        
        # Eye-in-hand specific transformation matrices
        self.cam2end_matrix = None              # Camera to end-effector transformation (primary result)
        self.target2base_matrix = None          # Target to robot base transformation (secondary result)
    
    def calibrate(self, method: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform eye-in-hand calibration using the specified method or find the best method.
        
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
            - 'cam2end_matrix': np.ndarray - Camera to end-effector transformation matrix
            - 'target2base_matrix': np.ndarray - Target to base transformation matrix
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
            self.detect_pattern_points(verbose=verbose)

            # Calculate target2cam matrices
            self._calculate_target2cam_matrices(verbose=verbose)

            # Validate prerequisites
            self._validate_calibration_prerequisites()
            
            valid_images = len([p for p in self.image_points if p is not None])
            total_images = len(self.image_points) if self.image_points else 0
            
            if verbose:
                print(f"🤖 Running eye-in-hand calibration with {valid_images} image-pose pairs")
            
            # Determine which methods to test
            available_methods = self.get_available_methods()
            
            if method is None or method not in available_methods:
                # No method provided or invalid method - try all available methods
                methods_to_try = available_methods
                if verbose:
                    if method is None:
                        print("🔍 No method specified, testing all available methods...")
                    else:
                        print(f"⚠️ Invalid method specified: {method}")
                        print(f"🔍 Valid methods are: {list(available_methods.keys())}")
                        print(f"🔍 Falling back to testing all available methods...")
            else:
                # Valid method provided - use only that method
                method_name = available_methods[method]
                methods_to_try = {method: method_name}
                if verbose:
                    print(f"🎯 Using specified method: {method_name} ({method})")
            
            # Try methods and find the best result
            best_method = None
            best_method_name = None
            best_rms_error = float('inf')
            best_cam2end = None
            best_target2base = None
            best_per_image_errors = None
            
            for test_method, method_name in methods_to_try.items():
                if verbose and len(methods_to_try) > 1:
                    print(f"\n🧪 Testing method: {method_name} ({test_method})")
                
                try:
                    # Perform calibration with this method
                    success, cam2end_matrix, target2base_matrix, rms_error, per_image_errors = self._perform_single_calibration(test_method, verbose=False)
                    
                    if success and rms_error < best_rms_error:
                        best_method = test_method
                        best_method_name = method_name
                        best_rms_error = rms_error
                        best_cam2end = cam2end_matrix.copy()
                        best_target2base = target2base_matrix.copy()
                        best_per_image_errors = per_image_errors.copy()
                        
                        if verbose and len(methods_to_try) > 1:
                            print(f"   ✅ New best method: {method_name} with RMS error {rms_error:.4f}")
                    elif success:
                        if verbose and len(methods_to_try) > 1:
                            print(f"   ✅ Method {method_name} succeeded with RMS error {rms_error:.4f}")
                    else:
                        if verbose:
                            if len(methods_to_try) > 1:
                                print(f"   ❌ Method {method_name} failed")
                            else:
                                print(f"❌ Eye-in-hand calibration failed with method {method_name}")
                            
                except Exception as e:
                    if verbose:
                        if len(methods_to_try) > 1:
                            print(f"   ❌ Method {method_name} failed with error: {e}")
                        else:
                            print(f"❌ Eye-in-hand calibration failed with method {method_name}: {e}")
                    continue
            
            # return none if best_method is not found
            if best_method is None:
                if verbose:
                    if len(methods_to_try) > 1:
                        print("❌ All calibration methods failed")
                    # Single method failure message already printed above
                return None

            if verbose:
                if len(methods_to_try) > 1:
                    print(f"\n🎉 Best method selected: {best_method_name} with RMS error {best_rms_error:.4f}")
                else:
                    print(f"✅ Eye-in-hand calibration completed successfully!")
                    print(f"RMS reprojection error: {best_rms_error:.4f} pixels")
                print(f"Camera to end-effector transformation matrix:")
                print(f"{best_cam2end}")

            # Store the best results
            self.cam2end_matrix = best_cam2end
            self.target2base_matrix = best_target2base
            self.rms_error = best_rms_error
            self.per_image_errors = best_per_image_errors

            # Store initial calibration results for optimization comparison
            initial_results = {
                'cam2end_matrix': self.cam2end_matrix.copy(),
                'target2base_matrix': self.target2base_matrix.copy(),
                'rms_error': self.rms_error,
            }
            
            # Attempt optimization if nlopt is available
            optimized_results = initial_results.copy()
            if HAS_NLOPT:
                if verbose:
                    print(f"\n🔧 Attempting optimization...")
                
                try:
                    # Perform optimization
                    initial_error, optimized_rms = self.optimize_calibration(ftol_rel=1e-6, verbose=verbose)
                    
                    # Check if optimization actually improved the error
                    improvement = initial_error - optimized_rms
                    improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
                    
                    if optimized_rms < initial_error:
                        # Optimization improved - store optimized results
                        optimized_results.update({
                            'cam2end_matrix': self.cam2end_matrix.copy(),
                            'target2base_matrix': self.target2base_matrix.copy(),
                            'rms_error': optimized_rms,
                        })
                        
                        if verbose:
                            print(f"✅ Optimization completed!")
                            print(f"   Initial RMS error: {initial_error:.4f} pixels")
                            print(f"   Optimized RMS error: {optimized_rms:.4f} pixels") 
                            print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                    else:
                        # Optimization didn't improve or made it worse - keep initial results
                        if verbose:
                            print(f"⚠️ Optimization did not improve results")
                            print(f"   Initial RMS error: {initial_results['rms_error']:.4f} pixels")
                            print(f"   Optimized RMS error: {optimized_rms:.4f} pixels")
                            print(f"   Keeping initial calibration results")
                        
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Optimization failed: {e}")
                        print(f"   Returning initial calibration results")
            else:
                if verbose:
                    print(f"⚠️ nlopt not available, skipping optimization")
            
            # Add both initial and optimized results to the return dictionary
            optimized_results['before_opt'] = initial_results
            
            # Set calibration completed flag only after optimization is complete
            self.calibration_completed = True
            
            # Return optimized calibration results as dictionary
            return optimized_results
                
        except Exception as e:
            if verbose:
                print(f"❌ Eye-in-hand calibration failed: {e}")
            self.calibration_completed = False
            return None

    def to_json(self) -> dict:
        """
        Serialize eye-in-hand calibrator state to JSON-compatible dictionary.
        
        Extends HandEyeBaseCalibrator.to_json() to include eye-in-hand specific data:
        - cam2end_matrix: Camera to end-effector transformation matrix
        - target2base_matrix: Target to base transformation matrix
        
        Returns:
            dict: JSON-compatible dictionary containing complete calibrator state
        """
        # Get base class data (HandEyeBaseCalibrator -> BaseCalibrator)
        data = super().to_json()
        
        # Add eye-in-hand specific data
        if self.cam2end_matrix is not None:
            data['cam2end_matrix'] = self.cam2end_matrix.tolist()
            
        if self.target2base_matrix is not None:
            data['target2base_matrix'] = self.target2base_matrix.tolist()
        
        # Add calibration type identifier
        data['calibration_type'] = 'eye_in_hand'
        
        return data

    def from_json(self, data: dict) -> None:
        """
        Deserialize eye-in-hand calibrator state from JSON-compatible dictionary.
        
        Extends HandEyeBaseCalibrator.from_json() to load eye-in-hand specific data:
        - cam2end_matrix: Camera to end-effector transformation matrix
        - target2base_matrix: Target to base transformation matrix
        
        Args:
            data: JSON-compatible dictionary containing calibrator state
        """
        # Load base class data first (HandEyeBaseCalibrator -> BaseCalibrator)
        super().from_json(data)
        
        # Load eye-in-hand specific data
        if 'cam2end_matrix' in data:
            self.cam2end_matrix = np.array(data['cam2end_matrix'], dtype=np.float32)
            
        if 'target2base_matrix' in data:
            self.target2base_matrix = np.array(data['target2base_matrix'], dtype=np.float32)

    # ============================================================================
    # Eye-in-Hand Specific IO Methods
    # ============================================================================
    
    def set_cam2end_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the camera to end-effector transformation matrix.
        
        Args:
            matrix: 4x4 transformation matrix from camera to end-effector
            
        Raises:
            ValueError: If matrix has invalid format
        """
        if matrix is None:
            self.cam2end_matrix = None
            return
            
        if not isinstance(matrix, np.ndarray):
            raise ValueError("cam2end_matrix must be a numpy array")
        
        if matrix.shape != (4, 4):
            raise ValueError(f"cam2end_matrix must be 4x4, got shape {matrix.shape}")
            
        self.cam2end_matrix = matrix.copy()
    
    def set_target2base_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the target to base transformation matrix.
        
        Args:
            matrix: 4x4 transformation matrix from target to base
            
        Raises:
            ValueError: If matrix has invalid format
        """
        if matrix is None:
            self.target2base_matrix = None
            return
            
        if not isinstance(matrix, np.ndarray):
            raise ValueError("target2base_matrix must be a numpy array")
        
        if matrix.shape != (4, 4):
            raise ValueError(f"target2base_matrix must be 4x4, got shape {matrix.shape}")
            
        self.target2base_matrix = matrix.copy()
    
    def get_cam2end_matrix(self) -> Optional[np.ndarray]:
        """
        Get the camera to end-effector transformation matrix.
        
        Returns:
            np.ndarray or None: 4x4 transformation matrix if available
        """
        return self.cam2end_matrix
    
    def get_target2base_matrix(self) -> Optional[np.ndarray]:
        """
        Get the target to base transformation matrix.
        
        Returns:
            np.ndarray or None: 4x4 transformation matrix if available
        """
        return self.target2base_matrix
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the main transformation matrix (cam2end_matrix for eye-in-hand).
        
        Returns:
            np.ndarray or None: Camera to end-effector transformation matrix
        """
        return self.cam2end_matrix
    
    def get_reproject_rvec_tvec(self) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Get rotation and translation vectors for reprojection visualization from robot kinematic chain.
        
        For eye-in-hand calibration, calculates rvec and tvec for each image from the robot
        transformation chain: target2cam = inv(cam2end) @ inv(end2base) @ target2base
        
        Returns:
            Tuple containing:
            - List of rotation vectors (one per image, None for failed calculations)
            - List of translation vectors (one per image, None for failed calculations)
            
        Raises:
            ValueError: If hand-eye calibration not completed or required data missing
        """
        # Check if hand-eye calibration is completed
        if not self.is_calibrated() or self.cam2end_matrix is None or self.target2base_matrix is None:
            raise ValueError("Hand-eye calibration not completed. Run calibrate() first.")
        
        # Check if we have the required data
        if not self.end2base_matrices:
            raise ValueError("Robot end-effector poses not available. Set end2base_matrices first.")
        
        rvecs = []
        tvecs = []
        
        for i in range(len(self.images)):
            if (self.end2base_matrices[i] is not None and 
                self.image_points[i] is not None and 
                self.object_points[i] is not None):
                
                try:
                    # Eye-in-hand transformation chain: target2cam = inv(cam2end) @ inv(end2base) @ target2base
                    end2cam_matrix = np.linalg.inv(self.cam2end_matrix)
                    base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                    
                    # Calculate pattern2camera transformation for this pose
                    pattern2cam_matrix = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
                    
                    # Extract rotation matrix and translation vector
                    rotation_matrix = pattern2cam_matrix[:3, :3]
                    translation_vector = pattern2cam_matrix[:3, 3]
                    
                    # Convert rotation matrix to rotation vector
                    rvec, _ = cv2.Rodrigues(rotation_matrix)
                    tvec = translation_vector.reshape(-1, 1)
                    
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    
                except Exception:
                    # Failed to calculate transformation for this image
                    rvecs.append(None)
                    tvecs.append(None)
            else:
                # Missing required data for this image
                rvecs.append(None)
                tvecs.append(None)
        
        return rvecs, tvecs

    # ============================================================================
    # Validation Methods
    # ============================================================================
    
    def validate_eye_in_hand_data(self) -> bool:
        """
        Validate that all required data for eye-in-hand calibration is available.
        
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
            
        print("✅ All required data for eye-in-hand calibration is available")
        return True
    
    def is_eye_in_hand_calibrated(self) -> bool:
        """
        Check if eye-in-hand calibration has been completed successfully.
        
        Returns:
            bool: True if eye-in-hand calibration is complete
        """
        return (self.is_calibrated() and 
                self.cam2end_matrix is not None)

    def _calculate_optimal_target2base_matrix(self, cam2end_4x4: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate the single target2base matrix that minimizes overall reprojection error.
        
        This method finds the target2base transformation that best explains all the
        observed target positions across all images, rather than calculating a
        separate target2base for each image which would result in zero error.
        
        Args:
            cam2end_4x4: The camera to end-effector transformation matrix from calibration
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to base transformation matrix
        """
        if verbose:
            print("Calculating optimal target2base matrix...")
        
        best_error = float('inf')
        best_target2base = None
        
        # Try using each image's target2cam transformation to estimate target2base
        # Then find the one that gives the smallest overall reprojection error
        candidate_target2base_matrices = []
        
        for i in range(len(self.target2cam_matrices)):
            if self.target2cam_matrices[i] is not None:
                # Calculate target2base using this image's measurements
                candidate_target2base = self.end2base_matrices[i] @ cam2end_4x4 @ self.target2cam_matrices[i]
                candidate_target2base_matrices.append(candidate_target2base)
        
        # Test each candidate to find the one with minimum overall reprojection error
        for candidate_idx, candidate_target2base in enumerate(candidate_target2base_matrices):
            # Use the separate reprojection error function for each candidate
            rms_error, _ = self._calculate_reprojection_errors(cam2end_4x4, candidate_target2base, verbose=False)
            
            if rms_error < best_error:
                best_error = rms_error
                best_target2base = candidate_target2base.copy()
                if verbose:
                    print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f} (best so far)")
            elif verbose:
                print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f}")
        
        if best_target2base is not None:
            if verbose:
                print(f"✅ Optimal target2base matrix found with RMS error: {best_error:.4f}")
                print("Target2base transformation matrix:")
                print(best_target2base)
        else:
            if verbose:
                print("⚠️ Could not find optimal target2base matrix, using first candidate")
            best_target2base = candidate_target2base_matrices[0] if candidate_target2base_matrices else np.eye(4)
        
        return best_target2base

    def calculate_reprojection_errors(self, cam2end_matrix: Optional[np.ndarray] = None, target2base_matrix: Optional[np.ndarray] = None, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Calculate reprojection errors using hand-eye calibration results.
        
        This is a public method that can be used to calculate reprojection errors
        for given transformation matrices or the stored calibration results.
        
        Args:
            cam2end_matrix: 4x4 camera-to-end-effector transformation matrix. 
                          If None, uses self.cam2end_matrix
            target2base_matrix: 4x4 target-to-base transformation matrix.
                              If None, uses self.target2base_matrix
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple of (rms_error, per_image_errors):
            - rms_error: Overall RMS reprojection error across all valid images
            - per_image_errors: List of reprojection errors for each image (inf for invalid images)
            
        Raises:
            ValueError: If required matrices or data are not available
        """
        # Use provided matrices or stored calibration results
        if cam2end_matrix is None:
            if self.cam2end_matrix is None:
                raise ValueError("No cam2end_matrix provided and no calibration results stored")
            cam2end_matrix = self.cam2end_matrix
            
        if target2base_matrix is None:
            if self.target2base_matrix is None:
                raise ValueError("No target2base_matrix provided and no calibration results stored")
            target2base_matrix = self.target2base_matrix
        
        # Validate prerequisites for reprojection error calculation
        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not detected. Run detect_pattern_points() first.")
            
        if self.end2base_matrices is None:
            raise ValueError("Robot poses not loaded")
            
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters not available")
        
        return self._calculate_reprojection_errors(cam2end_matrix, target2base_matrix, verbose)

    def _calculate_reprojection_errors(self, cam2end_matrix: np.ndarray, target2base_matrix: np.ndarray, verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Calculate reprojection errors using hand-eye calibration results.
        
        This method projects 3D calibration points to 2D image points using the calibrated
        hand-eye transformation and compares with detected pattern points to compute
        reprojection errors.
        
        Args:
            cam2end_matrix: 4x4 camera-to-end-effector transformation matrix
            target2base_matrix: 4x4 target-to-base transformation matrix
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
                    end2cam_matrix = np.linalg.inv(cam2end_matrix)
                    base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                    
                    # Target to camera using hand-eye calibration and the target2base matrix
                    eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ target2base_matrix
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyeinhand_target2cam[:3, :3], 
                        eyeinhand_target2cam[:3, 3], 
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
            Tuple of (success, cam2end_matrix, target2base_matrix, rms_error, per_image_errors)
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
            end2base_Rs = np.array([self.end2base_matrices[i][:3, :3] for i in valid_indices])
            end2base_ts = np.array([self.end2base_matrices[i][:3, 3] for i in valid_indices])
            
            # Use rvecs and tvecs from target2cam matrices
            target2cam_Rs = np.array([self.target2cam_matrices[i][:3, :3] for i in valid_indices])
            target2cam_ts = np.array([self.target2cam_matrices[i][:3, 3] for i in valid_indices])
            
            # Convert rotation matrices to rotation vectors
            rvecs_array = np.array([cv2.Rodrigues(R)[0] for R in target2cam_Rs])
            tvecs_array = target2cam_ts.reshape(-1, 3, 1)
            
            # Perform eye-in-hand calibration using OpenCV
            cam2end_R, cam2end_t = cv2.calibrateHandEye(
                end2base_Rs, end2base_ts, rvecs_array, tvecs_array, method)
            
            # Create 4x4 transformation matrix
            cam2end_4x4 = np.eye(4)
            cam2end_4x4[:3, :3] = cam2end_R
            cam2end_4x4[:3, 3] = cam2end_t[:, 0]
            
            # Calculate the optimal target2base matrix
            target2base_matrix = self._calculate_optimal_target2base_matrix(cam2end_4x4, verbose)
            
            # Calculate reprojection errors using the separate function
            rms_error, per_image_errors = self._calculate_reprojection_errors(cam2end_4x4, target2base_matrix, verbose)
            
            return True, cam2end_4x4, target2base_matrix, rms_error, per_image_errors
            
        except Exception as e:
            if verbose:
                print(f"   Calibration failed: {e}")
            return False, None, None, float('inf'), None

    def optimize_calibration(self, ftol_rel: float = 1e-6, verbose: bool = False) -> Tuple[float, float]:
        """
        Optimize calibration results by jointly refining cam2end and target2base matrices.
        
        This method uses nonlinear optimization to minimize reprojection error by
        simultaneously optimizing both the camera-to-end-effector transformation
        and the target-to-base transformation.
        
        Args:
            ftol_rel: Relative tolerance for convergence
            verbose: Whether to print optimization progress
            
        Returns:
            Tuple[float, float]: (initial_error, final_error) - RMS reprojection errors before and after optimization
            
        Raises:
            ValueError: If calibration has not been completed
            ImportError: If nlopt optimization library is not available
        """
        if not hasattr(self, 'cam2end_matrix') or self.cam2end_matrix is None:
            raise ValueError("Initial calibration must be completed before optimization. Call calibrate() first.")
            
        if not HAS_NLOPT:
            raise ImportError("nlopt library is required for optimization but not available")
            
        if verbose:
            print(f"Starting optimization...")
            print(f"Initial RMS error: {self.rms_error:.4f} pixels")
            
        # Store initial values
        initial_cam2end = self.cam2end_matrix.copy()
        initial_target2base = self.target2base_matrix.copy()
        initial_error = self.rms_error
        
        try:
            # Perform joint optimization
            optimized_cam2end, optimized_target2base, initial_opt_error, final_opt_error = self._optimize_matrices_jointly(
                initial_cam2end, initial_target2base, ftol_rel, verbose
            )

            # record the optimization if improved
            if initial_opt_error > final_opt_error:
                # Update matrices with optimized results
                self.cam2end_matrix = optimized_cam2end
                self.target2base_matrix = optimized_target2base
                                
                # Recalculate errors with optimized matrices
                self.rms_error, self.per_image_errors = self.calculate_reprojection_errors(
                    self.cam2end_matrix, self.target2base_matrix, verbose=False
                )
                
                if verbose:
                    improvement = initial_error - self.rms_error
                    improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
                    print(f"Optimization completed!")
                    print(f"Final RMS error: {self.rms_error:.4f} pixels")
                    print(f"Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                
            return initial_error, self.rms_error
            
        except Exception as e:
            if verbose:
                print(f"Optimization failed: {e}")
            # Restore original matrices
            self.cam2end_matrix = initial_cam2end
            self.target2base_matrix = initial_target2base
            self.rms_error = initial_error
            return initial_error, initial_error

    def _optimize_matrices_jointly(self, initial_cam2end, initial_target2base, ftol_rel, verbose):
        """
        Optimize both cam2end and target2base matrices simultaneously.
        This should converge faster than iterative optimization.
        
        Args:
            initial_cam2end: Initial camera-to-end-effector transformation matrix
            initial_target2base: Initial target-to-base transformation matrix  
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print optimization progress
            
        Returns:
            tuple: (optimized_cam2end, optimized_target2base, initial_error, final_error)
        """
        try:
            import nlopt
            
            # Convert initial matrices to parameter vectors
            cam2end_params = matrix_to_xyz_rpy(initial_cam2end)  # [x, y, z, roll, pitch, yaw]
            target2base_params = matrix_to_xyz_rpy(initial_target2base)  # [x, y, z, roll, pitch, yaw]
            
            # Combined parameter vector: [cam2end_params, target2base_params] (12 total)
            initial_params = np.concatenate([cam2end_params, target2base_params])
            
            # Setup joint optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 12)  # 12 parameters total
            
            def joint_objective(params, grad):
                """Objective function that optimizes both matrices simultaneously."""
                # Split parameters back into two matrices
                cam2end_params = params[:6]  # First 6 parameters for cam2end
                target2base_params = params[6:]  # Last 6 parameters for target2base
                
                # Convert to matrices
                cam2end_matrix = xyz_rpy_to_matrix(cam2end_params)
                target2base_matrix = xyz_rpy_to_matrix(target2base_params)
                
                # Calculate error using both matrices
                rms_error, _ = self.calculate_reprojection_errors(cam2end_matrix, target2base_matrix, verbose=False)
                return rms_error
            
            opt.set_min_objective(joint_objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize both matrices jointly
            try:
                optimized_params = opt.optimize(initial_params)
                
                # Split optimized parameters back into matrices
                optimized_cam2end = xyz_rpy_to_matrix(optimized_params[:6])
                optimized_target2base = xyz_rpy_to_matrix(optimized_params[6:])
                
                # Check if optimization actually improved the result
                initial_error = joint_objective(initial_params, None)
                final_error = joint_objective(optimized_params, None)
                
                if final_error < initial_error:
                    # Optimization improved - return optimized matrices
                    if verbose:
                        print(f"   Joint optimization: {initial_error:.4f} -> {final_error:.4f} pixels")
                        improvement = (initial_error - final_error) / initial_error * 100
                        print(f"   Improvement: {improvement:.1f}%")
                    return optimized_cam2end, optimized_target2base, initial_error, final_error
                else:
                    # Optimization didn't improve or made it worse - return initial matrices
                    if verbose:
                        print(f"   Joint optimization did not improve: {initial_error:.4f} -> {final_error:.4f} pixels")
                        print(f"   Keeping initial matrices")
                    return initial_cam2end, initial_target2base, initial_error, initial_error  # final_error = initial_error
                
            except Exception as opt_e:
                if verbose:
                    print(f"   Joint optimization failed: {opt_e}")
                # Calculate initial error for return value
                initial_error = joint_objective(initial_params, None)
                return initial_cam2end, initial_target2base, initial_error, initial_error
                
        except ImportError:
            if verbose:
                print("   nlopt not available, skipping joint optimization")
            # Calculate initial error for return value (need to handle case where calculate_reprojection_errors might not work)
            try:
                initial_error, _ = self.calculate_reprojection_errors(initial_cam2end, initial_target2base, verbose=False)
            except:
                initial_error = float('inf')  # Fallback if calculation fails
            return initial_cam2end, initial_target2base, initial_error, initial_error

