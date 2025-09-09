"""
Eye-to-Hand Calibration Module
==============================

This module provides the EyeToHandCalibrator class that inherits from HandEyeBaseCalibrator.
It follows the same interface and structure as EyeInHandCalibrator but implements 
eye-to-hand specific matrix calculations.

Key Features:
- Inherits all common functionality from HandEyeBaseCalibrator
- Provides eye-to-hand specific data structures
- Complete calibration functionality with all OpenCV methods
- Automatic method selection with error comparison

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/IO)
       -> EyeToHandCalibrator (eye-to-hand specific calibration)
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


class EyeToHandCalibrator(HandEyeBaseCalibrator):
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
                 distortion_coefficients: Optional[np.ndarray] = None,
                 verbose: bool = False):
        """
        Initialize EyeToHandCalibrator.
        
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
            - 'before_opt': Dict - Initial calibration results before optimization
            
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
                print(f"🤖 Running eye-to-hand calibration with {valid_images} image-pose pairs")
            
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
            best_base2cam = None
            best_target2end = None
            best_per_image_errors = None
            
            for test_method, method_name in methods_to_try.items():
                if verbose and len(methods_to_try) > 1:
                    print(f"\n🧪 Testing method: {method_name} ({test_method})")
                
                try:
                    # Perform calibration with this method
                    success, base2cam_matrix, target2end_matrix, rms_error, per_image_errors = self._perform_single_calibration(test_method, verbose=False)
                    
                    if success and rms_error < best_rms_error:
                        best_method = test_method
                        best_method_name = method_name
                        best_rms_error = rms_error
                        best_base2cam = base2cam_matrix.copy()
                        best_target2end = target2end_matrix.copy()
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
                                print(f"❌ Eye-to-hand calibration failed with method {method_name}")
                            
                except Exception as e:
                    if verbose:
                        if len(methods_to_try) > 1:
                            print(f"   ❌ Method {method_name} failed with error: {e}")
                        else:
                            print(f"❌ Eye-to-hand calibration failed with method {method_name}: {e}")
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
                    print(f"✅ Eye-to-hand calibration completed successfully!")
                    print(f"RMS reprojection error: {best_rms_error:.4f} pixels")
                print(f"Base to camera transformation matrix:")
                print(f"{best_base2cam}")

            # Store the best results
            self.base2cam_matrix = best_base2cam
            self.target2end_matrix = best_target2end
            self.rms_error = best_rms_error
            self.per_image_errors = best_per_image_errors

            # Store initial calibration results for optimization comparison
            initial_results = {
                'base2cam_matrix': self.base2cam_matrix.copy(),
                'target2end_matrix': self.target2end_matrix.copy(),
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
                            'base2cam_matrix': self.base2cam_matrix.copy(),
                            'target2end_matrix': self.target2end_matrix.copy(),
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
                print(f"❌ Eye-to-hand calibration failed: {e}")
            self.calibration_completed = False
            return None

    def to_json(self) -> dict:
        """
        Serialize eye-to-hand calibrator state to JSON-compatible dictionary.
        
        Extends HandEyeBaseCalibrator.to_json() to include eye-to-hand specific data:
        - base2cam_matrix: Robot base to camera transformation matrix
        - target2end_matrix: Target to end-effector transformation matrix
        
        Returns:
            dict: JSON-compatible dictionary containing complete calibrator state
        """
        # Get base class data (HandEyeBaseCalibrator -> BaseCalibrator)
        data = super().to_json()
        
        # Add eye-to-hand specific data
        if self.base2cam_matrix is not None:
            data['base2cam_matrix'] = self.base2cam_matrix.tolist()
            
        if self.target2end_matrix is not None:
            data['target2end_matrix'] = self.target2end_matrix.tolist()
        
        # Add calibration type identifier
        data['calibration_type'] = 'eye_to_hand'
        
        return data

    def from_json(self, data: dict) -> None:
        """
        Deserialize eye-to-hand calibrator state from JSON-compatible dictionary.
        
        Extends HandEyeBaseCalibrator.from_json() to load eye-to-hand specific data:
        - base2cam_matrix: Robot base to camera transformation matrix
        - target2end_matrix: Target to end-effector transformation matrix
        
        Args:
            data: JSON-compatible dictionary containing calibrator state
        """
        # Load base class data first (HandEyeBaseCalibrator -> BaseCalibrator)
        super().from_json(data)
        
        # Load eye-to-hand specific data
        if 'base2cam_matrix' in data:
            self.base2cam_matrix = np.array(data['base2cam_matrix'], dtype=np.float32)
            
        if 'target2end_matrix' in data:
            self.target2end_matrix = np.array(data['target2end_matrix'], dtype=np.float32)

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
    
    def get_reproject_rvec_tvec(self) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Get rotation and translation vectors for reprojection visualization from robot kinematic chain.
        
        For eye-to-hand calibration, calculates rvec and tvec for each image from the robot
        transformation chain: target2cam = base2cam @ end2base @ target2end
        
        Returns:
            Tuple containing:
            - List of rotation vectors (one per image, None for failed calculations)
            - List of translation vectors (one per image, None for failed calculations)
            
        Raises:
            ValueError: If hand-eye calibration not completed or required data missing
        """
        # Check if hand-eye calibration is completed
        if not self.is_calibrated() or self.base2cam_matrix is None or self.target2end_matrix is None:
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
                    # Eye-to-hand transformation chain: target2cam = base2cam @ end2base @ target2end
                    pattern2cam_matrix = self.base2cam_matrix @ self.end2base_matrices[i] @ self.target2end_matrix
                    
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

    def optimize_calibration(self, ftol_rel: float = 1e-6, verbose: bool = False) -> Tuple[float, float]:
        """
        Optimize calibration results by jointly refining base2cam and target2end matrices.
        
        This method uses nonlinear optimization to minimize reprojection error by
        simultaneously optimizing both the robot-base-to-camera transformation
        and the target-to-end-effector transformation.
        
        Args:
            ftol_rel: Relative tolerance for convergence
            verbose: Whether to print optimization progress
            
        Returns:
            Tuple[float, float]: (initial_error, final_error) - RMS reprojection errors before and after optimization
            
        Raises:
            ValueError: If calibration has not been completed
            ImportError: If nlopt optimization library is not available
        """
        if not hasattr(self, 'base2cam_matrix') or self.base2cam_matrix is None:
            raise ValueError("Initial calibration must be completed before optimization. Call calibrate() first.")
            
        if not HAS_NLOPT:
            if verbose:
                print("⚠️ nlopt library not available, skipping optimization")
            return self.rms_error, self.rms_error
            
        if verbose:
            print(f"🔧 Starting eye-to-hand optimization...")
            print(f"   Initial RMS error: {self.rms_error:.4f} pixels")
            
        # Store initial values
        initial_base2cam = self.base2cam_matrix.copy()
        initial_target2end = self.target2end_matrix.copy()
        initial_error = self.rms_error
        
        try:
            # Two-stage joint optimization: first optimize secondary matrix only, then both
            if verbose:
                print("   Two-stage optimization approach:")
                print("   Stage 1: Optimizing secondary matrix (target2end) only")
            
            # Stage 1: Optimize only target2end (secondary matrix), fix base2cam (primary matrix)
            base2cam_stage1, target2end_stage1, error_before_stage1, error_after_stage1 = self._optimize_matrices_jointly(
                initial_base2cam, initial_target2end, ftol_rel, verbose, fix_primary_matrix=True
            )
            
            # Stage 2: Optimize both matrices using stage 1 results as starting point
            if verbose:
                print("   Stage 2: Optimizing both matrices jointly")
            optimized_base2cam, optimized_target2end, error_before_stage2, error_after_stage2 = self._optimize_matrices_jointly(
                base2cam_stage1, target2end_stage1, ftol_rel, verbose, fix_primary_matrix=False
            )
            
            if verbose:
                print(f"   Overall two-stage optimization: {error_before_stage1:.4f} -> {error_after_stage2:.4f} pixels")
                overall_improvement = (error_before_stage1 - error_after_stage2) / error_before_stage1 * 100
                print(f"   Overall improvement: {overall_improvement:.1f}%")
            
            # Update matrices with optimized results
            self.base2cam_matrix = optimized_base2cam
            self.target2end_matrix = optimized_target2end
            
            # Recalculate errors with optimized matrices
            final_error, _ = self._calculate_reprojection_errors(optimized_base2cam, optimized_target2end, verbose=False)
            self.rms_error = final_error
            
            if verbose:
                improvement = initial_error - final_error
                improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
                print(f"   Final RMS error: {final_error:.4f} pixels")
                print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                
            return initial_error, final_error
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Eye-to-hand optimization failed: {e}")
            # Restore original matrices
            self.base2cam_matrix = initial_base2cam
            self.target2end_matrix = initial_target2end
            self.rms_error = initial_error
            return initial_error, initial_error

    def _optimize_matrices_jointly(self, initial_base2cam, initial_target2end, ftol_rel, verbose, fix_primary_matrix=False):
        """
        Optimize both base2cam and target2end matrices simultaneously using delta transformations.
        This should converge faster than iterative optimization.
        
        Args:
            initial_base2cam: Initial base-to-camera transformation matrix
            initial_target2end: Initial target-to-end-effector transformation matrix  
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print optimization progress
            fix_primary_matrix: If True, only optimize target2end matrix (keep base2cam fixed)
            
        Returns:
            tuple: (optimized_base2cam, optimized_target2end, initial_error, final_error)
        """
        try:
            import nlopt
            
            # Define matrix configuration based on optimization mode
            # matrices: [initial_matrices, optimize_flags, matrix_names]
            initial_matrices = [initial_base2cam, initial_target2end]
            matrix_names = ["base2cam", "target2end"]
            
            if fix_primary_matrix:
                # Only optimize secondary matrix (target2end)
                optimize_flags = [False, True]  # [base2cam_fixed, target2end_optimized]
                opt_description = "target2end matrix (base2cam fixed)"
            else:
                # Optimize both matrices
                optimize_flags = [True, True]   # [base2cam_optimized, target2end_optimized]
                opt_description = "both base2cam and target2end matrices"
            
            # Calculate parameter count and create index mapping
            optimize_indices = [i for i, flag in enumerate(optimize_flags) if flag]
            param_count = len(optimize_indices) * 6  # 6 parameters per matrix (xyz + rpy)
            initial_delta_params = np.zeros(param_count)
            
            if verbose:
                print(f"   Optimizing {opt_description}")
            
            # Setup joint optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, param_count)
            
            def joint_objective(delta_params, grad):
                """Unified objective function using index-based parameter mapping."""
                try:
                    # Initialize result matrices with originals
                    result_matrices = [matrix.copy() for matrix in initial_matrices]
                    
                    # Apply delta transformations only to matrices being optimized
                    param_offset = 0
                    for i, should_optimize in enumerate(optimize_flags):
                        if should_optimize:
                            # Extract 6 parameters for this matrix
                            matrix_delta_params = delta_params[param_offset:param_offset + 6]
                            delta_matrix = xyz_rpy_to_matrix(matrix_delta_params)
                            
                            # Apply delta transformation: optimized = original @ delta
                            result_matrices[i] = initial_matrices[i] @ delta_matrix
                            param_offset += 6
                    
                    # Calculate error using the result matrices
                    rms_error, _ = self.calculate_reprojection_errors(result_matrices[0], result_matrices[1], verbose=False)
                    
                    # Validate result
                    if not np.isfinite(rms_error):
                        return 1e6  # Large penalty for invalid error
                        
                    return rms_error
                except Exception:
                    return 1e6  # Large penalty for any computation errors
            
            opt.set_min_objective(joint_objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Set bounds for delta parameters (small perturbations)
            # Translation deltas: ±0.1 meters, rotation deltas: ±0.2 radians (~11 degrees)
            single_matrix_bounds_low = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2])
            single_matrix_bounds_high = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
            
            # Create bounds array based on number of matrices being optimized
            delta_bounds_low = np.tile(single_matrix_bounds_low, len(optimize_indices))
            delta_bounds_high = np.tile(single_matrix_bounds_high, len(optimize_indices))
            
            opt.set_lower_bounds(delta_bounds_low)
            opt.set_upper_bounds(delta_bounds_high)
            
            # Optimize delta parameters
            try:
                optimized_delta_params = opt.optimize(initial_delta_params)
                
                # Apply optimized deltas to get final matrices using the same unified approach
                result_matrices = [matrix.copy() for matrix in initial_matrices]
                param_offset = 0
                for i, should_optimize in enumerate(optimize_flags):
                    if should_optimize:
                        matrix_delta_params = optimized_delta_params[param_offset:param_offset + 6]
                        delta_matrix = xyz_rpy_to_matrix(matrix_delta_params)
                        result_matrices[i] = initial_matrices[i] @ delta_matrix
                        param_offset += 6
                
                # Check if optimization actually improved the result
                initial_error = joint_objective(initial_delta_params, None)
                final_error = joint_objective(optimized_delta_params, None)
                
                if final_error < initial_error:
                    # Optimization improved - return optimized matrices
                    if verbose:
                        opt_type = "Secondary-only" if fix_primary_matrix else "Joint"
                        print(f"   {opt_type} delta optimization: {initial_error:.4f} -> {final_error:.4f} pixels")
                        improvement = (initial_error - final_error) / initial_error * 100
                        print(f"   Improvement: {improvement:.1f}%")
                        
                        # Print delta information for optimized matrices
                        param_offset = 0
                        delta_info = []
                        for i, should_optimize in enumerate(optimize_flags):
                            if should_optimize:
                                matrix_delta_params = optimized_delta_params[param_offset:param_offset + 6]
                                trans_delta = np.max(np.abs(matrix_delta_params[:3]))
                                rot_delta = np.max(np.abs(matrix_delta_params[3:]))
                                delta_info.append(f"{matrix_names[i]}: translation {trans_delta:.4f}m, rotation {rot_delta:.4f}rad")
                                param_offset += 6
                        
                        if len(delta_info) == 1:
                            print(f"   Max {delta_info[0]}")
                        else:
                            print(f"   Max deltas - {' | '.join(delta_info)}")
                            
                    return result_matrices[0], result_matrices[1], initial_error, final_error
                else:
                    # Optimization didn't improve or made it worse - return initial matrices
                    if verbose:
                        opt_type = "Secondary-only" if fix_primary_matrix else "Joint"
                        print(f"   {opt_type} delta optimization did not improve: {initial_error:.4f} -> {final_error:.4f} pixels")
                        print(f"   Keeping initial matrices")
                    return initial_base2cam, initial_target2end, initial_error, initial_error
                
            except Exception as opt_e:
                if verbose:
                    print(f"   Delta optimization failed: {opt_e}")
                # Calculate initial error for return value
                initial_error = joint_objective(initial_delta_params, None)
                return initial_base2cam, initial_target2end, initial_error, initial_error
                
        except ImportError:
            if verbose:
                print("   nlopt not available, skipping delta optimization")
            # Calculate initial error for return value
            def joint_objective_fallback(delta_params, grad):
                rms_error, _ = self.calculate_reprojection_errors(initial_base2cam, initial_target2end, verbose=False)
                return rms_error
            initial_error = joint_objective_fallback(np.zeros(6), None)
            return initial_base2cam, initial_target2end, initial_error, initial_error
