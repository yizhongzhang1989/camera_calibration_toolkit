"""
New Eye-to-Hand Calibration Module
==================================

This module provides the new EyeToHandCalibrator class that inherits from HandEyeBaseCalibrator.
This implementation validates the new base class architecture before refactoring
the existing EyeToHandCalibrator.

Key Features:
- Inherits all common functionality from HandEyeBaseCalibrator
- Implements eye-to-hand specific coordinate system transformations
- Uses base2end transformations for calibration (inverted from end2base)
- Provides base2cam_matrix as the primary result
- Maintains API compatibility with existing EyeToHandCalibrator

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/workflow)
       -> NewEyeToHandCalibrator (eye-to-hand specific logic)

Key Difference from Eye-in-Hand:
-------------------------------
- Eye-in-hand: cv2.calibrateHandEye(end2base_Rs, end2base_ts, target2cam_Rs, target2cam_ts) -> cam2end
- Eye-to-hand: cv2.calibrateHandEye(base2end_Rs, base2end_ts, target2cam_Rs, target2cam_ts) -> base2cam
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any, Union

# Optional import for optimization
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    nlopt = None
    HAS_NLOPT = False

from .hand_eye_base_calibration import HandEyeBaseCalibrator
from .calibration_patterns import CalibrationPattern
from .utils import xyz_rpy_to_matrix, matrix_to_xyz_rpy, inverse_transform_matrix


class NewEyeToHandCalibrator(HandEyeBaseCalibrator):
    """
    New Eye-to-Hand Calibration class inheriting from HandEyeBaseCalibrator.
    
    This class calibrates the transformation between a stationary camera and the robot
    base coordinate frame. The target/calibration pattern is mounted on the robot end-effector.
    
    Key Differences from Eye-in-Hand:
    - Uses base2end transformations (inverted from end2base)
    - Calibrates base2cam_matrix (instead of cam2end_matrix)
    - Target is directly attached to end-effector
    
    Transformation Chain for Eye-to-Hand:
    target2cam = base2cam * base2end * end2target
    where end2target is the inverse of target2end
    """
    
    def __init__(self, image_paths: Optional[List[str]] = None, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None,
                 calibration_pattern: Optional[CalibrationPattern] = None):
        """
        Initialize NewEyeToHandCalibrator with base class architecture.
        
        Args:
            image_paths: List of image file paths (auto-loads corresponding JSON pose files)
            images: List of image arrays (numpy arrays)
            end2base_matrices: List of robot poses (4x4 transformation matrices)
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coefficients: Distortion coefficients array (5 or more elements)
            calibration_pattern: CalibrationPattern instance for pattern detection
        """
        # Initialize parent class with common functionality
        super().__init__(
            images=images,
            end2base_matrices=end2base_matrices,
            image_paths=image_paths,
            calibration_pattern=calibration_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients
        )
        
        # Eye-to-hand specific result matrices
        self.base2cam_matrix = None          # Base to camera transformation matrix (primary result)
        self.target2end_matrix = None       # Target to end-effector transformation matrix (fixed attachment)
        
        # Best method tracking
        self.best_method = None
        self.best_method_name = ""
        
        print("✅ NewEyeToHandCalibrator initialized with base class architecture")
    
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_TSAI, verbose: bool = False) -> bool:
        """
        Perform eye-to-hand calibration using the specified method.
        
        This method provides backward compatibility with the original API while using
        the new base class architecture.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        # Use the base class workflow but call single method calibration
        if verbose:
            print(f"🔍 Running single-method eye-to-hand calibration...")
        
        # Validate input data (inherited from base class)
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters have not been loaded")
            
        if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
            raise ValueError("Robot poses have not been set")
            
        if self.images is None or len(self.images) == 0:
            raise ValueError("Images have not been loaded")
            
        if self.calibration_pattern is None:
            raise ValueError("Calibration pattern has not been set")
        
        # Detect patterns and extract valid data
        success = self.detect_pattern_points(verbose=verbose)
        if not success:
            raise ValueError("Pattern detection failed or insufficient patterns detected")
        
        self._calculate_poses_for_all_images(verbose=verbose)
        
        valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices = self._extract_valid_calibration_data(verbose=verbose)
        
        # Store valid data for use by calibration
        self._valid_calibration_indices = valid_indices
        self._valid_calibration_rvecs = valid_rvecs
        self._valid_calibration_tvecs = valid_tvecs
        self._valid_end2base_matrices = valid_end2base_matrices
        
        # Perform calibration with specified method
        result = self._calibrate_eye_to_hand_single_method(method, verbose)
        
        if result is not None:
            self.base2cam_matrix = result['base2cam_matrix']
            self.target2end_matrix = result['target2end_matrix']  # Use the optimal target2end from the result
            self.per_image_errors = result['per_image_errors']     # Use the correct per-image errors
            self.rms_error = result['rms_error']                   # Use the correct RMS error
            
            # Set up rvecs and tvecs aligned with all images
            self.rvecs = [None] * len(self.images)
            self.tvecs = [None] * len(self.images)
            
            for i, (valid_idx, rvec, tvec) in enumerate(zip(valid_indices, valid_rvecs, valid_tvecs)):
                self.rvecs[valid_idx] = rvec
                self.tvecs[valid_idx] = tvec
            
            self.calibration_completed = True
            
            if verbose:
                print(f"✅ Eye-to-hand calibration completed successfully!")
                print(f"   RMS error: {self.rms_error:.4f} pixels")
            
            return True
        else:
            if verbose:
                print(f"❌ Eye-to-hand calibration failed!")
            return False

    def _calibrate_eye_to_hand_single_method(self, method: int, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform eye-to-hand calibration using a single specified method.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            dict or None: Calibration results if successful, None if failed
        """
        try:
            # Validate input data (should already be set by base class workflow)
            if self.camera_matrix is None or self.distortion_coefficients is None:
                raise ValueError("Camera intrinsic parameters have not been loaded")
                
            if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
                raise ValueError("Robot poses have not been set")
                
            if not hasattr(self, '_valid_calibration_indices') or self._valid_calibration_indices is None:
                raise ValueError("Valid calibration data has not been extracted")
            
            # Use valid data stored during pattern detection
            valid_indices = self._valid_calibration_indices
            valid_rvecs = self._valid_calibration_rvecs
            valid_tvecs = self._valid_calibration_tvecs
            valid_end2base_matrices = self._valid_end2base_matrices
            
            if len(valid_indices) < 3:
                if verbose:
                    print(f"❌ Insufficient valid images: {len(valid_indices)} (need at least 3)")
                return None
            
            if verbose:
                print(f"Using {len(valid_indices)} valid images for eye-to-hand calibration")
            
            # Prepare data for OpenCV calibrateHandEye (eye-to-hand uses base2end transformations)
            # Convert end2base to base2end matrices for eye-to-hand
            base2end_matrices = []
            for end2base_matrix in valid_end2base_matrices:
                base2end_matrix = np.linalg.inv(end2base_matrix)
                base2end_matrices.append(base2end_matrix)
            
            # Extract rotation matrices and translation vectors
            base2end_Rs = np.array([matrix[:3, :3] for matrix in base2end_matrices])
            base2end_ts = np.array([matrix[:3, 3] for matrix in base2end_matrices])
            rvecs_array = np.array([rvec for rvec in valid_rvecs])
            tvecs_array = np.array([tvec for tvec in valid_tvecs])
            
            if verbose:
                print(f"Using base2end transformations (eye-to-hand configuration)")
            
            # Perform eye-to-hand calibration using OpenCV
            # NOTE: OpenCV calibrateHandEye returns cam2base for eye-to-hand, need to invert to get base2cam
            cam2base_R, cam2base_t = cv2.calibrateHandEye(
                base2end_Rs, base2end_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2base 4x4 transformation matrix
            cam2base_4x4 = np.eye(4)
            cam2base_4x4[:3, :3] = cam2base_R
            cam2base_4x4[:3, 3] = cam2base_t[:, 0]
            
            # For eye-to-hand, we need base2cam matrix, so invert the result
            base2cam_4x4 = np.linalg.inv(cam2base_4x4)
            
            # Store target2cam matrices as instance variables (like old implementation)
            self.target2cam_matrices = []
            for rvec, tvec in zip(valid_rvecs, valid_tvecs):
                target2cam_matrix = np.eye(4)
                target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                target2cam_matrix[:3, 3] = tvec[:, 0]
                self.target2cam_matrices.append(target2cam_matrix)
            
            # Store end2base matrices for the valid images only (to match target2cam_matrices indexing)
            self.end2base_matrices = valid_end2base_matrices
            
            # Calculate the optimal target2end matrix that minimizes reprojection error (using old approach)
            target2end_matrix = self._calculate_optimal_target2end_matrix_like_old(base2cam_4x4, verbose=True)
            
            # Calculate reprojection errors
            per_image_errors, rms_error = self._calculate_reprojection_errors(
                base2cam_4x4, target2end_matrix, verbose=True)
            
            if verbose:
                print(f"   RMS reprojection error: {rms_error:.4f} pixels")
            
            return {
                'base2cam_matrix': base2cam_4x4,
                'target2end_matrix': target2end_matrix,
                'rms_error': rms_error,
                'per_image_errors': per_image_errors
            }
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Calibration with {self._get_method_name(method)} failed: {e}")
            return None
    

    


    def _calculate_optimal_target2end_matrix_like_old(self, base2cam_4x4: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate the target2end matrix that minimizes reprojection error using old implementation approach.
        
        This matches the exact algorithm from the old eye_to_hand_calibration.py implementation.
        
        Args:
            base2cam_4x4: The base to camera transformation matrix from calibration
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to end-effector transformation matrix
        """
        if verbose:
            print("Calculating optimal target2end matrix using old algorithm...")
        
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
            
            for i in range(len(self.target2cam_matrices)):
                try:
                    # Calculate target2cam using the candidate target2end matrix
                    # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                    eyetohand_target2cam = base2cam_4x4 @ self.end2base_matrices[i] @ candidate_target2end
                    
                    # Get the corresponding valid image index for object/image points
                    valid_idx = self._valid_calibration_indices[i]
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[valid_idx], 
                        eyetohand_target2cam[:3, :3], 
                        eyetohand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[valid_idx], projected_points, cv2.NORM_L2) / len(projected_points)
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
            return best_target2end
        else:
            if verbose:
                print("⚠️ Could not find optimal target2end matrix, using first candidate")
            return candidate_target2end_matrices[0] if candidate_target2end_matrices else np.eye(4)

    def _calculate_optimal_target2end_matrix(self, base2cam_4x4: np.ndarray, 
                                           rvecs: List[np.ndarray], tvecs: List[np.ndarray],
                                           valid_end2base_matrices: List[np.ndarray],
                                           verbose: bool = False) -> np.ndarray:
        """
        Calculate the single target2end matrix that minimizes overall reprojection error.
        
        For eye-to-hand, the target is attached to the end-effector, so we need target2end.
        
        Args:
            base2cam_4x4: Base to camera transformation matrix from calibration
            rvecs: Rotation vectors from PnP solution (valid images only)
            tvecs: Translation vectors from PnP solution (valid images only)
            valid_end2base_matrices: End-effector to base matrices (valid images only)
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to end-effector transformation matrix
        """
        if verbose:
            print("Calculating optimal target2end matrix for eye-to-hand...")
        
        # Convert rvecs/tvecs to target2cam transformation matrices
        target2cam_matrices = []
        for rvec, tvec in zip(rvecs, tvecs):
            target2cam_matrix = np.eye(4)
            target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
            target2cam_matrix[:3, 3] = tvec[:, 0]
            target2cam_matrices.append(target2cam_matrix)
        
        best_error = float('inf')
        best_target2end = None
        
        # Try using each image's measurements to estimate target2end
        candidate_target2end_matrices = []
        cam2base_matrix = np.linalg.inv(base2cam_4x4)
        
        for i in range(len(target2cam_matrices)):
            # For eye-to-hand: target2end = base2end * cam2base * target2cam
            # where base2end = (end2base)^-1 and cam2base = (base2cam)^-1
            try:
                base2end_matrix = np.linalg.inv(valid_end2base_matrices[i])
                candidate_target2end = base2end_matrix @ cam2base_matrix @ target2cam_matrices[i]
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
            
            for i in range(len(target2cam_matrices)):
                try:
                    # Calculate target2cam using the candidate target2end matrix
                    # Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
                    eyetohand_target2cam = base2cam_4x4 @ valid_end2base_matrices[i] @ candidate_target2end
                    
                    # Extract rotation and translation for projection
                    R = eyetohand_target2cam[:3, :3]
                    t = eyetohand_target2cam[:3, 3:4]  # Keep as column vector
                    
                    # We need to find the corresponding object/image points
                    # Since we filtered valid data, we need to get them from the original arrays
                    # For now, let's use a simple approach - get all valid detection data
                    if hasattr(self, 'object_points') and hasattr(self, 'image_points'):
                        if i < len(self.object_points) and i < len(self.image_points):
                            # Project 3D points to image
                            projected_points, _ = cv2.projectPoints(
                                self.object_points[i], R, t, self.camera_matrix, self.distortion_coefficients
                            )
                            
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
            return best_target2end
        else:
            if verbose:
                print("⚠️ Could not find optimal target2end matrix, using first candidate")
            return candidate_target2end_matrices[0] if candidate_target2end_matrices else np.eye(4)
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get the base to camera transformation matrix (primary result for eye-to-hand)."""
        return self.base2cam_matrix
    
    def get_base2cam_matrix(self) -> Optional[np.ndarray]:
        """Get the base to camera transformation matrix."""
        return self.base2cam_matrix
    
    def get_target2end_matrix(self) -> Optional[np.ndarray]:
        """Get the target to end-effector transformation matrix."""
        return self.target2end_matrix
    
    def _get_method_name(self, method: int) -> str:
        """Convert OpenCV method constant to readable name."""
        method_names = {
            cv2.CALIB_HAND_EYE_TSAI: "TSAI",
            cv2.CALIB_HAND_EYE_PARK: "PARK",
            cv2.CALIB_HAND_EYE_HORAUD: "HORAUD",
            cv2.CALIB_HAND_EYE_ANDREFF: "ANDREFF",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "DANIILIDIS"
        }
        return method_names.get(method, f"Unknown({method})")
    
    def save_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to JSON file.
        
        Args:
            save_directory: Directory to save the results
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        results = {
            "camera_intrinsics": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist()
            },
            "eye_to_hand_calibration": {
                "base2cam_matrix": self.base2cam_matrix.tolist(),
                "target2end_matrix": self.target2end_matrix.tolist() if self.target2end_matrix is not None else None,
                "rms_error": self.rms_error,
                "per_image_errors": self.per_image_errors
            }
        }
        
        json_file_path = os.path.join(save_directory, "new_eye_to_hand_calibration_results.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"✅ New eye-to-hand calibration results saved to: {json_file_path}")

    # ============================================================================
    # Abstract Method Implementations (Required by HandEyeBaseCalibrator)
    # ============================================================================

    def _calibrate_with_best_method(self, verbose: bool = False) -> bool:
        """
        Perform eye-to-hand calibration using all methods and select the best one.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        # Available OpenCV hand-eye calibration methods
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
            (cv2.CALIB_HAND_EYE_PARK, "PARK"),
            (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
        ]
        
        if verbose:
            print(f"\n🧪 Testing {len(methods)} eye-to-hand calibration methods to find the best one...")
        
        best_method = None
        best_method_name = ""
        best_rms_error = float('inf')
        best_results = None
        
        # Test each method and find the one with smallest reprojection error
        for method, method_name in methods:
            try:
                if verbose:
                    print(f"\n--- Testing method: {method_name} ---")
                
                # Run calibration with this method
                result = self._calibrate_eye_to_hand_single_method(method, verbose)
                
                if result and result['rms_error'] < best_rms_error:
                    best_rms_error = result['rms_error']
                    best_method = method
                    best_method_name = method_name
                    best_results = result
                    
                    if verbose:
                        print(f"✅ New best method: {method_name} (RMS error: {best_rms_error:.4f})")
                elif result:
                    if verbose:
                        print(f"Method {method_name} completed with RMS error: {result['rms_error']:.4f}")
                else:
                    if verbose:
                        print(f"❌ Method {method_name} failed")
                        
            except Exception as e:
                if verbose:
                    print(f"❌ Method {method_name} failed with error: {e}")
                continue
        
        # Use the best results
        if best_results is not None:
            self.base2cam_matrix = best_results['base2cam_matrix']
            self.target2end_matrix = best_results['target2end_matrix']
            self.rms_error = best_results['rms_error']
            self.per_image_errors = best_results['per_image_errors']
            
            # Set up rvecs and tvecs aligned with all images (None for failed detections)
            self.rvecs = [None] * len(self.images)
            self.tvecs = [None] * len(self.images)
            
            # Map the valid rvecs/tvecs back to their original image positions
            valid_indices = self._valid_calibration_indices
            valid_rvecs = self._valid_calibration_rvecs
            valid_tvecs = self._valid_calibration_tvecs
            
            for i, (valid_idx, rvec, tvec) in enumerate(zip(valid_indices, valid_rvecs, valid_tvecs)):
                self.rvecs[valid_idx] = rvec
                self.tvecs[valid_idx] = tvec
            
            self.best_method = best_method
            self.best_method_name = best_method_name
            
            if verbose:
                print(f"\n🎯 Final result: Best method is {best_method_name}")
                print(f"   RMS reprojection error: {self.rms_error:.4f} pixels")
                print(f"   Eye-to-hand calibration completed successfully!")
            
            return True
        else:
            if verbose:
                print("\n❌ All calibration methods failed!")
            return False

    def _get_calibration_results_dict(self) -> Dict[str, Any]:
        """
        Get eye-to-hand calibration results in dictionary format for saving.
        
        Returns:
            dict: Eye-to-hand calibration results
        """
        return {
            "calibration_type": "eye_to_hand",
            "base2cam_matrix": self.base2cam_matrix.tolist() if self.base2cam_matrix is not None else None,
            "target2end_matrix": self.target2end_matrix.tolist() if self.target2end_matrix is not None else None,
            "best_method": int(self.best_method) if self.best_method is not None else None,
            "best_method_name": self.best_method_name
        }

    def _get_projected_points_for_image(self, image_index: int) -> Optional[np.ndarray]:
        """
        Get projected points for a specific image using eye-to-hand transformation chain.
        
        Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
        
        Args:
            image_index: Index of the image
            
        Returns:
            np.ndarray or None: Projected points if successful
        """
        if not self.is_calibrated():
            return None
            
        # Check if this image is in valid calibration data
        if image_index not in self._valid_calibration_indices:
            return None
            
        # Get the index in the valid data arrays
        valid_index = self._valid_calibration_indices.index(image_index)
        end2base_matrix = self._valid_end2base_matrices[valid_index]
        object_points = self._valid_object_points[valid_index]
            
        try:
            # Calculate target2cam transformation using eye-to-hand chain
            # Eye-to-hand: target2cam = base2cam * end2base * target2end
            target2cam = self.base2cam_matrix @ end2base_matrix @ self.target2end_matrix
            
            # Project 3D object points to image coordinates
            projected_points, _ = cv2.projectPoints(
                object_points,
                target2cam[:3, :3],
                target2cam[:3, 3],
                self.camera_matrix,
                self.distortion_coefficients
            )
            
            return projected_points
            
        except Exception:
            return None

    def _calculate_reprojection_errors(self, transformation_matrix: Optional[np.ndarray] = None,
                                     target2end_matrix: Optional[np.ndarray] = None,
                                     rvecs: Optional[List[np.ndarray]] = None, 
                                     tvecs: Optional[List[np.ndarray]] = None,
                                     verbose: bool = False) -> Tuple[List[float], float]:
        """
        Calculate reprojection errors using eye-to-hand transformation chain.
        
        Eye-to-hand transformation chain: target2cam = base2cam * end2base * target2end
        
        Args:
            transformation_matrix: base2cam transformation matrix (or None to use self.base2cam_matrix)
            target2end_matrix: target2end transformation matrix (or None to use self.target2end_matrix)
            rvecs: Rotation vectors (not used in eye-to-hand)
            tvecs: Translation vectors (not used in eye-to-hand)
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple[List[float], float]: (per_image_errors, rms_error)
        """
        # Use provided matrices or fall back to instance variables
        base2cam_matrix = transformation_matrix if transformation_matrix is not None else self.base2cam_matrix
        target2end_matrix = target2end_matrix if target2end_matrix is not None else self.target2end_matrix
        
        if base2cam_matrix is None:
            raise ValueError("base2cam_matrix not available")
        if target2end_matrix is None:
            raise ValueError("target2end_matrix not available")
        
        # Use valid data stored during calibration
        valid_indices = self._valid_calibration_indices
        valid_end2base_matrices = self.end2base_matrices  # This now contains only valid matrices
        valid_image_points = [self.image_points[i] for i in valid_indices]
        valid_object_points = [self.object_points[i] for i in valid_indices]
        
        per_image_errors = []
        total_error = 0.0
        valid_images = 0
        
        if verbose:
            print(f"Calculating reprojection errors for {len(valid_indices)} valid images (eye-to-hand)...")
        
        for i in range(len(valid_indices)):
            try:
                # Calculate target2cam transformation using eye-to-hand chain
                # Eye-to-hand: target2cam = base2cam * end2base * target2end
                target2cam = base2cam_matrix @ valid_end2base_matrices[i] @ target2end_matrix
                
                # Project 3D object points to image coordinates
                projected_points, _ = cv2.projectPoints(
                    valid_object_points[i],
                    target2cam[:3, :3],
                    target2cam[:3, 3],
                    self.camera_matrix,
                    self.distortion_coefficients
                )
                
                # Calculate reprojection error for this image
                error = cv2.norm(valid_image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                per_image_errors.append(error)
                total_error += error * error
                valid_images += 1
                
                if verbose:
                    print(f"   Image {valid_indices[i]}: {error:.4f} pixels")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Image {valid_indices[i]}: Error calculating reprojection - {e}")
                per_image_errors.append(float('inf'))
        
        # Calculate RMS error
        if valid_images > 0:
            rms_error = np.sqrt(total_error / valid_images)
            if verbose:
                print(f"📊 Eye-to-hand Reprojection Error Summary:")
                print(f"   • Valid images: {valid_images}/{len(valid_indices)}")
                print(f"   • RMS error: {rms_error:.4f} pixels")
                finite_errors = [e for e in per_image_errors if not np.isinf(e)]
                if finite_errors:
                    print(f"   • Min error: {min(finite_errors):.4f} pixels")
                    print(f"   • Max error: {max(finite_errors):.4f} pixels")
        else:
            rms_error = float('inf')
            if verbose:
                print("❌ No valid images for reprojection error calculation")
        
        return per_image_errors, rms_error

    def _optimize_transformation_matrices(self, iterations: int, ftol_rel: float, verbose: bool) -> Optional[float]:
        """
        Optimize the eye-to-hand transformation matrices to minimize reprojection error.
        
        Args:
            iterations: Number of optimization iterations
            ftol_rel: Relative tolerance for convergence
            verbose: Whether to print optimization progress
            
        Returns:
            float or None: Optimized RMS error if successful
        """
        if not HAS_NLOPT:
            if verbose:
                print("⚠️ nlopt not available - skipping optimization")
            return None
            
        try:
            if verbose:
                print(f"🔍 Optimizing eye-to-hand transformation matrices...")
                
            # Convert initial matrices to parameter vectors
            base2cam_params = matrix_to_xyz_rpy(self.base2cam_matrix)
            target2end_params = matrix_to_xyz_rpy(self.target2end_matrix)
            
            # Combined parameter vector: [base2cam_params, target2end_params] (12 total)
            initial_params = np.concatenate([base2cam_params, target2end_params])
            
            # Setup joint optimization (same as old implementation)
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 12)
            opt.set_maxeval(iterations)
            opt.set_ftol_rel(ftol_rel)
            
            def joint_objective(params, grad):
                """Objective function for joint optimization."""
                base2cam_params = params[:6]
                target2end_params = params[6:]
                
                base2cam_matrix = xyz_rpy_to_matrix(base2cam_params)
                target2end_matrix = xyz_rpy_to_matrix(target2end_params)
                
                # Calculate reprojection error using ALL original data (not just valid subset)
                # This matches the old implementation's behavior for optimization
                total_error = 0.0
                valid_images = 0
                
                for i in range(len(self.image_points)):
                    try:
                        # Use the same transformation chain as the old implementation:
                        # Eye-to-hand: target2cam = base2cam * end2base * target2end
                        target2cam = base2cam_matrix @ self.end2base_matrices[i] @ target2end_matrix
                        
                        # Project 3D object points to image coordinates
                        projected_points, _ = cv2.projectPoints(
                            self.object_points[i],
                            target2cam[:3, :3],
                            target2cam[:3, 3],
                            self.camera_matrix,
                            self.distortion_coefficients
                        )
                        
                        # Calculate reprojection error for this image
                        error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                        total_error += error * error
                        valid_images += 1
                        
                    except Exception:
                        continue
                
                # Return RMS error
                if valid_images > 0:
                    return np.sqrt(total_error / valid_images)
                else:
                    return float('inf')
            
            opt.set_min_objective(joint_objective)
            
            # Perform optimization
            initial_error = joint_objective(initial_params, None)
            if verbose:
                print(f"   Starting optimization from {initial_error:.4f} pixels...")
                
            optimized_params = opt.optimize(initial_params)
            final_error = joint_objective(optimized_params, None)
            
            if verbose:
                print(f"   Joint optimization: {initial_error:.4f} -> {final_error:.4f} pixels")
                improvement_pct = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0
                print(f"   Improvement: {improvement_pct:.1f}%")
            
            # Update matrices with optimized parameters
            self.base2cam_matrix = xyz_rpy_to_matrix(optimized_params[:6])
            self.target2end_matrix = xyz_rpy_to_matrix(optimized_params[6:])
            
            # Recalculate errors with optimized matrices using the same method as optimization
            # This ensures consistency between optimization objective and final error reporting
            total_error = 0.0
            valid_images = 0
            per_image_errors = []
            
            for i in range(len(self.image_points)):
                try:
                    # Use the same transformation chain as the old implementation:
                    # Eye-to-hand: target2cam = base2cam * end2base * target2end
                    target2cam = self.base2cam_matrix @ self.end2base_matrices[i] @ self.target2end_matrix
                    
                    # Project 3D object points to image coordinates
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i],
                        target2cam[:3, :3],
                        target2cam[:3, 3],
                        self.camera_matrix,
                        self.distortion_coefficients
                    )
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    per_image_errors.append(error)
                    total_error += error * error
                    valid_images += 1
                    
                except Exception:
                    per_image_errors.append(float('inf'))
            
            # Update with consistent error calculation
            if valid_images > 0:
                self.rms_error = np.sqrt(total_error / valid_images)
                self.per_image_errors = per_image_errors
            else:
                self.rms_error = float('inf')
                self.per_image_errors = [float('inf')] * len(self.image_points)
            
            if verbose:
                improvement = initial_error - final_error
                improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
                print(f"   Initial error: {initial_error:.4f} pixels")
                print(f"   Final error: {final_error:.4f} pixels")
                print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
            
            return final_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Optimization failed: {e}")
            return None

    def optimize_calibration(self, iterations: int = 100, ftol_rel: float = 1e-6, verbose: bool = False) -> float:
        """
        Optimize eye-to-hand calibration results by jointly refining base2cam and target2end matrices.
        
        This method uses nonlinear optimization to minimize reprojection error by
        simultaneously optimizing both the base-to-camera transformation
        and the target-to-end-effector transformation.
        
        Args:
            iterations: Maximum number of optimization iterations
            ftol_rel: Relative tolerance for convergence
            verbose: Whether to print optimization progress
            
        Returns:
            float: Final RMS reprojection error after optimization
            
        Raises:
            ValueError: If calibration has not been completed
            ImportError: If nlopt optimization library is not available
        """
        if not self.is_calibrated():
            raise ValueError("Calibration must be completed before optimization. Call calibrate() first.")
            
        if not HAS_NLOPT:
            raise ImportError("nlopt library is required for optimization but not available")
            
        if verbose:
            print(f"Starting eye-to-hand optimization with {iterations} max iterations...")
            print(f"Initial RMS error: {self.rms_error:.4f} pixels")
            print("   Note: Optimization refines both target2end and cam2base matrices")
            print("         using nonlinear optimization to minimize reprojection error")
            
        # Store initial values
        initial_base2cam = self.base2cam_matrix.copy()
        initial_target2end = self.target2end_matrix.copy()
        initial_error = self.rms_error
        
        try:
            # Perform joint optimization using the internal method
            optimized_error = self._optimize_transformation_matrices(iterations, ftol_rel, verbose)
            
            if optimized_error is not None:
                if verbose:
                    improvement = initial_error - self.rms_error
                    improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
                    print(f"Eye-to-hand optimization completed!")
                    print(f"Final RMS error: {self.rms_error:.4f} pixels")
                    if improvement > 0:
                        print(f"Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                    else:
                        print(f"Note: Optimization optimizes over all data, may show different baseline than filtered calibration")
                    
                return self.rms_error
            else:
                if verbose:
                    print("❌ Optimization failed - restoring initial matrices")
                # Restore original matrices
                self.base2cam_matrix = initial_base2cam
                self.target2end_matrix = initial_target2end
                self.rms_error = initial_error
                return initial_error
                
        except Exception as e:
            if verbose:
                print(f"❌ Eye-to-hand optimization failed: {e}")
            # Restore original matrices
            self.base2cam_matrix = initial_base2cam
            self.target2end_matrix = initial_target2end
            self.rms_error = initial_error
            return initial_error

    def _calculate_simple_target2end_matrix(self, base2cam_matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate target2end matrix using the same approach as the old implementation.
        
        This method finds the target2end transformation that best explains all the
        observed target positions across all images, providing a reasonable starting
        point for optimization.
        
        Args:
            base2cam_matrix: The base to camera transformation matrix from calibration
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to end-effector transformation matrix
        """
        if verbose:
            print("Calculating target2end matrix (old implementation approach)...")
        
        best_error = float('inf')
        best_target2end = None
        
        # Try using each image's target2cam transformation to estimate target2end
        # Then find the one that gives the smallest overall reprojection error
        candidate_target2end_matrices = []
        cam2base_matrix = np.linalg.inv(base2cam_matrix)
        
        for i in range(len(self.image_points)):
            try:
                # Calculate target2cam for this image using solvePnP
                ret, rvec, tvec = cv2.solvePnP(
                    self.object_points[i], 
                    self.image_points[i], 
                    self.camera_matrix, 
                    self.distortion_coefficients
                )
                
                if ret:
                    # Create target2cam transformation matrix
                    target2cam_matrix = np.eye(4)
                    target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                    target2cam_matrix[:3, 3] = tvec[:, 0]
                    
                    # For eye-to-hand: target2end = end2base * cam2base * target2cam
                    candidate_target2end = self.end2base_matrices[i] @ cam2base_matrix @ target2cam_matrix
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
                    target2cam = base2cam_matrix @ self.end2base_matrices[i] @ candidate_target2end
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        target2cam[:3, :3], 
                        target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients
                    )
                    
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
        else:
            if verbose:
                print("⚠️ Could not find optimal target2end matrix, using first candidate")
            best_target2end = candidate_target2end_matrices[0]
        
        return best_target2end
