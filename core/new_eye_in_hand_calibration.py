"""
New Eye-in-Hand Calibration Module
==================================

This module provides the new EyeInHandCalibrator class that inherits from HandEyeBaseCalibrator.
This is a test implementation to validate the new base class architecture before refactoring
the existing EyeInHandCalibrator.

Key Features:
- Inherits all common functionality from HandEyeBaseCalibrator
- Implements eye-in-hand specific coordinate system transformations
- Uses end2base transformations for calibration
- Provides cam2end_matrix as the primary result
- Maintains API compatibility with existing EyeInHandCalibrator

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/workflow)
       -> NewEyeInHandCalibrator (eye-in-hand specific logic)
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


class NewEyeInHandCalibrator(HandEyeBaseCalibrator):
    """
    New Eye-in-Hand Calibration class inheriting from HandEyeBaseCalibrator.
    
    This class calibrates the transformation between a camera mounted on a robot
    end-effector and the end-effector coordinate frame using the new base class architecture.
    
    Eye-in-hand specific attributes:
        cam2end_matrix: Camera to end-effector transformation matrix (primary result)
        target2base_matrix: Target to base transformation matrix (secondary result)
    
    Inherited from HandEyeBaseCalibrator:
        end2base_matrices: Robot pose transformations
        target2cam_matrices: Target to camera transformations
        calibration_completed, rms_error, per_image_errors: Status and quality metrics
    
    Inherited from BaseCalibrator:
        images, image_paths, image_points, object_points: Image and pattern data
        camera_matrix, distortion_coefficients: Camera intrinsics
        rvecs, tvecs: Extrinsic parameters for each image
    """
    
    def __init__(self, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None, 
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize NewEyeInHandCalibrator with the same interface as HandEyeCalibrator.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            end2base_matrices: List of 4x4 transformation matrices from end-effector to base
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None
            distortion_coefficients: Camera distortion coefficients or None
        """
        # Call parent constructor with unified interface
        super().__init__(images, end2base_matrices, image_paths, calibration_pattern, 
                        camera_matrix, distortion_coefficients)
        
        # Eye-in-hand specific results
        self.cam2end_matrix = None           # Camera to end-effector transformation matrix (primary result)
        self.target2base_matrix = None       # Target to base transformation matrix (secondary result)
        
        print("✅ NewEyeInHandCalibrator initialized with base class architecture")
    
    # ============================================================================
    # Implementation of Abstract Methods from HandEyeBaseCalibrator
    # ============================================================================
    
    def _calibrate_with_best_method(self, verbose: bool = False) -> bool:
        """
        Perform eye-in-hand calibration using all methods and select the best one.
        
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
            print(f"\n🧪 Testing {len(methods)} eye-in-hand calibration methods to find the best one...")
        
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
                result = self._calibrate_eye_in_hand_single_method(method, verbose)
                
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
            self.cam2end_matrix = best_results['cam2end_matrix']
            self.target2base_matrix = best_results['target2base_matrix']
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
                print(f"   Eye-in-hand calibration completed successfully!")
            
            return True
        else:
            if verbose:
                print("\n❌ All calibration methods failed!")
            return False
    
    def _calibrate_eye_in_hand_single_method(self, method: int, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform eye-in-hand calibration using a single specified method.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            dict or None: Calibration results if successful, None if failed
        """
        try:
            # Get valid data from stored calibration data
            valid_indices = self._valid_calibration_indices
            rvecs = self._valid_calibration_rvecs
            tvecs = self._valid_calibration_tvecs
            valid_end2base_matrices = self._valid_end2base_matrices
            
            if verbose:
                print(f"Using {len(valid_indices)} valid images for eye-in-hand calibration")
            
            # Prepare data for OpenCV calibrateHandEye (eye-in-hand uses end2base transformations)
            end2base_Rs = np.array([matrix[:3, :3] for matrix in valid_end2base_matrices])
            end2base_ts = np.array([matrix[:3, 3] for matrix in valid_end2base_matrices])
            rvecs_array = np.array([rvec for rvec in rvecs])
            tvecs_array = np.array([tvec for tvec in tvecs])
            
            if verbose:
                print(f"Using end2base transformations (eye-in-hand configuration)")
            
            # Perform eye-in-hand calibration using OpenCV
            cam2end_R, cam2end_t = cv2.calibrateHandEye(
                end2base_Rs, end2base_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2end 4x4 transformation matrix
            cam2end_4x4 = np.eye(4)
            cam2end_4x4[:3, :3] = cam2end_R
            cam2end_4x4[:3, 3] = cam2end_t[:, 0]
            
            # Calculate the optimal target2base matrix that minimizes reprojection error
            target2base_matrix = self._calculate_optimal_target2base_matrix(
                cam2end_4x4, rvecs, tvecs, valid_end2base_matrices, verbose)
            
            # Calculate reprojection errors
            per_image_errors, rms_error = self._calculate_reprojection_errors(
                cam2end_4x4, target2base_matrix, rvecs, tvecs, verbose)
            
            return {
                'cam2end_matrix': cam2end_4x4,
                'target2base_matrix': target2base_matrix,
                'rms_error': rms_error,
                'per_image_errors': per_image_errors
            }
            
        except Exception as e:
            if verbose:
                print(f"Eye-in-hand calibration failed: {e}")
            return None
    
    def _calculate_optimal_target2base_matrix(self, cam2end_4x4: np.ndarray, 
                                            rvecs: List[np.ndarray], tvecs: List[np.ndarray],
                                            valid_end2base_matrices: List[np.ndarray],
                                            verbose: bool = False) -> np.ndarray:
        """
        Calculate the single target2base matrix that minimizes overall reprojection error.
        
        Args:
            cam2end_4x4: Camera to end-effector transformation matrix from calibration
            rvecs: Rotation vectors from PnP solution (valid images only)
            tvecs: Translation vectors from PnP solution (valid images only)
            valid_end2base_matrices: End-effector to base matrices (valid images only)
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to base transformation matrix
        """
        if verbose:
            print("Calculating optimal target2base matrix for eye-in-hand...")
        
        # Convert rvecs/tvecs to target2cam transformation matrices
        target2cam_matrices = []
        for rvec, tvec in zip(rvecs, tvecs):
            target2cam_matrix = np.eye(4)
            target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
            target2cam_matrix[:3, 3] = tvec[:, 0]
            target2cam_matrices.append(target2cam_matrix)
        
        best_error = float('inf')
        best_target2base = None
        
        # Try using each image's measurements to estimate target2base
        candidate_target2base_matrices = []
        
        for i in range(len(target2cam_matrices)):
            # Calculate target2base using this image's measurements
            # Eye-in-hand: target2base = end2base * cam2end * target2cam
            candidate_target2base = valid_end2base_matrices[i] @ cam2end_4x4 @ target2cam_matrices[i]
            candidate_target2base_matrices.append(candidate_target2base)
        
        # Test each candidate to find the one with minimum overall reprojection error
        for candidate_idx, candidate_target2base in enumerate(candidate_target2base_matrices):
            total_error = 0.0
            valid_images = 0
            
            # Get valid image points and object points for this test
            valid_image_points = [self.image_points[i] for i in self._valid_calibration_indices]
            valid_object_points = [self.object_points[i] for i in self._valid_calibration_indices]
            
            for i in range(len(valid_image_points)):
                try:
                    # Calculate target2cam using the candidate target2base matrix
                    # Eye-in-hand: target2cam = inv(cam2end) * inv(end2base) * target2base
                    end2cam_matrix = np.linalg.inv(cam2end_4x4)
                    base2end_matrix = np.linalg.inv(valid_end2base_matrices[i])
                    eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ candidate_target2base
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        valid_object_points[i], 
                        eyeinhand_target2cam[:3, :3], 
                        eyeinhand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients
                    )
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(valid_image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
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
                    best_target2base = candidate_target2base.copy()
                    if verbose:
                        print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f} (best so far)")
                elif verbose:
                    print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f}")
        
        if best_target2base is not None:
            if verbose:
                print(f"✅ Optimal target2base matrix found with RMS error: {best_error:.4f}")
        else:
            if verbose:
                print("⚠️ Could not find optimal target2base matrix, using first candidate")
            best_target2base = candidate_target2base_matrices[0]
        
        return best_target2base
    
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_HORAUD, verbose: bool = False) -> bool:
        """
        Perform eye-in-hand calibration using the specified method.
        
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
            print(f"🔍 Running single-method eye-in-hand calibration...")
        
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
        result = self._calibrate_eye_in_hand_single_method(method, verbose)
        
        if result is not None:
            self.cam2end_matrix = result['cam2end_matrix']
            self.target2base_matrix = result['target2base_matrix']
            self.rms_error = result['rms_error']
            self.per_image_errors = result['per_image_errors']
            
            # Set up rvecs and tvecs aligned with all images
            self.rvecs = [None] * len(self.images)
            self.tvecs = [None] * len(self.images)
            
            for i, (valid_idx, rvec, tvec) in enumerate(zip(valid_indices, valid_rvecs, valid_tvecs)):
                self.rvecs[valid_idx] = rvec
                self.tvecs[valid_idx] = tvec
            
            self.calibration_completed = True
            
            if verbose:
                print(f"✅ Eye-in-hand calibration completed successfully!")
                print(f"   RMS error: {self.rms_error:.4f} pixels")
            
            return True
        else:
            if verbose:
                print(f"❌ Eye-in-hand calibration failed!")
            return False
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the camera to end-effector transformation matrix (primary result for eye-in-hand).
        
        Returns:
            np.ndarray or None: Camera to end-effector transformation matrix if calibration completed
        """
        return self.cam2end_matrix
    
    def _calculate_reprojection_errors(self, transformation_matrix: Optional[np.ndarray] = None,
                                     target2end_matrix: Optional[np.ndarray] = None,
                                     rvecs: Optional[List[np.ndarray]] = None, 
                                     tvecs: Optional[List[np.ndarray]] = None,
                                     verbose: bool = False) -> Tuple[List[float], float]:
        """
        Calculate reprojection errors using eye-in-hand transformation chain.
        
        Eye-in-hand transformation chain: target2cam = inv(cam2end) * inv(end2base) * target2base
        
        Args:
            transformation_matrix: cam2end transformation matrix (or None to use self.cam2end_matrix)
            target2end_matrix: target2base transformation matrix (or None to use self.target2base_matrix)
            rvecs: Rotation vectors (or None to use valid stored vectors)
            tvecs: Translation vectors (or None to use valid stored vectors)
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple[List[float], float]: (per_image_errors, rms_error)
        """
        # Use provided matrices or fall back to instance variables
        cam2end_matrix = transformation_matrix if transformation_matrix is not None else self.cam2end_matrix
        target2base_matrix = target2end_matrix if target2end_matrix is not None else self.target2base_matrix
        
        if cam2end_matrix is None:
            raise ValueError("cam2end_matrix not available")
        if target2base_matrix is None:
            raise ValueError("target2base_matrix not available")
        
        # Use valid data stored during calibration
        valid_indices = self._valid_calibration_indices
        valid_end2base_matrices = self._valid_end2base_matrices
        valid_image_points = [self.image_points[i] for i in valid_indices]
        valid_object_points = [self.object_points[i] for i in valid_indices]
        
        per_image_errors = []
        total_error = 0.0
        valid_images = 0
        
        if verbose:
            print(f"Calculating reprojection errors for {len(valid_indices)} valid images (eye-in-hand)...")
        
        for i in range(len(valid_indices)):
            try:
                # Calculate target2cam transformation using eye-in-hand chain
                # Eye-in-hand: target2cam = inv(cam2end) * inv(end2base) * target2base
                end2cam_matrix = np.linalg.inv(cam2end_matrix)
                base2end_matrix = np.linalg.inv(valid_end2base_matrices[i])
                target2cam = end2cam_matrix @ base2end_matrix @ target2base_matrix
                
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
                print(f"📊 Eye-in-hand Reprojection Error Summary:")
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
    
    def _get_projected_points_for_image(self, image_index: int) -> Optional[np.ndarray]:
        """
        Get projected points for a specific image using eye-in-hand transformation chain.
        
        Args:
            image_index: Index of the image
            
        Returns:
            np.ndarray or None: Projected points if successful
        """
        if not self.is_calibrated():
            return None
            
        try:
            # Calculate target2cam transformation using eye-in-hand chain
            end2cam_matrix = np.linalg.inv(self.cam2end_matrix)
            base2end_matrix = np.linalg.inv(self.end2base_matrices[image_index])
            target2cam = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
            
            # Project 3D object points to image coordinates
            projected_points, _ = cv2.projectPoints(
                self.object_points[image_index],
                target2cam[:3, :3],
                target2cam[:3, 3],
                self.camera_matrix,
                self.distortion_coefficients
            )
            
            return projected_points
            
        except Exception:
            return None
    
    def _get_calibration_results_dict(self) -> Dict[str, Any]:
        """
        Get eye-in-hand calibration results in dictionary format for saving.
        
        Returns:
            dict: Eye-in-hand calibration results
        """
        return {
            "calibration_type": "eye_in_hand",
            "cam2end_matrix": self.cam2end_matrix.tolist() if self.cam2end_matrix is not None else None,
            "target2base_matrix": self.target2base_matrix.tolist() if self.target2base_matrix is not None else None,
            "best_method": int(self.best_method) if self.best_method is not None else None,
            "best_method_name": self.best_method_name
        }
    
    # ============================================================================
    # Optimization Methods (Optional - uses nlopt if available)
    # ============================================================================
    
    def _optimize_transformation_matrices(self, iterations: int, ftol_rel: float, verbose: bool) -> Optional[float]:
        """
        Optimize the eye-in-hand transformation matrices to minimize reprojection error.
        
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
                print(f"🔍 Optimizing eye-in-hand transformation matrices...")
                
            # Convert initial matrices to parameter vectors
            cam2end_params = matrix_to_xyz_rpy(self.cam2end_matrix)
            target2base_params = matrix_to_xyz_rpy(self.target2base_matrix)
            
            # Combined parameter vector: [cam2end_params, target2base_params] (12 total)
            initial_params = np.concatenate([cam2end_params, target2base_params])
            
            # Setup joint optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 12)
            opt.set_maxeval(iterations)
            
            def joint_objective(params, grad):
                """Objective function for joint optimization."""
                cam2end_params = params[:6]
                target2base_params = params[6:]
                
                cam2end_matrix = xyz_rpy_to_matrix(cam2end_params)
                target2base_matrix = xyz_rpy_to_matrix(target2base_params)
                
                # Calculate reprojection error
                _, rms_error = self._calculate_reprojection_errors(
                    cam2end_matrix, target2base_matrix, verbose=False)
                
                return rms_error
            
            opt.set_min_objective(joint_objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Perform optimization
            initial_error = joint_objective(initial_params, None)
            optimized_params = opt.optimize(initial_params)
            final_error = joint_objective(optimized_params, None)
            
            # Update matrices with optimized parameters
            self.cam2end_matrix = xyz_rpy_to_matrix(optimized_params[:6])
            self.target2base_matrix = xyz_rpy_to_matrix(optimized_params[6:])
            
            # Recalculate errors with optimized matrices
            self.per_image_errors, self.rms_error = self._calculate_reprojection_errors()
            
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
    
    # ============================================================================
    # Compatibility Methods (matching original EyeInHandCalibrator API)
    # ============================================================================
    
    def get_rms_error(self) -> Optional[float]:
        """Get overall RMS reprojection error (lower is better)."""
        return self.rms_error
    
    def get_per_image_errors(self) -> Optional[List[float]]:
        """Get per-image reprojection errors."""
        return self.per_image_errors
    
    def save_results(self, save_directory: str) -> None:
        """
        Save eye-in-hand calibration results to JSON file.
        
        Args:
            save_directory: Directory to save the results
        """
        if not self.is_calibrated():
            raise ValueError("Calibration has not been completed yet")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Use base class method to get comprehensive results
        results = {
            "camera_intrinsics": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist()
            },
            "eye_in_hand_calibration": self._get_calibration_results_dict()
        }
        
        # Add common results from base class
        base_results = super().save_results.__wrapped__(self, save_directory) if hasattr(super().save_results, '__wrapped__') else {}
        
        json_file_path = os.path.join(save_directory, "new_eye_in_hand_calibration_results.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"✅ New eye-in-hand calibration results saved to: {json_file_path}")
    
    def __str__(self) -> str:
        """String representation of the calibrator."""
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        return f"NewEyeInHandCalibrator(status={status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"NewEyeInHandCalibrator("
                f"images={len(self.images) if self.images else 0}, "
                f"transforms={len(self.end2base_matrices) if self.end2base_matrices else 0}, "
                f"calibrated={self.is_calibrated()})")
