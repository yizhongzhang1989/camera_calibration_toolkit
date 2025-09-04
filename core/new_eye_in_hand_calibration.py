"""
New Eye-in-Hand Calibration Module
==================================

This module provides the NewEyeInHandCalibrator class that inherits from HandEyeBaseCalibrator.
It contains only IO functionality for eye-in-hand calibration data handling.

Key Features:
- Inherits all common IO functionality from HandEyeBaseCalibrator
- Provides eye-in-hand specific data structures
- Maintains API compatibility for data handling
- Separates IO from calibration logic

Architecture:
    BaseCalibrator (images/patterns)
    -> HandEyeBaseCalibrator (robot poses/transformations/IO)
       -> NewEyeInHandCalibrator (eye-in-hand specific IO)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import json
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from .hand_eye_base_calibration import HandEyeBaseCalibrator
from .calibration_patterns import CalibrationPattern


class NewEyeInHandCalibrator(HandEyeBaseCalibrator):
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
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize NewEyeInHandCalibrator for IO operations.
        
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
        
        # Eye-in-hand specific transformation matrices
        self.cam2end_matrix = None              # Camera to end-effector transformation (primary result)
        self.target2base_matrix = None          # Target to robot base transformation (secondary result)
    
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
    
    # ============================================================================
    # Eye-in-Hand Specific Result Access Methods
    # ============================================================================
    
    def get_calibration_results(self) -> Dict[str, Any]:
        """
        Get eye-in-hand specific calibration results.
        
        Returns:
            dict: Dictionary containing eye-in-hand calibration results
        """
        base_info = self.get_calibration_info()
        
        eye_in_hand_info = {
            "calibration_type": "eye_in_hand",
            "cam2end_matrix": self.cam2end_matrix.tolist() if self.cam2end_matrix is not None else None,
            "target2base_matrix": self.target2base_matrix.tolist() if self.target2base_matrix is not None else None,
            "has_cam2end": self.cam2end_matrix is not None,
            "has_target2base": self.target2base_matrix is not None
        }
        
        # Merge base info with eye-in-hand specific info
        base_info.update(eye_in_hand_info)
        return base_info
    
    def save_eye_in_hand_results(self, save_directory: str) -> None:
        """
        Save eye-in-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results
        """
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            # Get all calibration results
            results = self.get_calibration_results()
            
            # Save main results file
            results_file = os.path.join(save_directory, "eye_in_hand_calibration_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Eye-in-hand calibration results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save eye-in-hand results: {e}")
    
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

    def get_calibration_results(self) -> Dict[str, Any]:
        """
        Get complete eye-in-hand calibration results.
        
        Returns:
            Dict containing all calibration data and results
        """
        # Return all eye-in-hand calibration data
        return {
            'images': self.images,
            'image_paths': self.image_paths,
            'end2base_matrices': self.end2base_matrices,
            'cam2end_matrix': self.cam2end_matrix,
            'target2base_matrix': self.target2base_matrix,
            'camera_matrix': self.camera_matrix,
            'distortion_coefficients': self.distortion_coefficients,
            'calibration_pattern': self.calibration_pattern,
            'rms_error': self.rms_error
        }

    def calibrate(self, **kwargs) -> bool:
        """
        Placeholder calibrate method for IO-only architecture.
        
        Note: This is a placeholder method to satisfy the abstract base class.
        Actual calibration logic has been moved to dedicated calibration modules.
        
        Args:
            **kwargs: Additional parameters (not used in IO-only version)
            
        Returns:
            bool: Always returns True as this is IO-only
        """
        print("⚠️ Warning: calibrate() called on IO-only NewEyeInHandCalibrator")
        print("   Calibration algorithms have been moved to dedicated modules.")
        print("   This class now handles only data loading and management.")
        return True

    def save_results(self, save_directory: str) -> None:
        """
        Save eye-in-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results to
        """
        self.save_eye_in_hand_results(save_directory)
