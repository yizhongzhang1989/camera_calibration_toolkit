"""
New Eye-to-Hand Calibration Module - IO Only

This module provides IO operations for eye-to-hand calibration data management.
Eye-to-hand configuration: camera is stationary, target is mounted on robot end-effector.

All calibration algorithms have been removed. This module only handles:
- Data loading and validation
- Result storage and retrieval  
- Parameter management
- File I/O operations

Author: Camera Calibration Toolkit
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Any

from .hand_eye_base_calibration import HandEyeBaseCalibrator


class NewEyeToHandCalibrator(HandEyeBaseCalibrator):
    """
    Eye-to-Hand calibration data management class - IO operations only.
    
    This class provides IO functionality for eye-to-hand calibration data where:
    - Camera is stationary (fixed in workspace) 
    - Target/calibration pattern is mounted on robot end-effector
    - Calibrates base2cam_matrix (robot base to camera transformation)
    
    All calibration algorithms have been removed - use dedicated calibration modules.
    """
    
    def __init__(self, images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None,
                 calibration_pattern=None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize NewEyeToHandCalibrator for IO operations.
        
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
        self.base2cam_matrix = None             # Robot base to camera transformation (primary result)
        self.target2end_matrix = None           # Target to end-effector transformation (secondary result)
    
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

    def get_calibration_results(self) -> Dict[str, Any]:
        """
        Get complete eye-to-hand calibration results.
        
        Returns:
            Dict containing all calibration data and results
        """
        # Return all eye-to-hand calibration data
        return {
            'images': self.images,
            'image_paths': self.image_paths,
            'end2base_matrices': self.end2base_matrices,  # Fixed: use end2base_matrices instead of base2end_matrices
            'base2cam_matrix': self.base2cam_matrix,
            'target2end_matrix': self.target2end_matrix,  # Fixed: use target2end_matrix instead of target2cam_matrix
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
        print("⚠️ Warning: calibrate() called on IO-only NewEyeToHandCalibrator")
        print("   Calibration algorithms have been moved to dedicated modules.")
        print("   This class now handles only data loading and management.")
        return True

    def save_results(self, save_directory: str) -> None:
        """
        Save eye-to-hand calibration results to directory.
        
        Args:
            save_directory: Directory to save results to
        """
        self.save_eye_to_hand_results(save_directory)
