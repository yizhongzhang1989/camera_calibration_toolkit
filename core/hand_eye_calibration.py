"""
Hand-Eye Calibration Module
===========================

This module handles both eye-in-hand and eye-to-hand calibration scenarios.
It provides a unified interface for hand-eye calibration with automatic
handling of different calibration types and coordinate transformations.

Key Methods in HandEyeCalibrator class:
- Unified constructor: Initialize with calibration type and transformation matrices
- calibrate(): Perform hand-eye calibration using the specified method
- get_transformation(): Get the calibrated hand-eye transformation matrix

The class supports both calibration scenarios:
1. Eye-in-hand: Camera mounted on robot end-effector
2. Eye-to-hand: Camera mounted externally observing the robot

Usage:
    # Eye-in-hand calibration
    calibrator = HandEyeCalibrator(
        calibration_type="eye_in_hand",
        image_paths=paths,
        end2base_matrices=transforms,
        calibration_pattern=pattern
    )
    
    # Eye-to-hand calibration
    calibrator = HandEyeCalibrator(
        calibration_type="eye_to_hand", 
        image_paths=paths,
        end2base_matrices=transforms,
        calibration_pattern=pattern
    )
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Union, Literal
from .base_calibrator import BaseCalibrator
from .calibration_patterns import CalibrationPattern


class HandEyeCalibrator(BaseCalibrator):
    """
    Unified hand-eye calibration class for both eye-in-hand and eye-to-hand scenarios.
    
    This class inherits common functionality from BaseCalibrator and specializes in
    hand-eye calibration, automatically handling different calibration types and
    coordinate frame transformations.
    
    Hand-eye specific attributes:
        calibration_type (str): Type of calibration ("eye_in_hand" or "eye_to_hand")
        end2base_matrices (List[np.ndarray]): End-effector to base transformation matrices
        
    Inherited from BaseCalibrator:
        camera_matrix, distortion_coefficients, distortion_model: Camera intrinsic parameters
        rvecs, tvecs: Extrinsic parameters for each image
        rms_error, per_image_errors: Calibration quality metrics
        calibration_completed: Calibration status
    """
    
    def __init__(self, 
                 calibration_type: Literal["eye_in_hand", "eye_to_hand"],
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None, 
                 calibration_pattern: Optional[CalibrationPattern] = None):
        """
        Initialize HandEyeCalibrator with unified interface for both calibration types.
        
        Args:
            calibration_type: Type of calibration, must be "eye_in_hand" or "eye_to_hand"
            images: List of image arrays (numpy arrays) or None
            end2base_matrices: List of 4x4 transformation matrices from end-effector to base
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            
        Raises:
            ValueError: If calibration_type is not "eye_in_hand" or "eye_to_hand"
            
        Constructor Behavior:
            • Only image_paths provided: Automatically loads end2base matrices from JSON files
            • Only end2base_matrices provided: Uses provided matrices (no image loading)
            • Both provided: Uses provided end2base_matrices and loads images (JSON ignored)
            • Neither provided: Creates empty calibrator (use setters to add data later)
            
        Note:
            For eye-in-hand: Camera is mounted on robot end-effector
            For eye-to-hand: Camera is mounted externally observing robot
            
            end2base_matrices should contain 4x4 homogeneous transformation matrices
            representing the pose of the robot end-effector relative to the base frame.
            
            When both image_paths and end2base_matrices are provided, the provided 
            end2base_matrices take precedence and JSON files are NOT loaded automatically.
            Use set_images_from_paths() explicitly if you want to load from JSON files.
        """
        # Validate calibration type first
        if calibration_type not in ["eye_in_hand", "eye_to_hand"]:
            raise ValueError(f"calibration_type must be 'eye_in_hand' or 'eye_to_hand', got '{calibration_type}'")
        
        # Hand-eye specific attributes
        self.calibration_type = calibration_type
        self.end2base_matrices = end2base_matrices
        
        # Handle special case: if both image_paths and end2base_matrices are provided,
        # don't automatically load from JSON files to avoid overwriting the provided matrices
        if image_paths is not None and end2base_matrices is not None:
            # Initialize base class WITHOUT calling set_images_from_paths automatically
            # We'll handle image loading manually to preserve the provided end2base_matrices
            super().__init__(images=None, image_paths=None, calibration_pattern=calibration_pattern)
            
            # Set images manually using base class method to avoid JSON loading
            success = super().set_images_from_paths(image_paths)
            if not success:
                raise ValueError("Failed to load images from provided paths")
                
            print(f"ℹ️  Loaded {len(self.images)} images from paths, using provided end2base_matrices")
            print(f"   (JSON files were not loaded to preserve provided transformation matrices)")
            
        else:
            # Standard initialization - let base class handle image loading
            super().__init__(images, image_paths, calibration_pattern)
        
        # Validation of input consistency
        self._validate_input_consistency()
    
    def _validate_input_consistency(self) -> None:
        """
        Validate that input data is consistent across images and transformation matrices.
        
        Raises:
            ValueError: If data dimensions are inconsistent
        """
        # Check that we have some way to get images
        if self.images is None and self.image_paths is None:
            # This is okay - images can be set later, but still validate transformation matrices
            if self.end2base_matrices is not None:
                self._validate_transformation_matrices()
            return
            
        # Check consistency between images and transformation matrices
        if self.end2base_matrices is not None:
            if self.images is not None:
                if len(self.images) != len(self.end2base_matrices):
                    raise ValueError(f"Number of images ({len(self.images)}) must match "
                                   f"number of transformation matrices ({len(self.end2base_matrices)})")
            
            if self.image_paths is not None:
                if len(self.image_paths) != len(self.end2base_matrices):
                    raise ValueError(f"Number of image paths ({len(self.image_paths)}) must match "
                                   f"number of transformation matrices ({len(self.end2base_matrices)})")
            
            # Validate transformation matrix format
            self._validate_transformation_matrices()
    
    def _validate_transformation_matrices(self) -> None:
        """
        Validate the format and content of transformation matrices.
        
        Raises:
            ValueError: If matrices have invalid format
        """
        if self.end2base_matrices is None:
            return
            
        for i, matrix in enumerate(self.end2base_matrices):
            if matrix is None:
                continue
                
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"Transformation matrix {i} must be a numpy array")
            
            if matrix.shape != (4, 4):
                raise ValueError(f"Transformation matrix {i} must be 4x4, got shape {matrix.shape}")
            
            # Check if it looks like a valid transformation matrix
            if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
                print(f"Warning: Transformation matrix {i} bottom row is not [0, 0, 0, 1]")
    
    def set_images_from_paths(self, image_paths: List[str]) -> bool:
        """
        Set images from file paths and read corresponding JSON files with end2base matrices.
        
        For each image file, this method will:
        1. Load the image file
        2. Look for a JSON file with the same name (e.g., image.jpg -> image.json)
        3. Extract the "end2base" matrix from the JSON file
        
        Data is only valid if ALL images and corresponding JSON files are successfully read.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            bool: True if all images and JSON files loaded successfully
        """
        if not image_paths:
            print("Error: No image paths provided")
            return False
        
        try:
            images = []
            end2base_matrices = []
            valid_paths = []
            
            print(f"Loading {len(image_paths)} images and corresponding JSON files...")
            
            for i, img_path in enumerate(image_paths):
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not load image {img_path}")
                    return False
                
                # Construct JSON file path (same name, different extension)
                base_name = os.path.splitext(img_path)[0]  # Remove extension
                json_path = base_name + '.json'
                
                # Check if JSON file exists
                if not os.path.exists(json_path):
                    print(f"Error: JSON file not found: {json_path}")
                    return False
                
                # Load and parse JSON file
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Extract end2base matrix
                    if 'end2base' not in json_data:
                        print(f"Error: 'end2base' key not found in {json_path}")
                        return False
                    
                    end2base = json_data['end2base']
                    
                    # Convert to numpy array and validate
                    end2base_matrix = np.array(end2base, dtype=np.float64)
                    
                    if end2base_matrix.shape != (4, 4):
                        print(f"Error: end2base matrix in {json_path} is not 4x4, got shape {end2base_matrix.shape}")
                        return False
                    
                    # Validate that it looks like a proper transformation matrix
                    if not np.allclose(end2base_matrix[3, :], [0, 0, 0, 1], atol=1e-6):
                        print(f"Warning: end2base matrix in {json_path} bottom row is not [0, 0, 0, 1]: {end2base_matrix[3, :]}")
                        # Don't return False here - just warn, as some matrices might have slight numerical errors
                    
                    # If we get here, both image and JSON loaded successfully
                    images.append(img)
                    end2base_matrices.append(end2base_matrix)
                    valid_paths.append(img_path)
                    
                    print(f"✅ Loaded image {i+1}/{len(image_paths)}: {os.path.basename(img_path)} with transform")
                    
                except json.JSONDecodeError as e:
                    print(f"Error: Could not parse JSON file {json_path}: {e}")
                    return False
                except Exception as e:
                    print(f"Error: Could not load JSON file {json_path}: {e}")
                    return False
            
            # If we get here, all images and JSON files were loaded successfully
            self.images = images
            self.image_paths = valid_paths
            self.end2base_matrices = end2base_matrices
            
            # Set image size from first image
            if self.images:
                h, w = self.images[0].shape[:2]
                self.image_size = (w, h)
            
            # Initialize filename manager for systematic duplicate handling
            from .utils import FilenameManager
            self.filename_manager = FilenameManager(valid_paths)
            
            print(f"✅ Successfully loaded {len(self.images)} images with end2base matrices")
            print(f"📏 Image size: {self.image_size}")
            print(f"🤖 Calibration type: {self.calibration_type}")
            
            # Validate consistency of loaded data
            self._validate_input_consistency()
            
            return True
            
        except Exception as e:
            print(f"Error loading images and transformations: {e}")
            return False
    
    def set_end2base_matrices(self, matrices: List[np.ndarray]) -> None:
        """
        Set the end-effector to base transformation matrices.
        
        Args:
            matrices: List of 4x4 transformation matrices
            
        Raises:
            ValueError: If matrices have wrong format or inconsistent dimensions
        """
        self.end2base_matrices = matrices
        self._validate_input_consistency()
    
    def get_calibration_info(self) -> dict:
        """
        Get comprehensive calibration information.
        
        Returns:
            dict: Dictionary containing calibration type, status, and parameters
        """
        return {
            "calibration_type": self.calibration_type,
            "calibration_completed": self.calibration_completed,
            "has_intrinsics": self.camera_matrix is not None,
            "has_extrinsics": self.rvecs is not None and self.tvecs is not None,
            "image_count": len(self.images) if self.images else 0,
            "transform_count": len(self.end2base_matrices) if self.end2base_matrices else 0,
            "pattern_type": self.calibration_pattern.pattern_id if self.calibration_pattern else None
        }
    
    # Abstract method implementations
    def calibrate(self, **kwargs) -> bool:
        """
        Perform hand-eye calibration using the specified method.
        
        This method will:
        1. Perform intrinsic calibration if not already done
        2. Detect calibration pattern poses relative to camera
        3. Solve the hand-eye calibration problem
        
        Args:
            **kwargs: Calibration method parameters
            
        Returns:
            bool: True if calibration succeeded, False if failed
            
        Note:
            This is a placeholder - specific calibration implementation will be added
            in subsequent steps.
        """
        # TODO: Implement calibration logic
        # This will be implemented in following steps:
        # 1. Intrinsic calibration (if needed)
        # 2. Pattern pose estimation
        # 3. Hand-eye transformation solving
        raise NotImplementedError("Calibration implementation will be added in next steps")
    
    def save_results(self, save_directory: str) -> None:
        """
        Save hand-eye calibration results to files.
        
        Args:
            save_directory: Directory to save results
        """
        if not self.is_calibrated():
            raise ValueError("No calibration results to save. Run calibration first.")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Use the new to_json serialization method
        calibration_data = self.to_json()
        
        # Add hand-eye specific data
        calibration_data["calibration_type"] = self.calibration_type
        
        # Save to JSON file
        filepath = os.path.join(save_directory, "hand_eye_calibration_results.json")
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"✅ Hand-eye calibration results saved to: {filepath}")
    
    def is_calibrated(self) -> bool:
        """
        Check if hand-eye calibration has been completed successfully.
        
        Returns:
            bool: True if both intrinsic and extrinsic calibration are complete
        """
        return (self.calibration_completed and 
                self.camera_matrix is not None and 
                self.rvecs is not None and 
                self.tvecs is not None)
    
    def __str__(self) -> str:
        """String representation of the calibrator."""
        status = "calibrated" if self.is_calibrated() else "not calibrated"
        return f"HandEyeCalibrator(type={self.calibration_type}, status={status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"HandEyeCalibrator(calibration_type='{self.calibration_type}', "
                f"images={len(self.images) if self.images else 0}, "
                f"transforms={len(self.end2base_matrices) if self.end2base_matrices else 0}, "
                f"calibrated={self.is_calibrated()})")
