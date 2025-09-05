"""
Hand-Eye Base Calibration Module
===============================

This module provides the base class for hand-eye calibration operations.
It contains common IO functionality shared between eye-in-hand and eye-to-hand calibration types.

HandEyeBaseCalibrator is an abstract base class that defines:
- Common data structures for robot poses and transformations
- Common IO methods for robot pose handling and data management
- Input validation for hand-eye calibration data

This design eliminates code duplication between eye-in-hand and eye-to-hand calibrators
while providing a consistent interface for data handling.

Key Design Principles:
- Inherits from BaseCalibrator for common image/pattern functionality
- Adds robot-specific data structures (poses, transformations)
- Provides common validation and IO methods
- Separates data handling from calibration logic
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
from .base_calibrator import BaseCalibrator
from .calibration_patterns import CalibrationPattern


class HandEyeBaseCalibrator(BaseCalibrator):
    """
    Abstract base class for hand-eye calibration data handling.
    
    This class provides common IO functionality for both eye-in-hand and eye-to-hand calibration:
    - Robot pose data management
    - Transformation matrix IO and validation
    - Common data structures and validation
    
    Specialized calibrators inherit from this class and implement calibration logic.
    """
    
    def __init__(self, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None, 
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize HandEyeBaseCalibrator with unified interface for hand-eye calibration.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            end2base_matrices: List of 4x4 transformation matrices from end-effector to base
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None (if None, will be calibrated)
            distortion_coefficients: Camera distortion coefficients or None (if None, will be calibrated)
            
        Constructor Behavior:
            • Only image_paths provided: Automatically loads end2base matrices from JSON files
            • Only end2base_matrices provided: Uses provided matrices (no image loading)
            • Both provided: Uses provided end2base_matrices and loads images (JSON ignored)
            • Neither provided: Creates empty calibrator (use setters to add data later)
            
        Note:
            end2base_matrices should contain 4x4 homogeneous transformation matrices
            representing the pose of the robot end-effector relative to the base frame.
            
            If camera_matrix and distortion_coefficients are provided, intrinsic calibration
            will be skipped and the provided parameters will be used directly.
            
            When both image_paths and end2base_matrices are provided, the provided 
            end2base_matrices take precedence and JSON files are NOT loaded automatically.
            Use set_images_from_paths() explicitly if you want to load from JSON files.
        """
        # Robot pose data (common to both eye-in-hand and eye-to-hand)
        self.end2base_matrices = end2base_matrices
        
        # Target to camera transformation matrices (computed from rvec/tvec)
        self.target2cam_matrices = None
        
        # Calibration result attributes
        self.best_method = None
        self.best_method_name = None
        
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
        
        # Set camera intrinsics if provided
        if camera_matrix is not None:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        if distortion_coefficients is not None:
            self.distortion_coefficients = np.array(distortion_coefficients, dtype=np.float32)
            
        # Validate camera matrix if provided
        if self.camera_matrix is not None:
            if self.camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix must be 3x3, got shape {self.camera_matrix.shape}")
                
        # Validate distortion coefficients if provided
        if self.distortion_coefficients is not None:
            if len(self.distortion_coefficients.shape) != 1 or self.distortion_coefficients.shape[0] < 4:
                raise ValueError(f"distortion_coefficients must be a 1D array with at least 4 elements, got shape {self.distortion_coefficients.shape}")
        
        # Validation of input consistency
        self._validate_input_consistency()
    
    @abstractmethod
    def calibrate(self, method: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform hand-eye calibration using the specified method or find the best method.
        
        This is the main calibration interface that should be implemented by inheriting classes.
        The function provides a unified interface for both eye-in-hand and eye-to-hand calibration.
        
        Args:
            method: Optional OpenCV calibration method constant. If None, all methods will be 
                   tested and the best one (lowest reprojection error) will be selected.
                   Valid methods depend on calibration type:
                   - Eye-in-hand: cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, 
                                  cv2.CALIB_HAND_EYE_HORAUD, cv2.CALIB_HAND_EYE_ANDREFF, 
                                  cv2.CALIB_HAND_EYE_DANIILIDIS
                   - Eye-to-hand: Same methods but different transformation relationships
            verbose: Whether to print detailed calibration progress and results
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing calibration results if successful, None if failed.
            The result dictionary should contain at minimum:
            - 'success': bool - True if calibration succeeded
            - 'method': int - OpenCV method constant used
            - 'method_name': str - Human-readable method name
            - 'rms_error': float - Overall RMS reprojection error
            - 'per_image_errors': List[float] - Per-image reprojection errors
            - 'valid_images': int - Number of valid images used in calibration
            - 'total_images': int - Total number of images processed
            Additional keys specific to eye-in-hand or eye-to-hand should be included.
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
            ValueError: If required data is missing or invalid
            
        Note:
            Before calling this method, ensure that:
            1. Images and robot poses are loaded
            2. Camera intrinsic parameters are available (via intrinsic calibration)
            3. Calibration patterns are detected in images
            4. Target-to-camera matrices are calculated
            
            After successful calibration:
            - self.calibration_completed will be True
            - self.best_method and self.best_method_name will contain the best method info
            - Transformation matrices will be available via getter methods
            - RMS error and per-image errors will be calculated
        """
        raise NotImplementedError("calibrate() method must be implemented by subclasses")

    def to_json(self) -> dict:
        """
        Serialize hand-eye calibrator state to JSON-compatible dictionary.
        
        Extends BaseCalibrator.to_json() to include hand-eye specific data:
        - end2base_matrices: Robot end-effector to base transformation matrices
        - target2cam_matrices: Target to camera transformation matrices  
        - best_method: Best calibration method used
        - best_method_name: Name of best calibration method
        
        Returns:
            dict: JSON-compatible dictionary containing complete calibrator state
        """
        # Get base class data
        data = super().to_json()
        
        # Add hand-eye specific data
        if self.end2base_matrices is not None:
            data['end2base_matrices'] = []
            for matrix in self.end2base_matrices:
                if matrix is not None:
                    data['end2base_matrices'].append(matrix.tolist())
                else:
                    data['end2base_matrices'].append(None)
        
        if self.target2cam_matrices is not None:
            data['target2cam_matrices'] = []
            for matrix in self.target2cam_matrices:
                if matrix is not None:
                    data['target2cam_matrices'].append(matrix.tolist())
                else:
                    data['target2cam_matrices'].append(None)
        
        if self.best_method is not None:
            data['best_method'] = int(self.best_method)
            
        if self.best_method_name is not None:
            data['best_method_name'] = str(self.best_method_name)
        
        return data

    def from_json(self, data: dict) -> None:
        """
        Deserialize hand-eye calibrator state from JSON-compatible dictionary.
        
        Extends BaseCalibrator.from_json() to load hand-eye specific data:
        - end2base_matrices: Robot end-effector to base transformation matrices
        - target2cam_matrices: Target to camera transformation matrices
        - best_method: Best calibration method used  
        - best_method_name: Name of best calibration method
        
        Args:
            data: JSON-compatible dictionary containing calibrator state
        """
        # Load base class data first
        super().from_json(data)
        
        # Load hand-eye specific data
        if 'end2base_matrices' in data:
            self.end2base_matrices = []
            for matrix_data in data['end2base_matrices']:
                if matrix_data is not None:
                    self.end2base_matrices.append(np.array(matrix_data, dtype=np.float32))
                else:
                    self.end2base_matrices.append(None)
        
        if 'target2cam_matrices' in data:
            self.target2cam_matrices = []
            for matrix_data in data['target2cam_matrices']:
                if matrix_data is not None:
                    self.target2cam_matrices.append(np.array(matrix_data, dtype=np.float32))
                else:
                    self.target2cam_matrices.append(None)
        
        if 'best_method' in data:
            self.best_method = int(data['best_method'])
            
        if 'best_method_name' in data:
            self.best_method_name = str(data['best_method_name'])

    @abstractmethod
    def save_results(self, save_directory: str) -> None:
        """
        Save hand-eye calibration results to files.
        Must be implemented by subclasses.
        
        Args:
            save_directory: Directory to save results
        """
        pass

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
            
            # Validate consistency of loaded data
            self._validate_input_consistency()
            
            return True
            
        except Exception as e:
            print(f"Error loading images and transformations: {e}")
            return False

    def set_images_from_arrays(self, images: List[np.ndarray]) -> bool:
        """
        Set images from numpy arrays.
        
        Args:
            images: List of image arrays
            
        Returns:
            bool: True if images set successfully
        """
        self.images = images
        if images:
            h, w = images[0].shape[:2]
            self.image_size = (w, h)
            
        print(f"Set {len(images)} images from arrays")
        return True

    def set_calibration_pattern(self, pattern: CalibrationPattern):
        """
        Set calibration pattern and related parameters.
        
        Args:
            pattern: CalibrationPattern instance
        """
        self.calibration_pattern = pattern

    def detect_pattern_points(self, verbose: bool = False) -> bool:
        """
        Detect calibration pattern points in all images using modern pattern system.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            bool: True if pattern detection completed successfully
        """
        if self.images is None:
            print("Error: No images loaded")
            return False
            
        if self.calibration_pattern is None:
            raise ValueError("Calibration pattern must be set first")
        
        # Initialize arrays aligned with image count - maintain 1:1 correspondence
        num_images = len(self.images)
        self.image_points = [None] * num_images      # Image points for each image (None if failed)
        self.point_ids = [None] * num_images         # Point IDs for each image (None if failed)
        self.object_points = [None] * num_images     # Object points for each image (None if failed)
        successful_detections = 0
        
        if verbose:
            print(f"Detecting patterns in {len(self.images)} images...")
        
        for i, img in enumerate(self.images):
            success, img_pts, point_ids = self.calibration_pattern.detect_corners(img)
            
            if success:
                # Ensure proper data types and formats for OpenCV calibration
                img_pts = np.array(img_pts, dtype=np.float32)
                
                # For ArUco patterns, ensure proper array shape for calibration
                if hasattr(self.calibration_pattern, 'pattern_id') and self.calibration_pattern.pattern_id == 'grid_board':
                    # Grid board returns [N, 2] corners, need [N, 1, 2] for calibration
                    if len(img_pts.shape) == 2 and img_pts.shape[1] == 2:
                        img_pts = img_pts.reshape(-1, 1, 2)
                
                # Store data at the same index as the image (maintaining alignment)
                self.image_points[i] = img_pts
                self.point_ids[i] = point_ids
                
                # Generate corresponding object points
                if self.calibration_pattern.is_planar:
                    obj_pts = self.calibration_pattern.generate_object_points(point_ids)
                else:
                    obj_pts = self.calibration_pattern.generate_object_points()
                
                # Ensure proper data type for object points
                obj_pts = np.array(obj_pts, dtype=np.float32)
                self.object_points[i] = obj_pts
                successful_detections += 1
                
                if verbose:
                    print(f"Image {i}: ✅ Detected {len(img_pts)} features")
            else:
                # Keep None for failed detections (maintains array alignment)
                self.image_points[i] = None
                self.point_ids[i] = None
                self.object_points[i] = None
                
                if verbose:
                    print(f"Image {i}: ❌ No pattern detected")
        
        if successful_detections < 3:
            print(f"Insufficient detections: need at least 3, got {successful_detections}")
            return False
        
        if verbose:
            print(f"Successfully detected pattern in {successful_detections}/{len(self.images)} images")
        
        return True

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

    # ============================================================================
    # Hand-Eye Specific Methods
    # ============================================================================

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

    def set_camera_intrinsics(self, camera_matrix: np.ndarray, 
                              distortion_coefficients: np.ndarray) -> None:
        """
        Set camera intrinsic parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coefficients: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

    def _validate_calibration_prerequisites(self) -> None:
        """
        Validate that all prerequisites for calibration are met.
        
        Raises:
            ValueError: If required data is missing or invalid
        """
        # Check basic data availability
        if self.images is None or len(self.images) == 0:
            raise ValueError("No images loaded. Load images before calibration.")
            
        if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
            raise ValueError("No robot poses loaded. Load end2base matrices before calibration.")
            
        if len(self.images) != len(self.end2base_matrices):
            raise ValueError(f"Image count ({len(self.images)}) must match pose count ({len(self.end2base_matrices)})")
        
        # Check intrinsic calibration
        if self.camera_matrix is None:
            raise ValueError("Camera intrinsic matrix not available. Perform intrinsic calibration first.")
            
        if self.distortion_coefficients is None:
            raise ValueError("Camera distortion coefficients not available. Perform intrinsic calibration first.")
        
        # Check pattern detection and pose calculation
        if not hasattr(self, 'image_points') or self.image_points is None:
            raise ValueError("Pattern points not detected. Run detect_pattern_points() first.")
            
        if not hasattr(self, 'target2cam_matrices') or self.target2cam_matrices is None:
            raise ValueError("Target-to-camera matrices not calculated. Run _calculate_target2cam_matrices() first.")
        
        # Count valid data points
        valid_points = sum(1 for pts in self.image_points if pts is not None)
        valid_matrices = sum(1 for matrix in self.target2cam_matrices if matrix is not None)
        
        if valid_points < 3:
            raise ValueError(f"Need at least 3 images with detected patterns, got {valid_points}")
            
        if valid_matrices < 3:
            raise ValueError(f"Need at least 3 valid target2cam matrices, got {valid_matrices}")

    def _calculate_target2cam_matrices(self, verbose: bool = False) -> None:
        """
        Calculate target-to-camera transformation matrices for all detected calibration patterns.
        
        This function attempts to calculate poses for all images with detected patterns and
        converts them to 4x4 transformation matrices representing the pose of the calibration
        target (pattern) relative to the camera coordinate system.
        
        Args:
            verbose: Whether to print detailed information about pose calculation
            
        Note:
            Results are stored in:
            - self.rvecs and self.tvecs arrays (rotation vectors and translation vectors)
            - self.target2cam_matrices array (4x4 transformation matrices)
            All arrays are aligned with self.images. None values indicate failed calculations.
            
            Pose calculation attempts solvePnP for each detected pattern and validates:
            - solvePnP success (ret == True)
            - No NaN or infinite values in rvec/tvec
            - Non-zero pose magnitudes (filters out degenerate solutions)
        """
        if verbose:
            print(f"📐 Calculating target-to-camera matrices for all detected patterns...")
        
        # Initialize pose arrays aligned with images
        self.rvecs = [None] * len(self.images)
        self.tvecs = [None] * len(self.images)
        self.target2cam_matrices = [None] * len(self.images)
        
        successful_poses = 0
        
        for i in range(len(self.images)):
            if (self.image_points[i] is not None and 
                self.object_points[i] is not None):
                
                # Try to calculate pose from detected pattern points
                try:
                    ret, rvec, tvec = cv2.solvePnP(
                        self.object_points[i], 
                        self.image_points[i], 
                        self.camera_matrix, 
                        self.distortion_coefficients
                    )
                    
                    if ret and rvec is not None and tvec is not None:
                        # Check if pose is reasonable (not NaN or infinite)
                        if (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)) and
                            np.linalg.norm(rvec) > 1e-6 and np.linalg.norm(tvec) > 1e-6):
                            self.rvecs[i] = rvec
                            self.tvecs[i] = tvec
                            
                            # Convert rvec and tvec to 4x4 transformation matrix
                            rotation_matrix, _ = cv2.Rodrigues(rvec)
                            target2cam_matrix = np.eye(4, dtype=np.float32)
                            target2cam_matrix[:3, :3] = rotation_matrix
                            target2cam_matrix[:3, 3] = tvec.flatten()
                            self.target2cam_matrices[i] = target2cam_matrix
                            
                            successful_poses += 1
                            
                            if verbose:
                                print(f"   ✅ Image {i}: Valid target2cam matrix calculated")
                        elif verbose:
                            print(f"   ⚠️  Image {i}: Invalid pose calculated (NaN or unreasonable values)")
                    elif verbose:
                        print(f"   ❌ Image {i}: solvePnP failed")
                        
                except Exception as e:
                    if verbose:
                        print(f"   ❌ Image {i}: Pose calculation failed - {e}")
                    continue
            elif verbose:
                print(f"   ⚪ Image {i}: No pattern detected")
        
        if verbose:
            failed_poses = len(self.images) - successful_poses
            print(f"📊 Target2Cam Matrix Calculation Summary:")
            print(f"   • Successful matrices: {successful_poses}")
            print(f"   • Failed calculations: {failed_poses}")
            print(f"   • Total images: {len(self.images)}")

    # ============================================================================
    # Result Access Methods
    # ============================================================================

    def get_calibration_info(self) -> dict:
        """
        Get comprehensive calibration information.
        
        Returns:
            dict: Dictionary containing calibration type, status, and parameters
        """
        return {
            "calibration_completed": self.calibration_completed,
            "has_intrinsics": self.camera_matrix is not None,
            "has_extrinsics": self.rvecs is not None and self.tvecs is not None,
            "image_count": len(self.images) if self.images else 0,
            "transform_count": len(self.end2base_matrices) if self.end2base_matrices else 0,
            "pattern_type": self.calibration_pattern.pattern_id if self.calibration_pattern else None
        }

    def get_best_method(self) -> int:
        """
        Get the OpenCV method constant that produced the best calibration results.
        
        Returns:
            int: OpenCV CALIB_HAND_EYE_* constant for best method
            
        Raises:
            ValueError: If calibration has not been completed
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        return self.best_method

    def get_best_method_name(self) -> str:
        """
        Get the human-readable name of the method that produced the best results.
        
        Returns:
            str: Name of the best calibration method (e.g., "TSAI", "PARK", etc.)
            
        Raises:
            ValueError: If calibration has not been completed
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        return self.best_method_name

    def get_rms_error(self) -> Optional[float]:
        """Get overall RMS reprojection error."""
        return self.rms_error

    def get_per_image_errors(self) -> Optional[List[float]]:
        """Get per-image reprojection errors."""
        return self.per_image_errors

    @staticmethod
    def get_available_methods() -> Dict[int, str]:
        """
        Get all available OpenCV hand-eye calibration methods.
        
        Returns:
            dict: Mapping of OpenCV method constants to their human-readable names
        """
        return {
            cv2.CALIB_HAND_EYE_TSAI: "TSAI",
            cv2.CALIB_HAND_EYE_PARK: "PARK", 
            cv2.CALIB_HAND_EYE_HORAUD: "HORAUD",
            cv2.CALIB_HAND_EYE_ANDREFF: "ANDREFF",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "DANIILIDIS"
        }

    @staticmethod
    def get_method_name(method: int) -> str:
        """
        Get human-readable name for an OpenCV calibration method constant.
        
        Args:
            method: OpenCV method constant (e.g., cv2.CALIB_HAND_EYE_TSAI)
            
        Returns:
            str: Human-readable method name (e.g., "TSAI")
        """
        methods = HandEyeBaseCalibrator.get_available_methods()
        return methods.get(method, f"Unknown method ({method})")
