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
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None):
        """
        Initialize HandEyeCalibrator with unified interface for both calibration types.
        
        Args:
            calibration_type: Type of calibration, must be "eye_in_hand" or "eye_to_hand"
            images: List of image arrays (numpy arrays) or None
            end2base_matrices: List of 4x4 transformation matrices from end-effector to base
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None (if None, will be calibrated)
            distortion_coefficients: Camera distortion coefficients or None (if None, will be calibrated)
            
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
            
            If camera_matrix and distortion_coefficients are provided, intrinsic calibration
            will be skipped and the provided parameters will be used directly.
            
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
        
        # calibration target
        if calibration_type == "eye_in_hand":
            self.cam2end_matrix = None
            self.target2base_matrix = None
        else:
            self.cam2base_matrix = None
            self.target2end_matrix = None

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
        
        # Initialize calibration result attributes
        self.best_method = None
        self.best_method_name = None
    
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
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the transformation matrix from the calibration.
        
        Returns:
            np.ndarray: 4x4 transformation matrix (base2cam for both eye-in-hand and eye-to-hand)
            
        Raises:
            ValueError: If calibration has not been completed
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        return self.base2cam_matrix
    
    def get_rms_error(self) -> float:
        """
        Get the RMS reprojection error from calibration.
        
        Returns:
            float: RMS reprojection error in pixels
            
        Raises:
            ValueError: If calibration has not been completed
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        return self.rms_error
    
    def get_per_image_errors(self) -> List[float]:
        """
        Get the reprojection errors for each image.
        
        Returns:
            List[float]: Per-image reprojection errors in pixels
            
        Raises:
            ValueError: If calibration has not been completed
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run calibrate() first.")
        return self.per_image_errors.copy() if self.per_image_errors else []
    
    # Abstract method implementations
    def calibrate(self, verbose: bool = False) -> bool:
        """
        Perform hand-eye calibration using the best available method.
        
        This method automatically tests all available OpenCV hand-eye calibration methods
        and selects the one that produces the smallest reprojection error.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False if failed
            
        Note:
            After successful calibration, use getter methods to access results:
            - get_rms_error(): Overall RMS reprojection error  
            - get_transformation_matrix(): Transformation matrix (base2cam or cam2base)
            - get_per_image_errors(): Per-image reprojection errors
            - get_best_method(): The OpenCV method that produced the best results
        """
        # Step 1: Validate input data
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters have not been loaded")
            
        if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
            raise ValueError("Robot poses have not been set")
            
        if self.images is None or len(self.images) == 0:
            raise ValueError("Images have not been loaded")
            
        if self.calibration_pattern is None:
            raise ValueError("Calibration pattern has not been set")
            
        if len(self.images) != len(self.end2base_matrices):
            raise ValueError(f"Mismatch: {len(self.images)} images vs {len(self.end2base_matrices)} robot poses")
        
        # Step 2: Detect calibration patterns in all images
        if verbose:
            print(f"🔍 Detecting calibration patterns in {len(self.images)} images...")
            
        success = self.detect_pattern_points(verbose=verbose)
        if not success:
            raise ValueError("Pattern detection failed or insufficient patterns detected")
        
        # Step 2.5: Calculate poses for all detected patterns
        # This populates self.rvecs and self.tvecs with None for failed calculations
        self._calculate_poses_for_all_images(verbose=verbose)
        
        # Step 3: Extract valid data for hand-eye calibration
        # Only images with successful pattern detection AND pose calculation
        valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices = self._extract_valid_calibration_data(verbose=verbose)
        
        # Store the valid data for use by calibration methods
        self._valid_calibration_indices = valid_indices
        self._valid_calibration_rvecs = valid_rvecs
        self._valid_calibration_tvecs = valid_tvecs
        self._valid_end2base_matrices = valid_end2base_matrices
        
        # Available OpenCV hand-eye calibration methods
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
            (cv2.CALIB_HAND_EYE_PARK, "PARK"),
            (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
        ]
        
        if verbose:
            print(f"\n🧪 Testing {len(methods)} hand-eye calibration methods to find the best one...")
            print(f"Calibration type: {self.calibration_type}")
        
        best_method = None
        best_method_name = ""
        best_rms_error = float('inf')
        best_results = None
        
        # Step 4: Test each method and find the one with smallest reprojection error
        for method, method_name in methods:
            try:
                if verbose:
                    print(f"\n--- Testing method: {method_name} ---")
                
                # Run calibration with this method
                if self.calibration_type == "eye_in_hand":
                    result = self._calibrate_eye_in_hand(method, verbose)
                elif self.calibration_type == "eye_to_hand":
                    result = self._calibrate_eye_to_hand(method, verbose)
                else:
                    raise ValueError(f"Unknown calibration type: {self.calibration_type}")
                
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
            self.rms_error = best_results['rms_error']
            self.per_image_errors = best_results['per_image_errors']
            
            # Assign results based on calibration type to match existing naming
            if self.calibration_type == "eye_in_hand":
                self.cam2end_matrix = best_results.get('base2cam_matrix')  # For eye-in-hand, this is actually end2cam inverted
                self.target2base_matrix = best_results.get('target2end_matrix')
            else:
                self.base2cam_matrix = best_results.get('base2cam_matrix')
                self.target2end_matrix = best_results.get('target2end_matrix')
            
            # Set up rvecs and tvecs aligned with all images (None for failed detections)
            # This ensures compatibility with base class visualization methods
            self.rvecs = [None] * len(self.images)
            self.tvecs = [None] * len(self.images)
            
            # Get the valid indices and rvecs/tvecs from the best calibration result
            if 'valid_indices' in best_results and 'rvecs' in best_results and 'tvecs' in best_results:
                valid_indices = best_results['valid_indices']
                valid_rvecs = best_results['rvecs']
                valid_tvecs = best_results['tvecs']
                
                # Map the valid rvecs/tvecs back to their original image positions
                for i, (valid_idx, rvec, tvec) in enumerate(zip(valid_indices, valid_rvecs, valid_tvecs)):
                    self.rvecs[valid_idx] = rvec
                    self.tvecs[valid_idx] = tvec
                    
            self.calibration_completed = True
            self.best_method = best_method
            self.best_method_name = best_method_name
            
            if verbose:
                print(f"\n🎯 Final result: Best method is {best_method_name}")
                print(f"   RMS reprojection error: {self.rms_error:.4f} pixels")
                print(f"   Calibration completed successfully!")
            
            return True
        else:
            if verbose:
                print("\n❌ All calibration methods failed!")
            return False
    
    def _calibrate_eye_in_hand(self, method: int, verbose: bool = False) -> dict:
        """
        Perform eye-in-hand calibration using specified method.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            dict: Calibration results or None if failed
        """
        try:
            # Get valid data directly from stored calibration data
            if not hasattr(self, '_valid_calibration_indices'):
                raise ValueError("Valid calibration data not found. Call calibrate() first.")
                
            valid_indices = self._valid_calibration_indices
            rvecs = self._valid_calibration_rvecs
            tvecs = self._valid_calibration_tvecs
            valid_end2base_matrices = self._valid_end2base_matrices
            
            if verbose:
                print(f"Using pre-extracted valid data: {len(valid_indices)} valid images")
            
            # Create arrays for valid detections only
            valid_image_points = [self.image_points[i] for i in valid_indices]
            valid_object_points = [self.object_points[i] for i in valid_indices]
            
            # Prepare data for OpenCV calibrateHandEye (eye-in-hand uses end2base transformations)
            end2base_Rs = np.array([matrix[:3, :3] for matrix in valid_end2base_matrices])
            end2base_ts = np.array([matrix[:3, 3] for matrix in valid_end2base_matrices])
            rvecs_array = np.array([rvec for rvec in rvecs])
            tvecs_array = np.array([tvec for tvec in tvecs])
            
            if verbose:
                print(f"Using end2base transformations (eye-in-hand configuration)")
                print(f"Working with {len(valid_indices)} valid detections")
            
                # Perform eye-in-hand calibration using OpenCV
            cam2end_R, cam2end_t = cv2.calibrateHandEye(
                end2base_Rs, end2base_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2end 4x4 transformation matrix
            cam2end_4x4 = np.eye(4)
            cam2end_4x4[:3, :3] = cam2end_R
            cam2end_4x4[:3, 3] = cam2end_t[:, 0]

            # For eye-in-hand, we need base2cam matrix, so invert the result
            end2cam_4x4 = np.linalg.inv(cam2end_4x4)
            
            # Calculate the optimal target2end matrix that minimizes reprojection error
            target2end_matrix = self._calculate_optimal_target2end_matrix(end2cam_4x4, rvecs, tvecs, verbose)
            
            # Calculate reprojection errors using the reusable function with temporary valid data
            # Save original arrays temporarily 
            orig_image_points = self.image_points
            orig_object_points = self.object_points
            orig_end2base_matrices = self.end2base_matrices
            
            # Temporarily set instance variables to valid arrays for reprojection calculation
            self.image_points = valid_image_points
            self.object_points = valid_object_points  
            self.end2base_matrices = valid_end2base_matrices
            
            try:
                per_image_errors, rms_error = self._calculate_reprojection_errors(
                    cam2end_4x4, target2end_matrix, rvecs, tvecs, verbose)
            finally:
                # Always restore original arrays
                self.image_points = orig_image_points
                self.object_points = orig_object_points
                self.end2base_matrices = orig_end2base_matrices
            
            return {
                'rms_error': rms_error,
                'per_image_errors': per_image_errors,
                'base2cam_matrix': end2cam_4x4,  # For eye-in-hand
                'target2end_matrix': target2end_matrix,
                'valid_indices': valid_indices,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
            
        except Exception as e:
            if verbose:
                print(f"Eye-in-hand calibration failed: {e}")
            return None
    
    def _calibrate_eye_to_hand(self, method: int, verbose: bool = False) -> dict:
        """
        Perform eye-to-hand calibration using specified method.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            dict: Calibration results or None if failed
        """
        try:
            # Get valid data directly from stored calibration data
            if not hasattr(self, '_valid_calibration_indices'):
                raise ValueError("Valid calibration data not found. Call calibrate() first.")
                
            valid_indices = self._valid_calibration_indices
            rvecs = self._valid_calibration_rvecs
            tvecs = self._valid_calibration_tvecs
            valid_end2base_matrices = self._valid_end2base_matrices
            
            if verbose:
                print(f"Using pre-extracted valid data: {len(valid_indices)} valid images")
            
            # Create arrays for valid detections only
            valid_image_points = [self.image_points[i] for i in valid_indices]
            valid_object_points = [self.object_points[i] for i in valid_indices]
            
            # For eye-to-hand, use base2end transformations (inverse of valid end2base)
            base2end_matrices = [np.linalg.inv(matrix) for matrix in valid_end2base_matrices]
            base2end_Rs = np.array([matrix[:3, :3] for matrix in base2end_matrices])
            base2end_ts = np.array([matrix[:3, 3] for matrix in base2end_matrices])
            rvecs_array = np.array([rvec for rvec in rvecs])
            tvecs_array = np.array([tvec for tvec in tvecs])
            
            if verbose:
                print(f"Using base2end transformations (eye-to-hand configuration)")
                print(f"Working with {len(valid_indices)} valid detections")
            
            # Perform eye-to-hand calibration using OpenCV
            cam2base_R, cam2base_t = cv2.calibrateHandEye(
                base2end_Rs, base2end_ts, rvecs_array, tvecs_array, method)
            
            # Create cam2base 4x4 transformation matrix
            cam2base_4x4 = np.eye(4)
            cam2base_4x4[:3, :3] = cam2base_R
            cam2base_4x4[:3, 3] = cam2base_t[:, 0]
            
            # For eye-to-hand, we need base2cam matrix, so invert the result
            base2cam_4x4 = np.linalg.inv(cam2base_4x4)
            
            # Calculate the optimal target2end matrix that minimizes reprojection error
            target2end_matrix = self._calculate_optimal_target2end_matrix(base2cam_4x4, rvecs, tvecs, verbose)
            
            # Calculate reprojection errors using the reusable function with temporary valid data
            # Save original arrays temporarily 
            orig_image_points = self.image_points
            orig_object_points = self.object_points
            orig_end2base_matrices = self.end2base_matrices
            
            # Temporarily set instance variables to valid arrays for reprojection calculation
            self.image_points = valid_image_points
            self.object_points = valid_object_points  
            self.end2base_matrices = valid_end2base_matrices
            
            try:
                per_image_errors, rms_error = self._calculate_reprojection_errors(
                    base2cam_4x4, target2end_matrix, rvecs, tvecs, verbose)
            finally:
                # Always restore original arrays
                self.image_points = orig_image_points
                self.object_points = orig_object_points
                self.end2base_matrices = orig_end2base_matrices
            
            return {
                'rms_error': rms_error,
                'per_image_errors': per_image_errors,
                'base2cam_matrix': base2cam_4x4,  # For eye-to-hand
                'target2end_matrix': target2end_matrix,
                'valid_indices': valid_indices,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
            
        except Exception as e:
            if verbose:
                print(f"Eye-to-hand calibration failed: {e}")
            return None
    
    def _calculate_optimal_target2end_matrix(self, base2cam_or_end2cam_matrix: np.ndarray, 
                                           rvecs: List[np.ndarray], tvecs: List[np.ndarray], 
                                           verbose: bool = False) -> np.ndarray:
        """
        Calculate the optimal target2end matrix that minimizes overall reprojection error.
        
        This method finds the best target pose in the end-effector coordinate system
        by analyzing the relationship between detected target poses and robot positions.
        
        Args:
            base2cam_or_end2cam_matrix: Base-to-camera or end-effector-to-camera transformation matrix
            rvecs: Rotation vectors from PnP solution
            tvecs: Translation vectors from PnP solution
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target2end transformation matrix
        """
        if len(self.end2base_matrices) == 0:
            raise ValueError("No robot poses available for target2end calculation")
        
        # Use the first detected target pose as reference for target2end calculation
        # This is a simplified approach - could be improved with averaging or optimization
        reference_target2cam = np.eye(4)
        reference_target2cam[:3, :3] = cv2.Rodrigues(rvecs[0])[0] if len(rvecs) > 0 else np.eye(3)
        reference_target2cam[:3, 3] = tvecs[0][:, 0] if len(tvecs) > 0 else np.zeros(3)
        
        # Calculate target2end matrix based on calibration type
        if self.calibration_type == "eye_in_hand":
            # For eye-in-hand: target2end = inv(end2base) * inv(end2cam) * target2cam
            target2end_matrix = np.linalg.inv(self.end2base_matrices[0]) @ np.linalg.inv(base2cam_or_end2cam_matrix) @ reference_target2cam
        else:
            # For eye-to-hand: target2end = inv(base2end) * inv(base2cam) * target2cam
            base2end_matrix = np.linalg.inv(self.end2base_matrices[0])  
            target2end_matrix = base2end_matrix @ np.linalg.inv(base2cam_or_end2cam_matrix) @ reference_target2cam
        
        if verbose:
            print(f"Calculated target2end matrix using reference pose")
            print(f"Target2end matrix:\n{target2end_matrix}")
        
        return target2end_matrix
    
    def _calculate_poses_for_all_images(self, verbose: bool = False) -> None:
        """
        Calculate camera poses (rvec/tvec) for all detected calibration patterns.
        
        This function attempts to calculate poses for all images with detected patterns.
        Failed calculations result in None values, maintaining alignment with the image array.
        
        Args:
            verbose: Whether to print detailed information about pose calculation
            
        Note:
            Results are stored in self.rvecs and self.tvecs arrays, aligned with self.images.
            None values indicate failed pose calculations for those images.
            
            Pose calculation attempts solvePnP for each detected pattern and validates:
            - solvePnP success (ret == True)
            - No NaN or infinite values in rvec/tvec
            - Non-zero pose magnitudes (filters out degenerate solutions)
        """
        if verbose:
            print(f"📐 Calculating camera poses for all detected patterns...")
        
        # Initialize pose arrays aligned with images
        self.rvecs = [None] * len(self.images)
        self.tvecs = [None] * len(self.images)
        
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
                            successful_poses += 1
                            
                            if verbose:
                                print(f"   ✅ Image {i}: Valid pose calculated")
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
            print(f"📊 Pose Calculation Summary:")
            print(f"   • Successful poses: {successful_poses}")
            print(f"   • Failed poses: {failed_poses}")
            print(f"   • Total images: {len(self.images)}")
    
    def _extract_valid_calibration_data(self, verbose: bool = False) -> tuple:
        """
        Extract valid data for hand-eye calibration from calculated poses.
        
        This function filters the pose arrays to include only images where:
        1. Pattern detection succeeded
        2. Pose calculation succeeded  
        3. Corresponding robot pose exists
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            tuple: (valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices) where:
                - valid_indices: List of image indices with valid data
                - valid_rvecs: List of rotation vectors for valid images
                - valid_tvecs: List of translation vectors for valid images
                - valid_end2base_matrices: List of robot poses for valid images
                
        Raises:
            ValueError: If insufficient valid data found (< 3 required for hand-eye calibration)
            
        Note:
            This function requires that _calculate_poses_for_all_images() has been called first.
        """
        if self.rvecs is None or self.tvecs is None:
            raise ValueError("Poses not calculated. Call _calculate_poses_for_all_images() first.")
            
        if verbose:
            print(f"🔍 Extracting valid data for hand-eye calibration...")
        
        valid_indices = []
        valid_rvecs = []
        valid_tvecs = []
        valid_end2base_matrices = []
        
        for i in range(len(self.images)):
            # Check if all required data is available and valid for this image
            if (self.rvecs[i] is not None and 
                self.tvecs[i] is not None and
                i < len(self.end2base_matrices) and
                self.end2base_matrices[i] is not None):
                
                valid_indices.append(i)
                valid_rvecs.append(self.rvecs[i])
                valid_tvecs.append(self.tvecs[i])
                valid_end2base_matrices.append(self.end2base_matrices[i])
                
                if verbose:
                    print(f"   ✅ Image {i}: Valid for calibration")
            elif verbose:
                missing_items = []
                if self.rvecs[i] is None:
                    missing_items.append("rvec")
                if self.tvecs[i] is None:
                    missing_items.append("tvec")
                if i >= len(self.end2base_matrices) or self.end2base_matrices[i] is None:
                    missing_items.append("robot_pose")
                print(f"   ❌ Image {i}: Missing {', '.join(missing_items)}")
        
        # Validate we have enough valid data for hand-eye calibration
        if len(valid_indices) < 3:
            raise ValueError(f"Insufficient valid data for hand-eye calibration: need at least 3, got {len(valid_indices)}")
            
        if verbose:
            print(f"📊 Valid Data Summary:")
            print(f"   • Valid images: {len(valid_indices)}")
            print(f"   • Invalid images: {len(self.images) - len(valid_indices)}")
            print(f"   • Valid indices: {valid_indices}")
            
        return valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices
    
    def _calculate_reprojection_errors(self, transformation_matrix: np.ndarray, target2end_matrix: np.ndarray, 
                                     rvecs: Optional[List[np.ndarray]] = None, tvecs: Optional[List[np.ndarray]] = None,
                                     verbose: bool = False) -> Tuple[List[float], float]:
        """
        Calculate reprojection errors for hand-eye calibration results.
        
        This function computes per-image and overall RMS reprojection errors by projecting
        3D pattern points through the calibrated transformation chain back to image coordinates
        and comparing with detected 2D points.
        
        Args:
            transformation_matrix: Main transformation matrix (cam2end for eye-in-hand or base2cam for eye-to-hand)
            target2end_matrix: Target to end-effector transformation matrix
            rvecs: Rotation vectors from PnP (optional, will use self.rvecs if not provided)
            tvecs: Translation vectors from PnP (optional, will use self.tvecs if not provided)
            verbose: Whether to print detailed error information
            
        Returns:
            Tuple[List[float], float]: (per_image_errors, rms_error)
                - per_image_errors: List of reprojection errors for each image in pixels
                - rms_error: Overall RMS reprojection error in pixels
                
        Note:
            Transformation chains:
            - Eye-in-hand: target2cam = end2cam * end2base * target2end
            - Eye-to-hand: target2cam = base2cam * base2end * target2end
        """
        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not available for reprojection error calculation")
        
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera parameters not available for reprojection error calculation")
            
        if len(self.end2base_matrices) == 0:
            raise ValueError("Robot poses not available for reprojection error calculation")
        
        # Use provided rvecs/tvecs or fall back to instance variables
        if rvecs is None:
            if not hasattr(self, 'rvecs') or self.rvecs is None:
                raise ValueError("Rotation vectors not available and not provided")
            rvecs = self.rvecs
            
        if tvecs is None:
            if not hasattr(self, 'tvecs') or self.tvecs is None:
                raise ValueError("Translation vectors not available and not provided")
            tvecs = self.tvecs
        
        per_image_errors = []
        total_error = 0.0
        valid_images = 0
        
        if verbose:
            print(f"Calculating reprojection errors for {len(self.image_points)} images...")
            print(f"Calibration type: {self.calibration_type}")
        
        for i in range(len(self.image_points)):
            try:
                # Calculate target2cam transformation based on calibration type
                if self.calibration_type == "eye_in_hand":
                    # Eye-in-hand: target2cam = end2cam * end2base * target2end
                    # transformation_matrix is cam2end, so invert it to get end2cam
                    end2cam_matrix = np.linalg.inv(transformation_matrix)
                    target2cam = end2cam_matrix @ self.end2base_matrices[i] @ target2end_matrix
                    
                elif self.calibration_type == "eye_to_hand":
                    # Eye-to-hand: target2cam = base2cam * base2end * target2end
                    # transformation_matrix is base2cam, base2end is inverse of end2base
                    base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                    target2cam = transformation_matrix @ base2end_matrix @ target2end_matrix
                else:
                    raise ValueError(f"Unknown calibration type: {self.calibration_type}")
                
                # Extract rotation and translation from transformation matrix
                target2cam_R = target2cam[:3, :3]
                target2cam_t = target2cam[:3, 3]
                
                # Project 3D object points to image coordinates
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i],
                    target2cam_R,
                    target2cam_t,
                    self.camera_matrix,
                    self.distortion_coefficients
                )
                
                # Calculate reprojection error for this image
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                per_image_errors.append(error)
                total_error += error * error
                valid_images += 1
                
                if verbose:
                    print(f"   Image {i}: {error:.4f} pixels")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Image {i}: Error calculating reprojection - {e}")
                per_image_errors.append(float('inf'))
        
        # Calculate RMS error
        if valid_images > 0:
            rms_error = np.sqrt(total_error / valid_images)
            if verbose:
                print(f"📊 Reprojection Error Summary:")
                print(f"   • Valid images: {valid_images}/{len(self.image_points)}")
                print(f"   • RMS error: {rms_error:.4f} pixels")
                print(f"   • Min error: {min([e for e in per_image_errors if not np.isinf(e)]):.4f} pixels")
                print(f"   • Max error: {max([e for e in per_image_errors if not np.isinf(e)]):.4f} pixels")
        else:
            rms_error = float('inf')
            if verbose:
                print("❌ No valid images for reprojection error calculation")
        
        return per_image_errors, rms_error
    
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
