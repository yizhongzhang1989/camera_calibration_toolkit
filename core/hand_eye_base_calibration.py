"""
Hand-Eye Base Calibration Module
===============================

This module provides the base class for hand-eye calibration operations.
It contains common functionality shared between eye-in-hand and eye-to-hand calibration types.

HandEyeBaseCalibrator is an abstract base class that defines:
- Common data structures for robot poses, transformations, and hand-eye calibration
- Common methods for robot pose handling, transformation calculations, and optimization
- Abstract methods that must be implemented by specialized eye-in-hand and eye-to-hand calibrators

This design eliminates code duplication between eye-in-hand and eye-to-hand calibrators
while providing a consistent interface and allowing specialized functionality.

Key Design Principles:
- Inherits from BaseCalibrator for common image/pattern functionality
- Adds robot-specific functionality (poses, transformations, hand-eye calibration)
- Uses template method pattern for calibration workflow
- Provides common optimization and error calculation methods
- Separates coordinate system handling between eye-in-hand and eye-to-hand
"""

import os
import json
import numpy as np
import cv2
from abc import abstractmethod
from typing import Tuple, List, Optional, Union, Dict, Any
from .base_calibrator import BaseCalibrator
from .calibration_patterns import CalibrationPattern
from .utils import xyz_rpy_to_matrix, matrix_to_xyz_rpy, inverse_transform_matrix

# Optional import for optimization
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    nlopt = None
    HAS_NLOPT = False


class HandEyeBaseCalibrator(BaseCalibrator):
    """
    Abstract base class for hand-eye calibration operations.
    
    This class provides common functionality for both eye-in-hand and eye-to-hand calibration:
    - Robot pose data management
    - Transformation matrix calculations
    - Common calibration workflow
    - Optimization methods
    - Error calculation and visualization
    
    Specialized calibrators (EyeInHandCalibrator, EyeToHandCalibrator) inherit from this class
    and implement the specific coordinate system transformations and calibration logic.
    
    Common Workflow:
    1. Initialize with robot poses, images, and camera parameters
    2. Detect calibration pattern points (inherited from BaseCalibrator)
    3. Calculate target-to-camera transformations (common method)
    4. Perform hand-eye calibration (abstract method - specialized implementation)
    5. Calculate reprojection errors (common method with specialized transformation chain)
    6. Optional optimization (common method with specialized error calculation)
    7. Save results and generate visualization (common methods)
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
        
        # Target-to-camera transformations (common intermediate results)
        self.target2cam_matrices = None            # List of target to camera transformation matrices
        
        # Hand-eye calibration results (abstract - defined by subclasses)
        # Eye-in-hand: cam2end_matrix, target2base_matrix
        # Eye-to-hand: base2cam_matrix, target2end_matrix
        
        # Optimization and error tracking (common to both calibration types)
        self.optimization_completed = False        # Whether optimization has been completed
        self.initial_rms_error = None             # RMS error before optimization
        self.optimized_rms_error = None           # RMS error after optimization
        
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
        
        # Initialize calibration result attributes (common)
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
    
    # ============================================================================
    # Robot Pose Management (Updated to match HandEyeCalibrator interface)
    # ============================================================================
    
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
    # ============================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # ============================================================================
    
    @abstractmethod
    def _calibrate_with_best_method(self, verbose: bool = False) -> bool:
        """
        Perform calibration using all methods and select the best one.
        Must be implemented by subclasses.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_HORAUD, verbose: bool = False) -> bool:
        """
        Perform hand-eye calibration using the specified method.
        Must be implemented by subclasses to handle specific coordinate system transformations.
        
        Args:
            method: OpenCV hand-eye calibration method
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """
        Get the main transformation matrix result from calibration.
        - Eye-in-hand: Returns cam2end_matrix
        - Eye-to-hand: Returns base2cam_matrix
        
        Returns:
            np.ndarray or None: Transformation matrix if calibration completed
        """
        pass
    
    @abstractmethod
    def _calculate_reprojection_errors(self, transformation_matrix: Optional[np.ndarray] = None,
                                     target2end_matrix: Optional[np.ndarray] = None,
                                     rvecs: Optional[List[np.ndarray]] = None, 
                                     tvecs: Optional[List[np.ndarray]] = None,
                                     verbose: bool = False) -> Tuple[List[float], float]:
        """
        Calculate reprojection errors using the appropriate transformation chain.
        - Eye-in-hand: Uses cam2end and target2base matrices
        - Eye-to-hand: Uses base2cam and target2end matrices
        
        Must set self.per_image_errors and self.rms_error
        
        Returns:
            Tuple[List[float], float]: (per_image_errors, rms_error)
        """
        pass
    
    @abstractmethod
    def _get_calibration_results_dict(self) -> Dict[str, Any]:
        """
        Get calibration results in dictionary format for saving.
        Each subclass returns its specific transformation matrices and parameters.
        
        Returns:
            dict: Calibration results specific to the calibration type
        """
        pass
    
    # ============================================================================
    # Common Workflow Methods
    # ============================================================================
    
    def perform_calibration_workflow(self, verbose: bool = False) -> bool:
        """
        Perform the complete calibration workflow using the best available method.
        
        This method automatically tests all available OpenCV hand-eye calibration methods
        and selects the one that produces the smallest reprojection error.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            After successful calibration, use getter methods to access results:
            - get_rms_error(): Overall RMS reprojection error  
            - get_transformation_matrix(): Transformation matrix
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
        self._calculate_poses_for_all_images(verbose=verbose)
        
        # Step 3: Extract valid data for hand-eye calibration
        valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices = self._extract_valid_calibration_data(verbose=verbose)
        
        # Store the valid data for use by calibration methods
        self._valid_calibration_indices = valid_indices
        self._valid_calibration_rvecs = valid_rvecs
        self._valid_calibration_tvecs = valid_tvecs
        self._valid_end2base_matrices = valid_end2base_matrices
        
        # Step 4: Perform calibration (implemented by subclass) using best method selection
        if not self._calibrate_with_best_method(verbose):
            return False
        
        self.calibration_completed = True
        
        if verbose:
            print("✅ Hand-eye calibration workflow completed successfully")
            print(f"   RMS error: {self.rms_error:.4f} pixels")
        
        return True
    
    def _calculate_poses_for_all_images(self, verbose: bool = False) -> None:
        """
        Calculate camera poses (rvec/tvec) for all detected calibration patterns.
        
        Args:
            verbose: Whether to print detailed information about pose calculation
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
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            tuple: (valid_indices, valid_rvecs, valid_tvecs, valid_end2base_matrices)
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
    
    @abstractmethod
    def _calibrate_with_best_method(self, verbose: bool = False) -> bool:
        """
        Perform calibration using all methods and select the best one.
        Must be implemented by subclasses.
        
        Args:
            verbose: Whether to print detailed information
            
        Returns:
            bool: True if calibration succeeded, False otherwise
        """
        pass
    
    # ============================================================================
    # Common Result Access Methods
    # ============================================================================
    
    def get_rms_error(self) -> Optional[float]:
        """Get overall RMS reprojection error."""
        return self.rms_error
    
    def get_per_image_errors(self) -> Optional[List[float]]:
        """Get per-image reprojection errors."""
        return self.per_image_errors
    
    def is_calibrated(self) -> bool:
        """
        Check if hand-eye calibration has been completed successfully.
        
        Returns:
            bool: True if both intrinsic and extrinsic calibration are complete
        """
        return (self.calibration_completed and 
                self.camera_matrix is not None and 
                self.rvecs is not None and 
                self.tvecs is not None and
                self.get_transformation_matrix() is not None)
    
    # ============================================================================
    # Common Optimization Methods (Optional)
    # ============================================================================
    
    def optimize_calibration(self, iterations: int = 5, ftol_rel: float = 1e-6, 
                           verbose: bool = False) -> Optional[float]:
        """
        Optimize calibration results to minimize reprojection error.
        
        Args:
            iterations: Number of optimization iterations
            ftol_rel: Relative tolerance for convergence
            verbose: Whether to print optimization progress
            
        Returns:
            float or None: Optimized RMS error if successful, None if failed
        """
        if not HAS_NLOPT:
            print("⚠️ Optimization requires nlopt package (pip install nlopt)")
            return None
        
        if not self.is_calibrated():
            print("❌ Must complete calibration before optimization")
            return None
        
        try:
            if verbose:
                print(f"🔍 Starting calibration optimization...")
                print(f"   Initial RMS error: {self.rms_error:.4f} pixels")
            
            self.initial_rms_error = self.rms_error
            
            # Perform optimization (implemented by subclass)
            optimized_error = self._optimize_transformation_matrices(iterations, ftol_rel, verbose)
            
            if optimized_error is not None and optimized_error < self.initial_rms_error:
                improvement = self.initial_rms_error - optimized_error
                improvement_pct = (improvement / self.initial_rms_error) * 100
                
                self.optimized_rms_error = optimized_error
                self.optimization_completed = True
                
                if verbose:
                    print(f"✅ Optimization completed successfully!")
                    print(f"   Optimized RMS error: {optimized_error:.4f} pixels")
                    print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                
                return optimized_error
            else:
                if verbose:
                    print(f"⚠️ Optimization completed with no significant improvement")
                return self.initial_rms_error
                
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None
    
    @abstractmethod
    def _optimize_transformation_matrices(self, iterations: int, ftol_rel: float, verbose: bool) -> Optional[float]:
        """
        Optimize the transformation matrices specific to the calibration type.
        Must be implemented by subclasses.
        
        Returns:
            float or None: Optimized RMS error if successful
        """
        pass
    
    # ============================================================================
    # Common Visualization and Results Saving
    # ============================================================================
    
    def draw_reprojection_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw reprojection analysis on images showing detected vs projected points.
        Uses the transformation chain specific to the calibration type.
        
        Returns:
            List of (filename, image_with_reprojection) tuples
        """
        if not self.is_calibrated():
            print("❌ Must complete calibration before drawing reprojections")
            return []
        
        try:
            reprojection_images = []
            
            for i in range(len(self.images)):
                # Create copy of original image
                img_copy = self.images[i].copy()
                
                # Get projected points using calibration-specific transformation chain
                projected_points = self._get_projected_points_for_image(i)
                
                if projected_points is not None:
                    # Draw detected points in green
                    for point in self.image_points[i]:
                        cv2.circle(img_copy, tuple(point[0].astype(int)), 3, (0, 255, 0), -1)
                    
                    # Draw projected points in red
                    for point in projected_points:
                        cv2.circle(img_copy, tuple(point[0].astype(int)), 2, (0, 0, 255), -1)
                    
                    # Draw lines connecting corresponding points
                    for j in range(len(self.image_points[i])):
                        if j < len(projected_points):
                            pt1 = tuple(self.image_points[i][j][0].astype(int))
                            pt2 = tuple(projected_points[j][0].astype(int))
                            cv2.line(img_copy, pt1, pt2, (255, 0, 0), 1)
                    
                    # Add error text
                    if self.per_image_errors:
                        error_text = f"Error: {self.per_image_errors[i]:.2f}px"
                        cv2.putText(img_copy, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                filename = f"image_{i:02d}_reprojection" if self.image_paths is None else os.path.splitext(os.path.basename(self.image_paths[i]))[0] + "_reprojection"
                reprojection_images.append((filename, img_copy))
            
            return reprojection_images
            
        except Exception as e:
            print(f"❌ Failed to draw reprojections: {e}")
            return []
    
    @abstractmethod
    def _get_projected_points_for_image(self, image_index: int) -> Optional[np.ndarray]:
        """
        Get projected points for a specific image using calibration-specific transformation chain.
        Must be implemented by subclasses.
        
        Args:
            image_index: Index of the image
            
        Returns:
            np.ndarray or None: Projected points if successful
        """
        pass
    
    def save_results(self, save_directory: str) -> None:
        """
        Save calibration results to directory.
        
        Args:
            save_directory: Directory to save results
        """
        if not self.is_calibrated():
            print("❌ Must complete calibration before saving results")
            return
        
        try:
            os.makedirs(save_directory, exist_ok=True)
            
            # Get calibration results from subclass
            results = self._get_calibration_results_dict()
            
            # Add common information
            results.update({
                "calibration_completed": self.calibration_completed,
                "rms_error": float(self.rms_error) if self.rms_error else None,
                "per_image_errors": [float(e) for e in self.per_image_errors] if self.per_image_errors else None,
                "optimization_completed": self.optimization_completed,
                "initial_rms_error": float(self.initial_rms_error) if self.initial_rms_error else None,
                "optimized_rms_error": float(self.optimized_rms_error) if self.optimized_rms_error else None,
                "num_images": len(self.images) if self.images else 0,
                "image_size": self.image_size,
                "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                "distortion_coefficients": self.distortion_coefficients.tolist() if self.distortion_coefficients is not None else None
            })
            
            # Save main results file
            results_file = os.path.join(save_directory, "hand_eye_calibration_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Hand-eye calibration results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
