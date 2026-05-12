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
from scipy.optimize import minimize
from .base_calibrator import BaseCalibrator
from .calibration_patterns import CalibrationPattern
from .utils import xyz_rpy_to_matrix


class HandEyeBaseCalibrator(BaseCalibrator):
    """
    Abstract base class for hand-eye calibration data handling.
    
    This class provides common IO functionality for both eye-in-hand and eye-to-hand calibration:
    - Robot pose data management
    - Transformation matrix IO and validation
    - Common data structures and validation
    
    Specialized calibrators inherit from this class and implement calibration logic.

    Subclasses are expected to set ``_primary_name`` and ``_secondary_name`` class
    attributes (used for log/diagnostic messages) and implement three small math
    hooks (``_compose_target2cam``, ``_solve_primary``, ``_compose_secondary_candidate``)
    plus ``_build_result_dict``. Everything else (method sweep, per-image
    reprojection error, two-stage joint optimization, JSON serialization, public
    accessors) is shared in this base class.
    """

    # Names used in log strings; subclasses override them.
    _primary_name: str = "primary"
    _secondary_name: str = "secondary"

    def __init__(self, 
                 images: Optional[List[np.ndarray]] = None,
                 end2base_matrices: Optional[List[np.ndarray]] = None,
                 image_paths: Optional[List[str]] = None, 
                 calibration_pattern: Optional[CalibrationPattern] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coefficients: Optional[np.ndarray] = None,
                 verbose: bool = False):
        """
        Initialize HandEyeBaseCalibrator with unified interface for hand-eye calibration.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            end2base_matrices: List of 4x4 transformation matrices from end-effector to base
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            camera_matrix: 3x3 camera intrinsic matrix or None (if None, will be calibrated)
            distortion_coefficients: Camera distortion coefficients or None (if None, will be calibrated)
            verbose: Whether to print progress information during initialization (default: False)
            
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

        # Generic primary/secondary transformation slots. Subclasses expose them
        # under domain-specific names via @property aliases (e.g. cam2end_matrix).
        self._primary_matrix: Optional[np.ndarray] = None
        self._secondary_matrix: Optional[np.ndarray] = None
        
        # Handle special case: if both image_paths and end2base_matrices are provided,
        # don't automatically load from JSON files to avoid overwriting the provided matrices
        if image_paths is not None and end2base_matrices is not None:
            # Initialize base class WITHOUT calling set_images_from_paths automatically
            # We'll handle image loading manually to preserve the provided end2base_matrices
            super().__init__(images=None, image_paths=None, calibration_pattern=calibration_pattern, verbose=verbose)
            
            # Set images manually using base class method to avoid JSON loading
            success = super().set_images_from_paths(image_paths, verbose=verbose)
            if not success:
                raise ValueError("Failed to load images from provided paths")
                
            if verbose:
                print(f"ℹ️  Loaded {len(self.images)} images from paths, using provided end2base_matrices")
                print(f"   (JSON files were not loaded to preserve provided transformation matrices)")
            
        else:
            # Standard initialization - let base class handle image loading
            super().__init__(images, image_paths, calibration_pattern, verbose=verbose)
        
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
    
    def calibrate(self, method: Optional[int] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Perform hand-eye calibration using the specified method or find the best method.

        This shared template implementation drives the same calibration flow for
        both eye-in-hand and eye-to-hand configurations. The subclass supplies the
        three math hooks (``_solve_primary``, ``_compose_secondary_candidate``,
        ``_compose_target2cam``) and ``_build_result_dict``; everything else
        (method sweep, optimization, result packaging) is identical.

        Args:
            method: Optional OpenCV calibration method constant. If ``None`` or
                invalid, every available method is tested and the one with the
                lowest reprojection error is chosen.
            verbose: Whether to print detailed calibration progress and results.

        Returns:
            Optional[Dict[str, Any]]: Result dictionary on success, or ``None``
            if no method succeeded. Exact keys (e.g. ``cam2end_matrix`` vs
            ``base2cam_matrix``) are determined by the subclass via
            ``_build_result_dict``.
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

            kind = f"{self._primary_name}/{self._secondary_name}"
            if verbose:
                print(f"🤖 Running hand-eye calibration ({kind}) with {valid_images} image-pose pairs")

            # Determine which methods to test
            available_methods = self.get_available_methods()

            if method is None or method not in available_methods:
                methods_to_try = available_methods
                if verbose:
                    if method is None:
                        print("🔍 No method specified, testing all available methods...")
                    else:
                        print(f"⚠️ Invalid method specified: {method}")
                        print(f"🔍 Valid methods are: {list(available_methods.keys())}")
                        print(f"🔍 Falling back to testing all available methods...")
            else:
                method_name = available_methods[method]
                methods_to_try = {method: method_name}
                if verbose:
                    print(f"🎯 Using specified method: {method_name} ({method})")

            best_method = None
            best_method_name = None
            best_rms_error = float('inf')
            best_primary = None
            best_secondary = None
            best_per_image_errors = None

            for test_method, method_name in methods_to_try.items():
                if verbose and len(methods_to_try) > 1:
                    print(f"\n🧪 Testing method: {method_name} ({test_method})")

                try:
                    success, primary_matrix, secondary_matrix, rms_error, per_image_errors = \
                        self._perform_single_calibration(test_method, verbose=False)

                    if success and rms_error < best_rms_error:
                        best_method = test_method
                        best_method_name = method_name
                        best_rms_error = rms_error
                        best_primary = primary_matrix.copy()
                        best_secondary = secondary_matrix.copy()
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
                                print(f"❌ Hand-eye calibration failed with method {method_name}")
                except Exception as e:
                    if verbose:
                        if len(methods_to_try) > 1:
                            print(f"   ❌ Method {method_name} failed with error: {e}")
                        else:
                            print(f"❌ Hand-eye calibration failed with method {method_name}: {e}")
                    continue

            if best_method is None:
                if verbose and len(methods_to_try) > 1:
                    print("❌ All calibration methods failed")
                return None

            if verbose:
                if len(methods_to_try) > 1:
                    print(f"\n🎉 Best method selected: {best_method_name} with RMS error {best_rms_error:.4f}")
                else:
                    print(f"✅ Hand-eye calibration completed successfully!")
                    print(f"RMS reprojection error: {best_rms_error:.4f} pixels")
                print(f"{self._primary_name} transformation matrix:")
                print(f"{best_primary}")

            # Store the best results in generic slots
            self._primary_matrix = best_primary
            self._secondary_matrix = best_secondary
            self.rms_error = best_rms_error
            self.per_image_errors = best_per_image_errors
            self.best_method = best_method
            self.best_method_name = best_method_name

            # Snapshot initial (pre-optimization) results for the caller
            initial_results = self._build_result_dict(
                best_primary, best_secondary, best_rms_error, before_opt=None
            )

            optimized_results = initial_results.copy()
            if verbose:
                print(f"\n🔧 Attempting optimization...")

            try:
                initial_error, optimized_rms = self.optimize_calibration(ftol_rel=1e-6, verbose=verbose)

                improvement = initial_error - optimized_rms
                improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0

                if optimized_rms < initial_error:
                    # ``optimize_calibration`` has already updated self._primary_matrix /
                    # self._secondary_matrix / self.rms_error in place when it improved.
                    optimized_results = self._build_result_dict(
                        self._primary_matrix, self._secondary_matrix, self.rms_error,
                        before_opt=None,
                    )
                    if verbose:
                        print(f"✅ Optimization completed!")
                        print(f"   Initial RMS error: {initial_error:.4f} pixels")
                        print(f"   Optimized RMS error: {optimized_rms:.4f} pixels")
                        print(f"   Improvement: {improvement:.4f} pixels ({improvement_pct:.1f}%)")
                else:
                    if verbose:
                        print(f"⚠️ Optimization did not improve results")
                        print(f"   Initial RMS error: {initial_results['rms_error']:.4f} pixels")
                        print(f"   Optimized RMS error: {optimized_rms:.4f} pixels")
                        print(f"   Keeping initial calibration results")
            except Exception as e:
                if verbose:
                    print(f"⚠️ Optimization failed: {e}")
                    print(f"   Returning initial calibration results")

            optimized_results['before_opt'] = initial_results

            self.calibration_completed = True
            return optimized_results

        except Exception as e:
            if verbose:
                print(f"❌ Hand-eye calibration failed: {e}")
            self.calibration_completed = False
            return None

    # ------------------------------------------------------------------
    # Abstract math hooks — subclasses must implement these tiny kernels.
    # Everything else is shared.
    # ------------------------------------------------------------------

    @abstractmethod
    def _compose_target2cam(self,
                            primary_matrix: np.ndarray,
                            secondary_matrix: np.ndarray,
                            end2base_matrix: np.ndarray) -> np.ndarray:
        """
        Compose the target→camera transformation for one image from the primary
        matrix, the secondary matrix and that image's end→base transformation.

        Eye-in-hand: ``inv(cam2end) @ inv(end2base) @ target2base``
        Eye-to-hand: ``base2cam @ end2base @ target2end``
        """
        raise NotImplementedError

    @abstractmethod
    def _solve_primary(self,
                       end2base_matrices_valid: List[np.ndarray],
                       target2cam_matrices_valid: List[np.ndarray],
                       method: int) -> np.ndarray:
        """
        Run ``cv2.calibrateHandEye`` with subclass-specific argument ordering
        and return the 4×4 primary transformation matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def _compose_secondary_candidate(self,
                                     primary_matrix: np.ndarray,
                                     end2base_matrix: np.ndarray,
                                     target2cam_matrix: np.ndarray) -> np.ndarray:
        """
        Build one candidate secondary matrix from a single image's measurements.

        Eye-in-hand: ``end2base @ cam2end @ target2cam``  (gives target2base)
        Eye-to-hand: ``inv(end2base) @ inv(base2cam) @ target2cam``  (gives target2end)
        """
        raise NotImplementedError

    @abstractmethod
    def _build_result_dict(self,
                           primary_matrix: np.ndarray,
                           secondary_matrix: np.ndarray,
                           rms_error: float,
                           before_opt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build the subclass-specific result dictionary (e.g. with the keys
        ``cam2end_matrix`` / ``target2base_matrix`` for eye-in-hand).
        """
        raise NotImplementedError

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

    def set_images_from_paths(self, image_paths: List[str], verbose: bool = False) -> bool:
        """
        Set images from file paths and read corresponding JSON files with end2base matrices.
        
        For each image file, this method will:
        1. Load the image file
        2. Look for a JSON file with the same name (e.g., image.jpg -> image.json)
        3. Extract the "end2base" matrix from the JSON file
        
        Data is only valid if ALL images and corresponding JSON files are successfully read.
        
        Args:
            image_paths: List of image file paths
            verbose: Whether to print progress information (default: False)
            
        Returns:
            bool: True if all images and JSON files loaded successfully
        """
        if not image_paths:
            if verbose:
                print("Error: No image paths provided")
            return False
        
        try:
            images = []
            end2base_matrices = []
            valid_paths = []
            
            if verbose:
                print(f"Loading {len(image_paths)} images and corresponding JSON files...")
            
            for i, img_path in enumerate(image_paths):
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    if verbose:
                        print(f"Error: Could not load image {img_path}")
                    return False
                
                # Construct JSON file path (same name, different extension)
                base_name = os.path.splitext(img_path)[0]  # Remove extension
                json_path = base_name + '.json'
                
                # Check if JSON file exists
                if not os.path.exists(json_path):
                    if verbose:
                        print(f"Error: JSON file not found: {json_path}")
                    return False
                
                # Load and parse JSON file
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Extract end2base matrix
                    if 'end2base' not in json_data:
                        if verbose:
                            print(f"Error: 'end2base' key not found in {json_path}")
                        return False
                    
                    end2base = json_data['end2base']
                    
                    # Convert to numpy array and validate
                    end2base_matrix = np.array(end2base, dtype=np.float64)
                    
                    if end2base_matrix.shape != (4, 4):
                        if verbose:
                            print(f"Error: end2base matrix in {json_path} is not 4x4, got shape {end2base_matrix.shape}")
                        return False
                    
                    # Validate that it looks like a proper transformation matrix
                    if not np.allclose(end2base_matrix[3, :], [0, 0, 0, 1], atol=1e-6):
                        if verbose:
                            print(f"Warning: end2base matrix in {json_path} bottom row is not [0, 0, 0, 1]: {end2base_matrix[3, :]}")
                        # Don't return False here - just warn, as some matrices might have slight numerical errors
                    
                    # If we get here, both image and JSON loaded successfully
                    images.append(img)
                    end2base_matrices.append(end2base_matrix)
                    valid_paths.append(img_path)
                    
                    if verbose:
                        print(f"✅ Loaded image {i+1}/{len(image_paths)}: {os.path.basename(img_path)} with transform")
                    
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"Error: Could not parse JSON file {json_path}: {e}")
                    return False
                except Exception as e:
                    if verbose:
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
            
            if verbose:
                print(f"✅ Successfully loaded {len(self.images)} images with end2base matrices")
                print(f"📏 Image size: {self.image_size}")
            
            # Validate consistency of loaded data
            self._validate_input_consistency()
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error loading images and transformations: {e}")
            return False

    def set_images_from_arrays(self, images: List[np.ndarray], verbose: bool = False) -> bool:
        """
        Set images from numpy arrays.
        
        Args:
            images: List of image arrays
            verbose: Whether to print progress information (default: False)
            
        Returns:
            bool: True if images set successfully
        """
        self.images = images
        if images:
            h, w = images[0].shape[:2]
            self.image_size = (w, h)
            
        if verbose:
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

    # ============================================================================
    # Shared calibration algorithm (used by both eye-in-hand and eye-to-hand)
    # ============================================================================

    def _calculate_reprojection_errors(self,
                                       primary_matrix: np.ndarray,
                                       secondary_matrix: np.ndarray,
                                       verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Project 3-D object points using the calibrated transformation chain and
        compare them with the detected image points to obtain per-image and
        overall RMS reprojection error.
        """
        per_image_errors: List[float] = []
        total_error = 0.0
        valid_error_count = 0

        for i in range(len(self.image_points)):
            if (self.image_points[i] is not None and
                self.object_points[i] is not None and
                self.end2base_matrices[i] is not None):
                try:
                    target2cam = self._compose_target2cam(
                        primary_matrix, secondary_matrix, self.end2base_matrices[i]
                    )

                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i],
                        target2cam[:3, :3],
                        target2cam[:3, 3],
                        self.camera_matrix,
                        self.distortion_coefficients,
                    )

                    norm_L2 = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2)
                    num_points = len(projected_points)

                    error = norm_L2 / np.sqrt(num_points)
                    per_image_errors.append(error)

                    total_error += norm_L2 ** 2
                    valid_error_count += num_points

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

        if valid_error_count > 0:
            rms_error = np.sqrt(total_error / valid_error_count)
            num_valid_images = len([e for e in per_image_errors if e != float('inf')])
            if verbose:
                print(f"   Overall RMS reprojection error: {rms_error:.4f} pixels "
                      f"({num_valid_images} valid images, {valid_error_count} total points)")
        else:
            rms_error = float('inf')
            if verbose:
                print("   No valid images for reprojection error calculation")

        return rms_error, per_image_errors

    def calculate_reprojection_errors(self,
                                      primary_matrix: Optional[np.ndarray] = None,
                                      secondary_matrix: Optional[np.ndarray] = None,
                                      verbose: bool = False) -> Tuple[float, List[float]]:
        """
        Public wrapper around :meth:`_calculate_reprojection_errors`. Falls back
        to the stored calibration matrices when arguments are ``None``.
        """
        if primary_matrix is None:
            if self._primary_matrix is None:
                raise ValueError("No primary matrix provided and no calibration results stored")
            primary_matrix = self._primary_matrix

        if secondary_matrix is None:
            if self._secondary_matrix is None:
                raise ValueError("No secondary matrix provided and no calibration results stored")
            secondary_matrix = self._secondary_matrix

        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not detected. Run detect_pattern_points() first.")

        if self.end2base_matrices is None:
            raise ValueError("Robot poses not loaded")

        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters not available")

        return self._calculate_reprojection_errors(primary_matrix, secondary_matrix, verbose)

    def get_reproject_rvec_tvec(self) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Per-image rvec/tvec computed from the robot kinematic chain (not from
        ``solvePnP``), used by the base class to draw reprojections.
        """
        if not self.is_calibrated() or self._primary_matrix is None or self._secondary_matrix is None:
            raise ValueError("Hand-eye calibration not completed. Run calibrate() first.")

        if not self.end2base_matrices:
            raise ValueError("Robot end-effector poses not available. Set end2base_matrices first.")

        rvecs: List[Optional[np.ndarray]] = []
        tvecs: List[Optional[np.ndarray]] = []

        for i in range(len(self.images)):
            if (self.end2base_matrices[i] is not None and
                self.image_points[i] is not None and
                self.object_points[i] is not None):
                try:
                    pattern2cam_matrix = self._compose_target2cam(
                        self._primary_matrix, self._secondary_matrix, self.end2base_matrices[i]
                    )

                    rotation_matrix = pattern2cam_matrix[:3, :3]
                    translation_vector = pattern2cam_matrix[:3, 3]

                    rvec, _ = cv2.Rodrigues(rotation_matrix)
                    tvec = translation_vector.reshape(-1, 1)

                    rvecs.append(rvec)
                    tvecs.append(tvec)
                except Exception:
                    rvecs.append(None)
                    tvecs.append(None)
            else:
                rvecs.append(None)
                tvecs.append(None)

        return rvecs, tvecs

    def _calculate_optimal_secondary_matrix(self,
                                            primary_4x4: np.ndarray,
                                            verbose: bool = False) -> np.ndarray:
        """
        Build per-image candidates for the secondary matrix from the chosen
        primary matrix and pick the one with the lowest overall reprojection
        error.
        """
        if verbose:
            print(f"Calculating optimal {self._secondary_name} matrix...")

        best_error = float('inf')
        best_secondary: Optional[np.ndarray] = None

        candidate_matrices: List[np.ndarray] = []
        for i in range(len(self.target2cam_matrices)):
            if self.target2cam_matrices[i] is not None:
                candidate = self._compose_secondary_candidate(
                    primary_4x4, self.end2base_matrices[i], self.target2cam_matrices[i]
                )
                candidate_matrices.append(candidate)

        for candidate_idx, candidate in enumerate(candidate_matrices):
            rms_error, _ = self._calculate_reprojection_errors(primary_4x4, candidate, verbose=False)

            if rms_error < best_error:
                best_error = rms_error
                best_secondary = candidate.copy()
                if verbose:
                    print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f} (best so far)")
            elif verbose:
                print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f}")

        if best_secondary is not None:
            if verbose:
                print(f"✅ Optimal {self._secondary_name} matrix found with RMS error: {best_error:.4f}")
                print(f"{self._secondary_name} transformation matrix:")
                print(best_secondary)
        else:
            if verbose:
                print(f"⚠️ Could not find optimal {self._secondary_name} matrix, using first candidate")
            best_secondary = candidate_matrices[0] if candidate_matrices else np.eye(4)

        return best_secondary

    def _perform_single_calibration(self,
                                    method: int,
                                    verbose: bool = False
                                    ) -> Tuple[bool,
                                               Optional[np.ndarray],
                                               Optional[np.ndarray],
                                               float,
                                               Optional[List[float]]]:
        """
        Run a single ``cv2.calibrateHandEye`` invocation with the given method,
        derive the optimal secondary matrix from it, and report reprojection
        error.
        """
        try:
            valid_indices: List[int] = []
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

            end2base_valid = [self.end2base_matrices[i] for i in valid_indices]
            target2cam_valid = [self.target2cam_matrices[i] for i in valid_indices]

            primary_4x4 = self._solve_primary(end2base_valid, target2cam_valid, method)

            secondary_matrix = self._calculate_optimal_secondary_matrix(primary_4x4, verbose)
            rms_error, per_image_errors = self._calculate_reprojection_errors(
                primary_4x4, secondary_matrix, verbose
            )

            return True, primary_4x4, secondary_matrix, rms_error, per_image_errors

        except Exception as e:
            if verbose:
                print(f"   Calibration failed: {e}")
            return False, None, None, float('inf'), None

    def optimize_calibration(self,
                             ftol_rel: float = 1e-6,
                             verbose: bool = False) -> Tuple[float, float]:
        """
        Two-stage scipy refinement of the primary and secondary matrices.

        Stage 1 holds the primary matrix fixed and refines only the secondary.
        Stage 2 refines both jointly. The stored calibration state is only
        overwritten if the refinement actually reduces the reprojection error
        (this matches the eye-in-hand behaviour and also fixes the eye-to-hand
        path where the original code silently accepted regressions).
        """
        if self._primary_matrix is None:
            raise ValueError("Initial calibration must be completed before optimization. Call calibrate() first.")

        if verbose:
            print(f"Starting optimization...")
            print(f"Initial RMS error: {self.rms_error:.4f} pixels")

        initial_primary = self._primary_matrix.copy()
        initial_secondary = self._secondary_matrix.copy()
        initial_error = self.rms_error

        try:
            if verbose:
                print("   Two-stage optimization approach:")
                print(f"   Stage 1: Optimizing secondary matrix ({self._secondary_name}) only")

            primary_stage1, secondary_stage1, error_before_stage1, _ = \
                self._optimize_matrices_jointly(
                    initial_primary, initial_secondary, ftol_rel, verbose,
                    fix_primary_matrix=True,
                )

            if verbose:
                print("   Stage 2: Optimizing both matrices jointly")
            optimized_primary, optimized_secondary, _, error_after_stage2 = \
                self._optimize_matrices_jointly(
                    primary_stage1, secondary_stage1, ftol_rel, verbose,
                    fix_primary_matrix=False,
                )

            initial_opt_error = error_before_stage1
            final_opt_error = error_after_stage2

            if verbose:
                print(f"   Overall two-stage optimization: {initial_opt_error:.4f} -> {final_opt_error:.4f} pixels")
                if initial_opt_error > 0:
                    overall_improvement = (initial_opt_error - final_opt_error) / initial_opt_error * 100
                    print(f"   Overall improvement: {overall_improvement:.1f}%")

            # Only commit the refinement if it actually improved.
            if initial_opt_error > final_opt_error:
                self._primary_matrix = optimized_primary
                self._secondary_matrix = optimized_secondary

                self.rms_error, self.per_image_errors = self.calculate_reprojection_errors(
                    self._primary_matrix, self._secondary_matrix, verbose=False
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
            self._primary_matrix = initial_primary
            self._secondary_matrix = initial_secondary
            self.rms_error = initial_error
            return initial_error, initial_error

    def _optimize_matrices_jointly(self,
                                   initial_primary: np.ndarray,
                                   initial_secondary: np.ndarray,
                                   ftol_rel: float,
                                   verbose: bool,
                                   fix_primary_matrix: bool = False):
        """
        Scipy-based joint optimization of the primary and secondary matrices
        using small xyz/rpy delta perturbations.
        """
        initial_matrices = [initial_primary, initial_secondary]
        matrix_names = [self._primary_name, self._secondary_name]

        if fix_primary_matrix:
            optimize_flags = [False, True]
            opt_description = f"{self._secondary_name} matrix ({self._primary_name} fixed)"
        else:
            optimize_flags = [True, True]
            opt_description = f"both {self._primary_name} and {self._secondary_name} matrices"

        optimize_indices = [i for i, flag in enumerate(optimize_flags) if flag]
        param_count = len(optimize_indices) * 6
        initial_delta_params = np.zeros(param_count)

        if verbose:
            print(f"   Optimizing {opt_description}")

        def joint_objective(delta_params):
            try:
                result_matrices = [matrix.copy() for matrix in initial_matrices]

                param_offset = 0
                for i, should_optimize in enumerate(optimize_flags):
                    if should_optimize:
                        matrix_delta_params = delta_params[param_offset:param_offset + 6]
                        delta_matrix = xyz_rpy_to_matrix(matrix_delta_params)

                        result_matrices[i] = initial_matrices[i] @ delta_matrix
                        param_offset += 6

                rms_error, _ = self.calculate_reprojection_errors(
                    result_matrices[0], result_matrices[1], verbose=False
                )

                if not np.isfinite(rms_error):
                    return 1e6

                return rms_error
            except Exception:
                return 1e6

        single_matrix_bounds_low = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2])
        single_matrix_bounds_high = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])

        delta_bounds_low = np.tile(single_matrix_bounds_low, len(optimize_indices))
        delta_bounds_high = np.tile(single_matrix_bounds_high, len(optimize_indices))

        bounds = [(low, high) for low, high in zip(delta_bounds_low, delta_bounds_high)]

        try:
            result = minimize(
                joint_objective,
                initial_delta_params,
                method='Nelder-Mead',
                options={'ftol': ftol_rel, 'disp': False},
            )
            optimized_delta_params = result.x

            result_matrices = [matrix.copy() for matrix in initial_matrices]
            param_offset = 0
            for i, should_optimize in enumerate(optimize_flags):
                if should_optimize:
                    matrix_delta_params = optimized_delta_params[param_offset:param_offset + 6]
                    delta_matrix = xyz_rpy_to_matrix(matrix_delta_params)
                    result_matrices[i] = initial_matrices[i] @ delta_matrix
                    param_offset += 6

            initial_error = joint_objective(initial_delta_params)
            final_error = joint_objective(optimized_delta_params)

            if final_error < initial_error:
                if verbose:
                    opt_type = "Secondary-only" if fix_primary_matrix else "Joint"
                    print(f"   {opt_type} delta optimization: {initial_error:.4f} -> {final_error:.4f} pixels")
                    if initial_error > 0:
                        improvement = (initial_error - final_error) / initial_error * 100
                        print(f"   Improvement: {improvement:.1f}%")

                    param_offset = 0
                    delta_info = []
                    for i, should_optimize in enumerate(optimize_flags):
                        if should_optimize:
                            matrix_delta_params = optimized_delta_params[param_offset:param_offset + 6]
                            trans_delta = np.max(np.abs(matrix_delta_params[:3]))
                            rot_delta = np.max(np.abs(matrix_delta_params[3:]))
                            delta_info.append(
                                f"{matrix_names[i]}: translation {trans_delta:.4f}m, rotation {rot_delta:.4f}rad"
                            )
                            param_offset += 6

                    if len(delta_info) == 1:
                        print(f"   Max {delta_info[0]}")
                    else:
                        print(f"   Max deltas - {' | '.join(delta_info)}")

                return result_matrices[0], result_matrices[1], initial_error, final_error
            else:
                if verbose:
                    opt_type = "Secondary-only" if fix_primary_matrix else "Joint"
                    print(f"   {opt_type} delta optimization did not improve: "
                          f"{initial_error:.4f} -> {final_error:.4f} pixels")
                    print(f"   Keeping initial matrices")
                return initial_primary, initial_secondary, initial_error, initial_error

        except Exception as opt_e:
            if verbose:
                print(f"   Delta optimization failed: {opt_e}")
            initial_error = joint_objective(initial_delta_params)
            return initial_primary, initial_secondary, initial_error, initial_error

    # ============================================================================
    # Shared validation helpers
    # ============================================================================

    def _validate_handeye_data(self) -> bool:
        """
        Verify that the basic prerequisites for hand-eye calibration are present
        (images, robot poses with matching count, intrinsics, pattern).
        """
        if not self.images or len(self.images) == 0:
            print("❌ No images loaded")
            return False

        if not self.end2base_matrices or len(self.end2base_matrices) == 0:
            print("❌ No end-effector to base transformation matrices")
            return False

        if len(self.images) != len(self.end2base_matrices):
            print(f"❌ Mismatch: {len(self.images)} images vs "
                  f"{len(self.end2base_matrices)} transformation matrices")
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

        print("✅ All required data for hand-eye calibration is available")
        return True

    def _is_handeye_calibrated(self) -> bool:
        """Return True iff calibration is complete and the primary matrix is set."""
        return self.is_calibrated() and self._primary_matrix is not None

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
