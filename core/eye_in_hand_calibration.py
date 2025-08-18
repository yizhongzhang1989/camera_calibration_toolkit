"""
Eye-in-Hand Calibration Module
==============================

This module handles eye-in-hand calibration for robot-mounted cameras.
It calibrates the transformation between the camera and robot end-effector.

Modern Member-Based API:
-----------------------
The EyeInHandCalibrator class follows a clean member-based architecture similar to IntrinsicCalibrator:

1. Initialize with data: EyeInHandCalibrator(camera_matrix, distortion_coefficients, calibration_pattern, ...)
2. Load calibration data: load_calibration_data(directory) or set methods
3. Detect patterns: detect_pattern_points()  
4. Calibrate: calibrate(method, verbose) - uses member variables only
5. Optimize (optional): optimize_calibration(iterations, ftol_rel, verbose)
6. Generate debug images: draw_pattern_on_images(), draw_axes_on_undistorted_images(), draw_reprojection_on_images()
7. Access results: self.cam2end_matrix, self.rms_error, self.per_image_errors

Legacy Methods (Deprecated):
----------------------------
- calculate_reprojection_errors(image_paths, ...) - Use member-based approach instead
- _generate_reprojection_visualization(...) - Use draw_reprojection_on_images() instead
- _calibrate_extrinsic_parameters(...) - Use detect_pattern_points() instead

The legacy methods are maintained for backward compatibility but will print deprecation warnings.
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any, Union
from glob import glob

# Optional import for optimization
try:
    import nlopt
    HAS_NLOPT = True
except ImportError:
    nlopt = None
    HAS_NLOPT = False
    print("Warning: nlopt not available. Optimization methods will be disabled.")

from .utils import (
    get_objpoints,
    get_chessboard_corners,
    find_chessboard_corners,
    calculate_single_image_reprojection_error,
    xyz_rpy_to_matrix,
    matrix_to_xyz_rpy,
    inverse_transform_matrix,
    load_images_from_directory
)

from .calibration_patterns import CalibrationPattern


class EyeInHandCalibrator:
    """
    Eye-in-hand calibration class for robot-mounted cameras.
    
    This class calibrates the transformation between a camera mounted on a robot
    end-effector and the end-effector coordinate frame.
    
    Following the same pattern as IntrinsicCalibrator with organized member variables:
    - Images and related: images, image_paths, image_points, object_points, image_size
    - Pattern and related: calibration_pattern, pattern_type, pattern_params
    - Robot pose data: end2base_matrices, base2end_matrices, robot_poses
    - Camera intrinsics: camera_matrix, distortion_coefficients
    - Output values: cam2end_matrix, target2base_matrix, rvecs, tvecs, rms_error
    - Calibration status: calibration_completed, optimization_completed
    """
    
    def __init__(self, images=None, image_paths=None, robot_poses=None, 
                 camera_matrix=None, distortion_coefficients=None, 
                 calibration_pattern=None, pattern_type=None):
        """
        Initialize EyeInHandCalibrator with smart constructor arguments.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            robot_poses: List of robot poses (transformation matrices or dicts) or None
            camera_matrix: Camera intrinsic matrix or None
            distortion_coefficients: Distortion coefficients array or None
            calibration_pattern: CalibrationPattern instance or None
            pattern_type: Pattern type string for backwards compatibility or None
        """
        # Images and related parameters (input data as members)
        self.images = None                    # List of image arrays
        self.image_paths = None              # List of image file paths
        self.image_points = None             # List of detected 2D points for each image
        self.object_points = None            # List of corresponding 3D object points
        self.image_size = None               # Image size (width, height)
        
        # Calibration pattern and related parameters (input data as members)
        self.calibration_pattern = None      # CalibrationPattern instance
        self.pattern_type = None             # Pattern type string
        self.pattern_params = None           # Pattern-specific parameters dict
        
        # Robot pose data (input data as members)
        self.robot_poses = None              # List of robot pose data (original format)
        self.end2base_matrices = None        # List of end-effector to base transformation matrices
        self.base2end_matrices = None        # List of base to end-effector transformation matrices
        
        # Camera intrinsic parameters (input data as members)
        self.camera_matrix = None            # Camera intrinsic matrix
        self.distortion_coefficients = None  # Distortion coefficients
        
        # Extrinsic parameters (intermediate results as members)
        self.rvecs = None                    # Target to camera rotation vectors
        self.tvecs = None                    # Target to camera translation vectors
        self.target2cam_matrices = None      # Target to camera transformation matrices
        
        # Output values and results (output data as members)
        self.cam2end_matrix = None           # Camera to end-effector transformation matrix
        self.target2base_matrix = None       # Target to base transformation matrix
        self.rms_error = None                # Overall RMS reprojection error
        self.per_image_errors = None         # RMS error for each image
        self.calibration_completed = False   # Whether calibration has been completed successfully
        self.optimization_completed = False  # Whether optimization has been completed
        
        # Initialize with provided data using smart constructor
        if image_paths is not None:
            self.set_images_from_paths(image_paths)
        elif images is not None:
            self.set_images_from_arrays(images)
            
        if robot_poses is not None:
            self.set_robot_poses(robot_poses)
            
        if camera_matrix is not None:
            self.load_camera_intrinsics(camera_matrix, distortion_coefficients)
            
        if calibration_pattern is not None:
            self.set_calibration_pattern(calibration_pattern, pattern_type)
    
    def set_images_from_paths(self, image_paths: List[str]) -> bool:
        """
        Set images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            bool: True if all images loaded successfully
        """
        self.image_paths = image_paths
        try:
            self.images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Could not load image {path}")
                    return False
                self.images.append(img)
                
            # Set image size from first image
            if self.images:
                h, w = self.images[0].shape[:2]
                self.image_size = (w, h)
                
            print(f"Successfully loaded {len(self.images)} images")
            return True
        except Exception as e:
            print(f"Error loading images: {e}")
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
    
    def set_robot_poses(self, robot_poses: List[Union[Dict, np.ndarray]]) -> bool:
        """
        Set robot poses from list of poses (dicts or matrices).
        
        Args:
            robot_poses: List of robot poses (dicts with xyz/rpy or transformation matrices)
            
        Returns:
            bool: True if poses set successfully
        """
        self.robot_poses = robot_poses
        try:
            self.end2base_matrices = []
            self.base2end_matrices = []
            
            for pose in robot_poses:
                if isinstance(pose, dict):
                    # Handle dict format with xyz/rpy
                    if "end_xyzrpy" in pose:
                        end_xyzrpy_dict = pose["end_xyzrpy"]
                        xyzrpy = np.array([
                            end_xyzrpy_dict["x"], end_xyzrpy_dict["y"], end_xyzrpy_dict["z"],
                            end_xyzrpy_dict["rx"], end_xyzrpy_dict["ry"], end_xyzrpy_dict["rz"]
                        ])
                        end2base_matrix = xyz_rpy_to_matrix(xyzrpy)
                    else:
                        raise ValueError("Dict pose must contain 'end_xyzrpy' key")
                elif isinstance(pose, np.ndarray):
                    # Handle matrix format directly
                    if pose.shape == (4, 4):
                        end2base_matrix = pose
                    else:
                        raise ValueError("Matrix pose must be 4x4 transformation matrix")
                else:
                    raise ValueError("Robot pose must be dict or 4x4 numpy array")
                    
                self.end2base_matrices.append(end2base_matrix)
                self.base2end_matrices.append(inverse_transform_matrix(end2base_matrix))
                
            print(f"Successfully set {len(self.end2base_matrices)} robot poses")
            return True
        except Exception as e:
            print(f"Error setting robot poses: {e}")
            return False
    
    def set_calibration_pattern(self, pattern: CalibrationPattern, pattern_type: str = None, **pattern_params):
        """
        Set calibration pattern and related parameters.
        
        Args:
            pattern: CalibrationPattern instance
            pattern_type: Pattern type string (optional)
            **pattern_params: Additional pattern parameters
        """
        self.calibration_pattern = pattern
        self.pattern_type = pattern_type
        self.pattern_params = pattern_params
        
    def load_camera_intrinsics(self, camera_matrix: np.ndarray, 
                              distortion_coefficients: np.ndarray) -> None:
        """
        Load camera intrinsic parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coefficients: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
    
    def load_camera_intrinsics_from_file(self, intrinsics_directory: str) -> None:
        """
        Load camera intrinsic parameters from files.
        
        Args:
            intrinsics_directory: Directory containing mtx.json and dist.json files
        """
        camera_matrix_path = os.path.join(intrinsics_directory, 'mtx.json')
        distortion_path = os.path.join(intrinsics_directory, 'dist.json')
        
        if not os.path.exists(camera_matrix_path):
            raise FileNotFoundError(f"Camera matrix file not found: {camera_matrix_path}")
        
        if not os.path.exists(distortion_path):
            raise FileNotFoundError(f"Distortion coefficients file not found: {distortion_path}")
        
        # Load camera matrix
        with open(camera_matrix_path, 'r', encoding='utf-8') as f:
            mtx_dict = json.load(f)
            self.camera_matrix = np.array(mtx_dict["camera_matrix"])
        
        # Load distortion coefficients
        with open(distortion_path, 'r', encoding='utf-8') as f:
            dist_dict = json.load(f)
            self.distortion_coefficients = np.array(dist_dict["distortion_coefficients"])
    
    def load_calibration_data(self, data_directory: str, 
                            selected_indices: Optional[List[int]] = None) -> bool:
        """
        Load calibration images and corresponding robot poses, storing them as member variables.
        
        Args:
            data_directory: Directory containing images and pose JSON files
            selected_indices: Optional list of indices to select specific images
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Load images
            image_paths = load_images_from_directory(data_directory)
            
            # Load poses
            pose_json_paths = glob(os.path.join(data_directory, "*.json"))
            pose_json_paths = sorted(pose_json_paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0])
                                   if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
            
            if selected_indices is not None:
                image_paths = [image_paths[i] for i in selected_indices if i < len(image_paths)]
                pose_json_paths = [pose_json_paths[i] for i in selected_indices if i < len(pose_json_paths)]
            
            if len(image_paths) != len(pose_json_paths):
                raise ValueError(f"Number of images ({len(image_paths)}) does not match number of pose files ({len(pose_json_paths)})")
            
            # Load robot poses from JSON files
            robot_poses = []
            for pose_path in pose_json_paths:
                with open(pose_path, 'r', encoding='utf-8') as f:
                    pose_data = json.load(f)
                    robot_poses.append(pose_data)
            
            # Set data as member variables
            self.set_images_from_paths(image_paths)
            self.set_robot_poses(robot_poses)
            
            print(f"Successfully loaded {len(image_paths)} calibration images and poses")
            return True
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def detect_pattern_points(self, XX: int = None, YY: int = None, L: float = None) -> bool:
        """
        Detect calibration pattern points in all images, storing results as member variables.
        
        Args:
            XX: Number of chessboard corners along x-axis (optional if pattern is set)
            YY: Number of chessboard corners along y-axis (optional if pattern is set)
            L: Size of chessboard squares in meters (optional if pattern is set)
            
        Returns:
            bool: True if pattern detection completed successfully
        """
        if self.images is None:
            print("Error: No images loaded")
            return False
            
        try:
            # Get pattern parameters
            if self.calibration_pattern is not None:
                # Use calibration pattern if available - access attributes directly
                if hasattr(self.calibration_pattern, 'width'):
                    XX = XX or self.calibration_pattern.width
                if hasattr(self.calibration_pattern, 'height'):
                    YY = YY or self.calibration_pattern.height
                if hasattr(self.calibration_pattern, 'square_size'):
                    L = L or self.calibration_pattern.square_size
                    
                # Also check for parameters dict (alternative format)
                if hasattr(self.calibration_pattern, 'parameters'):
                    params = self.calibration_pattern.parameters
                    XX = XX or params.get('width', None)
                    YY = YY or params.get('height', None) 
                    L = L or params.get('square_size', None)
            
            if XX is None or YY is None or L is None:
                print("Error: Pattern parameters (XX, YY, L) must be provided or set via calibration pattern")
                return False
            
            # Generate 3D object points
            self.object_points = get_objpoints(len(self.images), XX, YY, L)
            
            # Find chessboard corners
            if self.image_paths:
                self.image_points = get_chessboard_corners(self.image_paths, XX, YY)
            else:
                # Handle case where we have image arrays but no paths
                self.image_points = []
                for i, img in enumerate(self.images):
                    corners = find_chessboard_corners(img, XX, YY)
                    if corners is not None:
                        self.image_points.append(corners)
                    else:
                        print(f"Warning: Could not detect pattern in image {i}")
            
            if len(self.image_points) != len(self.images):
                print(f"Warning: Only {len(self.image_points)}/{len(self.images)} images had detectable corners")
                
            if len(self.image_points) < 3:
                print("Error: Need at least 3 images with detected patterns")
                return False
                
            # Store pattern parameters for later use
            self.pattern_params = {'XX': XX, 'YY': YY, 'L': L}
            
            print(f"Successfully detected pattern points in {len(self.image_points)} images")
            return True
            
        except Exception as e:
            print(f"Error detecting pattern points: {e}")
            return False
    
    def _calculate_optimal_target2base_matrix(self, cam2end_4x4: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Calculate the single target2base matrix that minimizes overall reprojection error.
        
        This method finds the target2base transformation that best explains all the
        observed target positions across all images, rather than calculating a
        separate target2base for each image which would result in zero error.
        
        Args:
            cam2end_4x4: The camera to end-effector transformation matrix from calibration
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: 4x4 target to base transformation matrix
        """
        if verbose:
            print("Calculating optimal target2base matrix...")
        
        best_error = float('inf')
        best_target2base = None
        
        # Try using each image's target2cam transformation to estimate target2base
        # Then find the one that gives the smallest overall reprojection error
        candidate_target2base_matrices = []
        
        for i in range(len(self.target2cam_matrices)):
            # Calculate target2base using this image's measurements
            candidate_target2base = self.end2base_matrices[i] @ cam2end_4x4 @ self.target2cam_matrices[i]
            candidate_target2base_matrices.append(candidate_target2base)
        
        # Test each candidate to find the one with minimum overall reprojection error
        for candidate_idx, candidate_target2base in enumerate(candidate_target2base_matrices):
            total_error = 0.0
            valid_images = 0
            
            for i in range(len(self.image_points)):
                try:
                    # Calculate target2cam using the candidate target2base matrix
                    end2cam_matrix = np.linalg.inv(cam2end_4x4)
                    base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                    eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ candidate_target2base
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyeinhand_target2cam[:3, :3], 
                        eyeinhand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
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
                    best_target2base = candidate_target2base.copy()
                    if verbose:
                        print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f} (best so far)")
                elif verbose:
                    print(f"  Candidate {candidate_idx}: RMS error = {rms_error:.4f}")
        
        if best_target2base is not None:
            if verbose:
                print(f"✅ Optimal target2base matrix found with RMS error: {best_error:.4f}")
                print("Target2base transformation matrix:")
                print(best_target2base)
        else:
            if verbose:
                print("⚠️ Could not find optimal target2base matrix, using first candidate")
            best_target2base = candidate_target2base_matrices[0]
        
        return best_target2base
    
    def calibrate(self, method: int = cv2.CALIB_HAND_EYE_HORAUD, verbose: bool = False) -> float:
        """
        Perform eye-in-hand calibration following OpenCV's calibrateHandEye interface.
        
        This method uses the data stored in class members (images, robot poses, camera intrinsics)
        to perform hand-eye calibration.
        
        Args:
            method: Hand-eye calibration method. Available options:
                - cv2.CALIB_HAND_EYE_TSAI (default: Tsai method)
                - cv2.CALIB_HAND_EYE_PARK (Park method)
                - cv2.CALIB_HAND_EYE_HORAUD (Horaud method - default)
                - cv2.CALIB_HAND_EYE_ANDREFF (Andreff method)
                - cv2.CALIB_HAND_EYE_DANIILIDIS (Daniilidis method)
            verbose: Whether to print detailed information
            
        Returns:
            float: RMS reprojection error (0.0 if calibration failed)
            
        Note:
            Before calling this method, you must:
            1. Set images: set_images_from_paths() or load_calibration_data()  
            2. Set robot poses: set_robot_poses() or load_calibration_data()
            3. Set camera intrinsics: load_camera_intrinsics()
            4. Detect pattern points: detect_pattern_points()
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters have not been loaded")
            
        if self.end2base_matrices is None or len(self.end2base_matrices) == 0:
            raise ValueError("Robot poses have not been set")
            
        if self.image_points is None or self.object_points is None:
            raise ValueError("Pattern points not detected. Call detect_pattern_points() first.")
            
        if len(self.image_points) != len(self.end2base_matrices):
            raise ValueError(f"Mismatch: {len(self.image_points)} detected patterns vs {len(self.end2base_matrices)} robot poses")
            
        if len(self.image_points) < 3:
            raise ValueError(f"Insufficient data: need at least 3 image-pose pairs, got {len(self.image_points)}")
        
        try:
            if verbose:
                print(f"Running eye-in-hand calibration with {len(self.image_points)} image-pose pairs")
                print(f"Using method: {method}")
        
            # Calculate target to camera transformations from detected pattern points
            self.rvecs = []
            self.tvecs = []
            self.target2cam_matrices = []
            
            for i in range(len(self.image_points)):
                ret, rvec, tvec = cv2.solvePnP(self.object_points[i], self.image_points[i], 
                                              self.camera_matrix, self.distortion_coefficients)
                if ret:
                    self.rvecs.append(rvec)
                    self.tvecs.append(tvec)
                    
                    # Create target to camera transformation matrix
                    target2cam_matrix = np.eye(4)
                    target2cam_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
                    target2cam_matrix[:3, 3] = tvec[:, 0]
                    self.target2cam_matrices.append(target2cam_matrix)
                else:
                    raise ValueError(f"Could not solve PnP for image {i}")
            
            # Prepare data for hand-eye calibration
            end2base_Rs = np.array([matrix[:3, :3] for matrix in self.end2base_matrices])
            end2base_ts = np.array([matrix[:3, 3] for matrix in self.end2base_matrices])
            rvecs_array = np.array([rvec for rvec in self.rvecs])
            tvecs_array = np.array([tvec for tvec in self.tvecs])
            
            # Perform eye-in-hand calibration using OpenCV
            cam2end_R, cam2end_t = cv2.calibrateHandEye(
                end2base_Rs, end2base_ts, rvecs_array, tvecs_array, method)
            
            # Create 4x4 transformation matrix
            cam2end_4x4 = np.eye(4)
            cam2end_4x4[:3, :3] = cam2end_R
            cam2end_4x4[:3, 3] = cam2end_t[:, 0]
            
            # Store results in class members
            self.cam2end_matrix = cam2end_4x4
            self.calibration_completed = True
            
            # Calculate the single target2base matrix that minimizes reprojection error
            self.target2base_matrix = self._calculate_optimal_target2base_matrix(cam2end_4x4, verbose)
            
            # Calculate reprojection errors using the single target2base matrix
            self.per_image_errors = []
            total_error = 0.0
            valid_images = 0
            
            for i in range(len(self.image_points)):
                try:
                    # Calculate reprojected points using hand-eye calibration result and single target2base matrix
                    end2cam_matrix = np.linalg.inv(cam2end_4x4)
                    base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                    
                    # Target to camera using hand-eye calibration and the single target2base matrix
                    eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
                    
                    # Project 3D points to image
                    projected_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        eyeinhand_target2cam[:3, :3], 
                        eyeinhand_target2cam[:3, 3], 
                        self.camera_matrix, 
                        self.distortion_coefficients)
                    
                    # Calculate reprojection error for this image
                    error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                    self.per_image_errors.append(error)
                    total_error += error * error
                    valid_images += 1
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not calculate reprojection error for image {i}: {e}")
                    self.per_image_errors.append(float('inf'))
            
            # Calculate RMS error
            if valid_images > 0:
                self.rms_error = np.sqrt(total_error / valid_images)
            else:
                self.rms_error = float('inf')
            
            if verbose:
                print("✅ Eye-in-hand calibration completed successfully!")
                print(f"RMS reprojection error: {self.rms_error:.4f} pixels")
                print(f"Camera to end-effector transformation matrix:")
                print(f"{cam2end_4x4}")
                print(f"Per-image errors: {[f'{err:.4f}' for err in self.per_image_errors if not np.isinf(err)]}")

            return self.rms_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Eye-in-hand calibration failed: {e}")
            self.calibration_completed = False
            return 0.0
    
    # Getter methods for results (following IntrinsicCalibrator pattern)
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get the camera to end-effector transformation matrix."""
        return self.cam2end_matrix
    
    def get_extrinsics(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Get rotation and translation vectors for each image (target to camera)."""
        return self.rvecs, self.tvecs
    
    def get_reprojection_error(self) -> Tuple[Optional[float], Optional[List[float]]]:
        """Get overall and per-image reprojection errors."""
        return self.rms_error, self.per_image_errors
    
    def is_calibrated(self) -> bool:
        """Check if calibration has been completed successfully."""
        return self.calibration_completed
    
    def is_optimized(self) -> bool:
        """Check if optimization has been completed successfully."""
        return self.optimization_completed
    
    def draw_pattern_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw detected calibration patterns on original images.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.images or not self.image_points:
            raise ValueError("No images or detected points available. Run detect_pattern_points() first.")
        
        debug_images = []
        
        for i, (img, corners) in enumerate(zip(self.images, self.image_points)):
            # Create copy of original image
            debug_img = img.copy()
            
            # Draw pattern-specific visualization
            if hasattr(self.calibration_pattern, 'draw_corners'):
                debug_img = self.calibration_pattern.draw_corners(debug_img, corners)
            else:
                # Fallback: draw circles at corner locations
                corners_2d = corners.reshape(-1, 2).astype(int)
                for corner in corners_2d:
                    cv2.circle(debug_img, tuple(corner), 5, (0, 255, 0), 2)
            
            # Get original filename without path and extension
            if hasattr(self, 'image_paths') and self.image_paths and i < len(self.image_paths):
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            else:
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, debug_img))
        
        return debug_images
    
    def draw_axes_on_undistorted_images(self, axis_length: Optional[float] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Draw 3D axes on undistorted images to verify calibration accuracy.
        
        Args:
            axis_length: Length of axes in world units. If None, calculates from pattern dimensions
            
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.is_calibrated():
            raise ValueError("Calibration not completed. Run calibrate() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
        if not self.rvecs or not self.tvecs:
            raise ValueError("No extrinsic parameters available. Ensure calibration completed successfully.")
        
        # Calculate appropriate axis length based on pattern dimensions
        if axis_length is None:
            if hasattr(self.calibration_pattern, 'square_size'):
                if hasattr(self.calibration_pattern, 'width') and hasattr(self.calibration_pattern, 'height'):
                    # For chessboard: X and Y axes should span the entire chessboard
                    x_axis_length = (self.calibration_pattern.width - 1) * self.calibration_pattern.square_size
                    y_axis_length = (self.calibration_pattern.height - 1) * self.calibration_pattern.square_size
                    z_axis_length = self.calibration_pattern.square_size  # Z-axis is one square size
                else:
                    # Default fallback for patterns without width/height
                    x_axis_length = y_axis_length = self.calibration_pattern.square_size * 3
                    z_axis_length = self.calibration_pattern.square_size
            else:
                # Default for patterns without square_size
                x_axis_length = y_axis_length = z_axis_length = 0.05  # Default 5cm
        else:
            # If axis_length is provided, use it for all axes
            x_axis_length = y_axis_length = z_axis_length = axis_length
        
        debug_images = []
        
        # Define 3D axis points with different lengths for each axis
        axis_3d = np.float32([
            [0, 0, 0],                    # Origin
            [x_axis_length, 0, 0],        # X-axis (red) - full chessboard width
            [0, y_axis_length, 0],        # Y-axis (green) - full chessboard height
            [0, 0, -z_axis_length]        # Z-axis (blue) - one square size, negative to point up
        ]).reshape(-1, 3)
        
        for i, (img, objp, imgp, rvec, tvec) in enumerate(zip(
            self.images, self.object_points, self.image_points, self.rvecs, self.tvecs
        )):
            # Undistort the image
            undistorted_img = cv2.undistort(img, self.camera_matrix, self.distortion_coefficients)
            
            # Project 3D axis points to image plane
            axis_2d, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, self.camera_matrix, 
                np.zeros((4, 1))  # No distortion for undistorted image
            )
            axis_2d = axis_2d.reshape(-1, 2).astype(int)
            
            # Draw axes
            origin = tuple(axis_2d[0])
            x_end = tuple(axis_2d[1])
            y_end = tuple(axis_2d[2]) 
            z_end = tuple(axis_2d[3])
            
            # Draw axis lines with thicker lines for better visibility
            cv2.arrowedLine(undistorted_img, origin, x_end, (0, 0, 255), 5)  # X-axis: red
            cv2.arrowedLine(undistorted_img, origin, y_end, (0, 255, 0), 5)  # Y-axis: green
            cv2.arrowedLine(undistorted_img, origin, z_end, (255, 0, 0), 5)  # Z-axis: blue
            
            # Add labels with better positioning
            cv2.putText(undistorted_img, 'X', (x_end[0] + 15, x_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(undistorted_img, 'Y', (y_end[0] + 15, y_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(undistorted_img, 'Z', (z_end[0] + 15, z_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # Get original filename without path and extension
            if hasattr(self, 'image_paths') and self.image_paths and i < len(self.image_paths):
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            else:
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, undistorted_img))
        
        return debug_images
    
    def draw_reprojection_on_undistorted_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw pattern point reprojections on undistorted images using hand-eye calibration results.
        
        This shows the accuracy of the hand-eye calibration by comparing detected pattern points
        with points reprojected using the calibrated transformation matrix.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.is_calibrated():
            raise ValueError("Calibration not completed. Run calibrate() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
        if not self.target2cam_matrices or not self.end2base_matrices:
            raise ValueError("No transformation matrices available. Ensure calibration completed successfully.")
        
        debug_images = []
        
        for i, (img, objp, detected_corners) in enumerate(zip(
            self.images, self.object_points, self.image_points
        )):
            # Undistort the image
            undistorted_img = cv2.undistort(img, self.camera_matrix, self.distortion_coefficients)
            
            # Calculate reprojected points using hand-eye calibration
            try:
                # Method 1: Direct target to camera from calibration
                target2cam_direct = self.target2cam_matrices[i]
                reprojected_direct, _ = cv2.projectPoints(
                    objp, target2cam_direct[:3, :3], target2cam_direct[:3, 3],
                    self.camera_matrix, np.zeros((4, 1))  # No distortion for undistorted image
                )
                
                # Method 2: Target to camera via hand-eye calibration chain using single target2base matrix
                end2cam_matrix = np.linalg.inv(self.cam2end_matrix)
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                
                # Use the single target2base matrix calculated during calibration
                eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
                
                reprojected_eyeinhand, _ = cv2.projectPoints(
                    objp, eyeinhand_target2cam[:3, :3], eyeinhand_target2cam[:3, 3],
                    self.camera_matrix, np.zeros((4, 1))  # No distortion for undistorted image
                )
                
                # Draw detected corners in green (ground truth)
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(undistorted_img, tuple(corner), 8, (0, 255, 0), 2)
                
                # Draw direct reprojection in blue (from direct PnP)
                direct_2d = reprojected_direct.reshape(-1, 2).astype(int)
                for corner in direct_2d:
                    cv2.drawMarker(undistorted_img, tuple(corner), (255, 0, 0), 
                                 cv2.MARKER_CROSS, 12, 2)
                
                # Draw hand-eye reprojection in red (from hand-eye calibration)
                eyeinhand_2d = reprojected_eyeinhand.reshape(-1, 2).astype(int)
                for corner in eyeinhand_2d:
                    cv2.drawMarker(undistorted_img, tuple(corner), (0, 0, 255), 
                                 cv2.MARKER_TRIANGLE_UP, 12, 2)
                
                # Add legend
                cv2.putText(undistorted_img, "Green: Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(undistorted_img, "Blue: Direct PnP", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(undistorted_img, "Red: Hand-Eye", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display error
                if self.per_image_errors and i < len(self.per_image_errors):
                    error_text = f"RMS Error: {self.per_image_errors[i]:.3f} px"
                    cv2.putText(undistorted_img, error_text, (10, undistorted_img.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Warning: Could not generate reprojection for image {i}: {e}")
                # Just draw detected corners if reprojection fails
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(undistorted_img, tuple(corner), 8, (0, 255, 0), 2)
            
            # Get original filename without path and extension
            if hasattr(self, 'image_paths') and self.image_paths and i < len(self.image_paths):
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            else:
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, undistorted_img))
        
        return debug_images
    
    def draw_reprojection_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw pattern point reprojections on original (distorted) images using hand-eye calibration results.
        
        This shows the accuracy of the hand-eye calibration by comparing detected pattern points
        with points reprojected using the calibrated transformation matrix on the original images.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.is_calibrated():
            raise ValueError("Calibration not completed. Run calibrate() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
        if not self.target2cam_matrices or not self.end2base_matrices:
            raise ValueError("No transformation matrices available. Ensure calibration completed successfully.")
        
        debug_images = []
        
        for i, (img, objp, detected_corners) in enumerate(zip(
            self.images, self.object_points, self.image_points
        )):
            # Use original (distorted) image
            original_img = img.copy()
            
            # Calculate reprojected points using hand-eye calibration
            try:
                # Method 1: Direct target to camera from calibration
                target2cam_direct = self.target2cam_matrices[i]
                reprojected_direct, _ = cv2.projectPoints(
                    objp, target2cam_direct[:3, :3], target2cam_direct[:3, 3],
                    self.camera_matrix, self.distortion_coefficients  # Include distortion for original image
                )
                
                # Method 2: Target to camera via hand-eye calibration chain using single target2base matrix
                end2cam_matrix = np.linalg.inv(self.cam2end_matrix)
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                
                # Use the single target2base matrix calculated during calibration
                eyeinhand_target2cam = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
                
                reprojected_eyeinhand, _ = cv2.projectPoints(
                    objp, eyeinhand_target2cam[:3, :3], eyeinhand_target2cam[:3, 3],
                    self.camera_matrix, self.distortion_coefficients  # Include distortion for original image
                )
                
                # Draw detected corners in green (ground truth)
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(original_img, tuple(corner), 8, (0, 255, 0), 2)
                
                # Draw direct reprojection in blue (from direct PnP)
                direct_2d = reprojected_direct.reshape(-1, 2).astype(int)
                for corner in direct_2d:
                    cv2.drawMarker(original_img, tuple(corner), (255, 0, 0), 
                                 cv2.MARKER_CROSS, 12, 2)
                
                # Draw hand-eye reprojection in red (from hand-eye calibration)
                eyeinhand_2d = reprojected_eyeinhand.reshape(-1, 2).astype(int)
                for corner in eyeinhand_2d:
                    cv2.drawMarker(original_img, tuple(corner), (0, 0, 255), 
                                 cv2.MARKER_TRIANGLE_UP, 12, 2)
                
                # Add legend
                cv2.putText(original_img, "Green: Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(original_img, "Blue: Direct PnP", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(original_img, "Red: Hand-Eye", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display error
                if self.per_image_errors and i < len(self.per_image_errors):
                    error_text = f"RMS Error: {self.per_image_errors[i]:.3f} px"
                    cv2.putText(original_img, error_text, (10, original_img.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Warning: Could not generate reprojection for image {i}: {e}")
                # Just draw detected corners if reprojection fails
                detected_2d = detected_corners.reshape(-1, 2).astype(int)
                for corner in detected_2d:
                    cv2.circle(original_img, tuple(corner), 8, (0, 255, 0), 2)
            
            # Get original filename without path and extension
            if hasattr(self, 'image_paths') and self.image_paths and i < len(self.image_paths):
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
            else:
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, original_img))
        
        return debug_images
    
    # =============================================================================
    # DEPRECATED METHODS - Use modern member-based API instead
    # =============================================================================
    
    def calculate_reprojection_errors(self, image_paths: List[str], 
                                    base2end_matrices: List[np.ndarray],
                                    end2base_matrices: List[np.ndarray],
                                    rvecs_target2cam: List[np.ndarray],
                                    tvecs_target2cam: List[np.ndarray],
                                    XX: int, YY: int, L: float,
                                    vis: bool = False, 
                                    save_dir: Optional[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        DEPRECATED: Calculate reprojection errors using external parameters.
        
        This method is deprecated and maintained only for backward compatibility.
        Use the modern member-based approach instead:
        1. Use detect_pattern_points() to populate member variables
        2. Use calibrate() to perform calibration  
        3. Access self.rms_error and self.per_image_errors for results
        4. Use draw_reprojection_on_images() for visualization
        
        Args:
            image_paths: List of calibration image paths
            base2end_matrices: List of base to end-effector transformation matrices
            end2base_matrices: List of end-effector to base transformation matrices
            rvecs_target2cam: Rotation vectors from target to camera
            tvecs_target2cam: Translation vectors from target to camera
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            vis: Whether to generate visualization images
            save_dir: Directory to save visualization images
            
        Returns:
            Tuple of (reprojection_errors, target2base_matrices)
        """
        print("⚠️  WARNING: calculate_reprojection_errors() is deprecated.")
        print("   Use modern member-based API: detect_pattern_points() + calibrate() + self.rms_error")
        
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        end2cam_4x4 = np.linalg.inv(self.cam2end_matrix)
        errors = []
        target2base_matrices = []
        
        print("Calculating reprojection errors...")
        
        for i in range(len(end2base_matrices)):
            # Calculate target to camera from image detection
            target2cam_4x4 = xyz_rpy_to_matrix(
                tvecs_target2cam[i].flatten().tolist() + rvecs_target2cam[i].flatten().tolist())
            
            # Calculate target to base transformation
            target2base_4x4 = end2base_matrices[i] @ self.cam2end_matrix @ target2cam_4x4
            target2base_matrices.append(target2base_4x4)
            
            # Calculate target to camera using eye-in-hand calibration
            eyeinhand_target2cam_4x4 = end2cam_4x4 @ base2end_matrices[i] @ target2base_4x4
            
            # Calculate reprojection error
            error = calculate_single_image_reprojection_error(
                image_paths[i], 
                eyeinhand_target2cam_4x4[:3, :3], 
                eyeinhand_target2cam_4x4[:3, 3], 
                self.camera_matrix, self.distortion_coefficients, 
                XX, YY, L)
            errors.append(error)
            
            # Generate visualization if requested
            if vis and save_dir:
                self._generate_reprojection_visualization(
                    image_paths[i], eyeinhand_target2cam_4x4, XX, YY, L, save_dir, 
                    suffix="eyeinhand")
        
        return np.array(errors), target2base_matrices
    
    def _generate_reprojection_visualization(self, image_path: str, target2cam_4x4: np.ndarray,
                                           XX: int, YY: int, L: float, save_dir: str,
                                           suffix: str = "") -> None:
        """
        DEPRECATED: Generate reprojection visualization using external parameters.
        
        This method is deprecated and maintained only for backward compatibility.
        Use draw_reprojection_on_images() instead for modern member-based visualization.
        
        Args:
            image_path: Path to the image
            target2cam_4x4: Target to camera transformation matrix
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            save_dir: Directory to save the visualization
            suffix: Suffix for the output filename
        """
        print("⚠️  WARNING: _generate_reprojection_visualization() is deprecated.")
        print("   Use modern member-based API: draw_reprojection_on_images()")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Generate 3D object points
        objpoints = get_objpoints(1, XX, YY, L)
        
        # Project 3D points to image
        reprojected_imgpoints, _ = cv2.projectPoints(
            objpoints[0], target2cam_4x4[:3, :3], target2cam_4x4[:3, 3], 
            self.camera_matrix, self.distortion_coefficients)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img_draw = img.copy()
        
        # Detect actual corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, detected_corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            detected_corners = cv2.cornerSubPix(gray, detected_corners, (11, 11), (-1, -1), criteria)
            
            # Draw actual corners in green
            for corner in detected_corners:
                cv2.circle(img_draw, tuple(corner[0].astype(int)), 8, (0, 255, 0), 2)
            
            # Draw reprojected corners in red
            for point in reprojected_imgpoints:
                center = tuple(point[0].astype(int))
                cv2.drawMarker(img_draw, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            
            # Add legend
            cv2.putText(img_draw, "Green: Detected corners", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_draw, "Red: Reprojected corners", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save visualization
        image_name = os.path.basename(image_path).split('.')[0]
        output_name = f'{suffix}_reproject_{image_name}.png' if suffix else f'reproject_{image_name}.png'
        output_path = os.path.join(save_dir, output_name)
        cv2.imwrite(output_path, img_draw)
    
    # =============================================================================
    # END DEPRECATED METHODS
    # =============================================================================
    
    def optimize_calibration(self, iterations: int = 5, ftol_rel: float = 1e-6, verbose: bool = False) -> float:
        """
        Optimize eye-in-hand calibration using nonlinear optimization.
        
        This method refines the calibration results obtained from the initial calibration
        by minimizing the overall reprojection error across all images.
        
        Args:
            iterations: Number of optimization iterations (default: 5)
            ftol_rel: Relative tolerance for optimization (default: 1e-6)
            verbose: Whether to print detailed optimization information
            
        Returns:
            float: Final RMS reprojection error after optimization
            
        Note:
            This method requires the initial calibration to be completed first.
            All required data (images, poses, etc.) should already be loaded as member variables.
        """
        if not self.calibration_completed:
            raise ValueError("Initial calibration must be completed before optimization")
            
        if not HAS_NLOPT:
            if verbose:
                print("Warning: nlopt not available. Returning current calibration without optimization.")
            return self.rms_error
            
        if verbose:
            print("🔧 Starting calibration optimization...")
            print(f"   Initial RMS error: {self.rms_error:.4f} pixels")
            print(f"   Optimization iterations: {iterations}")
            print(f"   Convergence tolerance: {ftol_rel}")
        
        # Get pattern parameters 
        XX = self.pattern_params['XX']
        YY = self.pattern_params['YY'] 
        L = self.pattern_params['L']
        
        try:
            # Find the image with minimum reprojection error as optimization starting point
            min_error_idx = np.argmin([e for e in self.per_image_errors if not np.isinf(e)])
            if verbose:
                print(f"   Using image {min_error_idx} (error: {self.per_image_errors[min_error_idx]:.4f}) as starting point")
            
            # Initial values for optimization
            initial_cam2end = self.cam2end_matrix.copy()
            initial_target2base = self.target2base_matrix.copy()
            best_cam2end = initial_cam2end.copy()
            best_target2base = initial_target2base.copy() 
            best_error = self.rms_error
            
            # Iterative optimization
            for iteration in range(iterations):
                if verbose:
                    print(f"   Iteration {iteration + 1}/{iterations}:")
                
                # Optimize the single target2base matrix
                optimized_target2base = self._optimize_target2base_matrix(
                    best_target2base, best_cam2end, XX, YY, L, ftol_rel, verbose)
                
                # Calculate error with optimized target2base
                error_target2base = self._calculate_optimization_error(
                    best_cam2end, optimized_target2base, XX, YY, L)
                
                if error_target2base < best_error:
                    best_target2base = optimized_target2base
                    best_error = error_target2base
                    if verbose:
                        print(f"     Target2base optimized: {error_target2base:.4f} pixels")
                
                # Optimize cam2end matrix  
                optimized_cam2end = self._optimize_cam2end_matrix(
                    best_cam2end, best_target2base, XX, YY, L, ftol_rel, verbose)
                
                # Calculate error with optimized cam2end
                error_cam2end = self._calculate_optimization_error(
                    optimized_cam2end, best_target2base, XX, YY, L)
                
                if error_cam2end < best_error:
                    best_cam2end = optimized_cam2end
                    best_error = error_cam2end
                    if verbose:
                        print(f"     Cam2end optimized: {error_cam2end:.4f} pixels")
                
                if verbose:
                    print(f"     Best error so far: {best_error:.4f} pixels")
            
            # Update calibration results with optimized values
            self.cam2end_matrix = best_cam2end
            self.target2base_matrix = best_target2base
            
            # Recalculate reprojection errors with optimized parameters
            self._recalculate_reprojection_errors()
            
            self.optimization_completed = True
            
            if verbose:
                print(f"✅ Optimization completed!")
                print(f"   Final RMS error: {self.rms_error:.4f} pixels")
                print(f"   Improvement: {(self.rms_error - best_error):.4f} pixels")
            
            return self.rms_error
            
        except Exception as e:
            if verbose:
                print(f"❌ Optimization failed: {e}")
            return self.rms_error
    
    def _calculate_optimization_error(self, cam2end_matrix: np.ndarray, target2base_matrix: np.ndarray,
                                    XX: int, YY: int, L: float) -> float:
        """
        Calculate RMS reprojection error for given transformation matrices.
        
        Args:
            cam2end_matrix: Camera to end-effector transformation matrix
            target2base_matrix: Target to base transformation matrix
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            
        Returns:
            float: RMS reprojection error
        """
        total_error = 0.0
        valid_images = 0
        
        for i in range(len(self.image_points)):
            try:
                # Calculate target to camera using hand-eye calibration
                end2cam_matrix = np.linalg.inv(cam2end_matrix)
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                target2cam = end2cam_matrix @ base2end_matrix @ target2base_matrix
                
                # Project 3D points to image
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], 
                    target2cam[:3, :3], 
                    target2cam[:3, 3], 
                    self.camera_matrix, 
                    self.distortion_coefficients)
                
                # Calculate reprojection error
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error * error
                valid_images += 1
                
            except Exception:
                continue
        
        if valid_images > 0:
            return np.sqrt(total_error / valid_images)
        else:
            return float('inf')
    
    def _recalculate_reprojection_errors(self):
        """Recalculate per-image errors and overall RMS error with current matrices."""
        self.per_image_errors = []
        total_error = 0.0
        valid_images = 0
        
        for i in range(len(self.image_points)):
            try:
                # Calculate target to camera using optimized matrices
                end2cam_matrix = np.linalg.inv(self.cam2end_matrix)
                base2end_matrix = np.linalg.inv(self.end2base_matrices[i])
                target2cam = end2cam_matrix @ base2end_matrix @ self.target2base_matrix
                
                # Project 3D points to image
                projected_points, _ = cv2.projectPoints(
                    self.object_points[i], 
                    target2cam[:3, :3], 
                    target2cam[:3, 3], 
                    self.camera_matrix, 
                    self.distortion_coefficients)
                
                # Calculate reprojection error for this image
                error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                self.per_image_errors.append(error)
                total_error += error * error
                valid_images += 1
                
            except Exception:
                self.per_image_errors.append(float('inf'))
        
        # Update overall RMS error
        if valid_images > 0:
            self.rms_error = np.sqrt(total_error / valid_images)
        else:
            self.rms_error = float('inf')
    
    def _optimize_target2base_matrix(self, initial_target2base: np.ndarray, cam2end_matrix: np.ndarray,
                                   XX: int, YY: int, L: float, ftol_rel: float, verbose: bool) -> np.ndarray:
        """
        Optimize the target2base matrix using NLopt.
        
        Args:
            initial_target2base: Initial target2base transformation matrix
            cam2end_matrix: Fixed camera to end-effector transformation matrix
            XX: Number of chessboard corners along x-axis  
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: Optimized target2base transformation matrix
        """
        if not HAS_NLOPT:
            return initial_target2base
            
        try:
            import nlopt
            
            # Extract initial pose from matrix
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(initial_target2base)
            
            # Setup optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
            
            def objective(params, grad):
                x, y, z, roll, pitch, yaw = params
                target2base_matrix = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
                return self._calculate_optimization_error(cam2end_matrix, target2base_matrix, XX, YY, L)
            
            opt.set_min_objective(objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize
            try:
                opt_params = opt.optimize([x, y, z, roll, pitch, yaw])
                optimized_matrix = xyz_rpy_to_matrix(opt_params)
                
                if verbose:
                    initial_error = objective([x, y, z, roll, pitch, yaw], None)
                    final_error = objective(opt_params, None)
                    print(f"       Target2base: {initial_error:.4f} -> {final_error:.4f} pixels")
                
                return optimized_matrix
                
            except Exception as opt_e:
                if verbose:
                    print(f"       Target2base optimization failed: {opt_e}")
                return initial_target2base
                
        except ImportError:
            return initial_target2base
    
    def _optimize_cam2end_matrix(self, initial_cam2end: np.ndarray, target2base_matrix: np.ndarray,
                               XX: int, YY: int, L: float, ftol_rel: float, verbose: bool) -> np.ndarray:
        """
        Optimize the cam2end matrix using NLopt.
        
        Args:
            initial_cam2end: Initial camera to end-effector transformation matrix
            target2base_matrix: Fixed target to base transformation matrix
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis  
            L: Size of chessboard squares in meters
            ftol_rel: Relative tolerance for optimization
            verbose: Whether to print detailed information
            
        Returns:
            np.ndarray: Optimized camera to end-effector transformation matrix
        """
        if not HAS_NLOPT:
            return initial_cam2end
            
        try:
            import nlopt
            
            # Extract initial pose from matrix
            x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(initial_cam2end)
            
            # Setup optimization
            opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
            
            def objective(params, grad):
                x, y, z, roll, pitch, yaw = params
                cam2end_matrix = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
                return self._calculate_optimization_error(cam2end_matrix, target2base_matrix, XX, YY, L)
            
            opt.set_min_objective(objective)
            opt.set_ftol_rel(ftol_rel)
            
            # Optimize
            try:
                opt_params = opt.optimize([x, y, z, roll, pitch, yaw])
                optimized_matrix = xyz_rpy_to_matrix(opt_params)
                
                if verbose:
                    initial_error = objective([x, y, z, roll, pitch, yaw], None)
                    final_error = objective(opt_params, None)
                    print(f"       Cam2end: {initial_error:.4f} -> {final_error:.4f} pixels")
                
                return optimized_matrix
                
            except Exception as opt_e:
                if verbose:
                    print(f"       Cam2end optimization failed: {opt_e}")
                return initial_cam2end
                
        except ImportError:
            return initial_cam2end
    
    def save_results(self, save_directory: str) -> None:
        """
        Save calibration results to JSON file.
        
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
            "eye_in_hand_calibration": {
                "cam2end_matrix": self.cam2end_matrix.tolist(),
            }
        }
        
        if self.optimization_completed and self.target2base_matrix is not None:
            results["eye_in_hand_calibration"]["target2base_matrix"] = self.target2base_matrix.tolist()
        
        json_file_path = os.path.join(save_directory, "eye_in_hand_calibration_results.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Eye-in-hand calibration results saved to: {json_file_path}")
    
    def load_results(self, results_file: str) -> None:
        """
        Load calibration results from JSON file.
        
        Args:
            results_file: Path to the results JSON file
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Load camera intrinsics
        if "camera_intrinsics" in results:
            self.camera_matrix = np.array(results["camera_intrinsics"]["camera_matrix"])
            self.distortion_coefficients = np.array(results["camera_intrinsics"]["distortion_coefficients"])
        
        # Load eye-in-hand calibration results
        if "eye_in_hand_calibration" in results:
            self.cam2end_matrix = np.array(results["eye_in_hand_calibration"]["cam2end_matrix"])
            self.calibration_completed = True
            
            if "target2base_matrix" in results["eye_in_hand_calibration"]:
                self.target2base_matrix = np.array(results["eye_in_hand_calibration"]["target2base_matrix"])
                self.optimization_completed = True
        
        print(f"Eye-in-hand calibration results loaded from: {results_file}")
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the camera to end-effector transformation matrix.
        
        Returns:
            4x4 transformation matrix from camera to end-effector
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        return self.cam2end_matrix
