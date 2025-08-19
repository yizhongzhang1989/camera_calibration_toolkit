"""
Intrinsic Camera Calibration Module
===================================

This module handles single camera intrinsic parameter calibration.
It can calibrate camera matrix and distortion coefficients from calibration images.
Supports multiple calibration pattern types through abstraction.

Key Methods in IntrinsicCalibrator class:
- Smart constructor: Initialize with images and patterns directly
- calibrate_camera(): OpenCV-style calibration with organized member variables
- detect_pattern_points(): Pattern detection with automatic point collection

The new interface provides clean separation of data and processing:
    # Smart constructor approach
    calibrator = IntrinsicCalibrator(
        image_paths=paths,
        calibration_pattern=pattern
    )
    calibrator.detect_pattern_points()
    rms_error = calibrator.calibrate_camera(cameraMatrix=None, distCoeffs=None)
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
from .utils import (
    get_objpoints, 
    get_chessboard_corners,
    find_chessboard_corners,
    load_images_from_directory
)
from .calibration_patterns import CalibrationPattern, create_chessboard_pattern


class IntrinsicCalibrator:
    """
    Camera intrinsic parameter calibration class.
    
    This class provides methods to calibrate camera intrinsic parameters following
    OpenCV's calibrateCamera interface design with organized member variables:
    
    - Images and related: images, image_paths, image_points, object_points, image_size
    - Pattern and related: calibration_pattern, pattern_type, pattern_params
    - Output values: camera_matrix, distortion_coefficients, rvecs, tvecs, rms_error
    - Function args: cameraMatrix (initial), distCoeffs (initial), flags, criteria
    """
    
    def __init__(self, images=None, image_paths=None, calibration_pattern=None, pattern_type=None):
        """
        Initialize IntrinsicCalibrator with smart constructor arguments.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None  
            calibration_pattern: CalibrationPattern instance or None
            pattern_type: Pattern type string for backwards compatibility or None
        """
        # Images and related parameters (set as members)
        self.images = None                    # List of image arrays
        self.image_paths = None              # List of image file paths  
        self.image_points = None             # List of detected 2D points for each image
        self.point_ids = None                # List of detected point IDs for each image (for ChArUco etc.)
        self.object_points = None            # List of corresponding 3D object points
        self.image_size = None               # Image size (width, height)
        
        # Calibration pattern and related parameters (set as members)
        self.calibration_pattern = None      # CalibrationPattern instance
        self.pattern_type = None             # Pattern type string
        self.pattern_params = None           # Pattern-specific parameters dict
        
        # Output values and results (set as members)
        self.camera_matrix = None            # Calibrated camera matrix
        self.distortion_coefficients = None  # Calibrated distortion coefficients
        self.distortion_model = None         # Distortion model used for calibration
        self.rvecs = None                    # Rotation vectors for each image (extrinsics)
        self.tvecs = None                    # Translation vectors for each image (extrinsics)
        self.rms_error = None                # Overall RMS reprojection error
        self.per_image_errors = None         # RMS error for each image
        self.calibration_completed = False   # Whether calibration has been completed successfully
        
        # Initialize with provided data using smart constructor
        if image_paths is not None:
            self.set_images_from_paths(image_paths)
        elif images is not None:
            self.set_images_from_arrays(images)
        
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
        self.images = []
        
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Failed to load image: {path}")
                return False
            self.images.append(img)
        
        if self.images:
            self.image_size = (self.images[0].shape[1], self.images[0].shape[0])
            
        return True
    
    def set_images_from_arrays(self, images: List[np.ndarray]) -> bool:
        """
        Set images from numpy arrays.
        
        Args:
            images: List of image arrays
            
        Returns:
            bool: True if all images are valid
        """
        if not images:
            return False
            
        self.images = images
        self.image_paths = None  # Clear paths since we're using arrays
        
        # Validate all images have same size
        first_size = (images[0].shape[1], images[0].shape[0])
        for i, img in enumerate(images):
            if img is None:
                print(f"Image {i} is None")
                return False
            current_size = (img.shape[1], img.shape[0])
            if current_size != first_size:
                print(f"Image {i} size mismatch: expected {first_size}, got {current_size}")
                return False
        
        self.image_size = first_size
        return True
    
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
    
    def detect_pattern_points(self, verbose: bool = False) -> bool:
        """
        Detect calibration pattern in all images and extract point correspondences.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            bool: True if successful detection in at least 3 images
        """
        if self.images is None or self.calibration_pattern is None:
            raise ValueError("Images and calibration pattern must be set first")
        
        self.image_points = []
        self.point_ids = []
        self.object_points = []
        successful_detections = 0
        
        if verbose:
            print(f"Detecting patterns in {len(self.images)} images...")
        
        for i, img in enumerate(self.images):
            success, img_pts, point_ids = self.calibration_pattern.detect_corners(img)
            
            if success:
                self.image_points.append(img_pts)
                self.point_ids.append(point_ids)  # Store point IDs for visualization
                
                # Generate corresponding object points
                if self.calibration_pattern.is_planar:
                    if point_ids is not None:
                        obj_pts = self.calibration_pattern.generate_object_points(point_ids)
                    else:
                        obj_pts = self.calibration_pattern.generate_object_points()
                else:
                    obj_pts = self.calibration_pattern.generate_object_points(point_ids)
                
                self.object_points.append(obj_pts)
                successful_detections += 1
                
                if verbose:
                    print(f"Image {i}: ✅ Detected {len(img_pts)} features")
            else:
                if verbose:
                    print(f"Image {i}: ❌ Pattern detection failed")
        
        if successful_detections < 3:
            print(f"Insufficient detections: need at least 3, got {successful_detections}")
            return False
        
        if verbose:
            print(f"Successfully detected pattern in {successful_detections}/{len(self.images)} images")
        
        return True
    
    def calibrate_camera(self, 
                        cameraMatrix: Optional[np.ndarray] = None,
                        distCoeffs: Optional[np.ndarray] = None, 
                        flags: int = 0,
                        criteria: Optional[Tuple] = None,
                        verbose: bool = False) -> float:
        """
        Calibrate camera following OpenCV's calibrateCamera interface.
        
        This method uses the point correspondences stored in class members
        (image_points, object_points) to calibrate the camera.
        
        Args:
            cameraMatrix: Initial camera matrix (3x3). If None, will be estimated.
            distCoeffs: Initial distortion coefficients. If None, will be estimated.
            flags: Calibration flags (same as OpenCV's calibrateCamera)
            criteria: Termination criteria for iterative algorithms
            verbose: Whether to print detailed progress
            
        Returns:
            float: RMS reprojection error (0.0 if calibration failed)
            
        Note:
            Before calling this method, you must:
            1. Set images: set_images_from_paths() or set_images_from_arrays()
            2. Set pattern: set_calibration_pattern()  
            3. Detect points: detect_pattern_points()
            
        Distortion Model Flags:
            Use cv2.CALIB_* flags to control distortion models:
            - Standard (5 coeff): no additional flags 
            - Rational (8 coeff): cv2.CALIB_RATIONAL_MODEL
            - Thin Prism (12 coeff): cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
            - Tilted (14 coeff): cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
        """
        if self.image_points is None or self.object_points is None:
            raise ValueError("Point correspondences not available. Call detect_pattern_points() first.")
        
        if len(self.image_points) < 3:
            raise ValueError(f"Insufficient point correspondences: need at least 3, got {len(self.image_points)}")
        
        if self.image_size is None:
            raise ValueError("Image size not set")
        
        # Handle initial camera matrix
        initial_camera_matrix = cameraMatrix
        calibration_flags = flags
        
        # Auto-generate initial camera matrix for non-planar patterns if not provided
        if initial_camera_matrix is None and self.calibration_pattern is not None and not self.calibration_pattern.is_planar:
            fx = fy = max(self.image_size) * 1.2  # Rough estimate
            cx, cy = self.image_size[0] / 2, self.image_size[1] / 2
            initial_camera_matrix = np.array([[fx, 0, cx],
                                            [0, fy, cy],
                                            [0, 0, 1]], dtype=np.float32)
            calibration_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            
            if verbose:
                print("Non-planar pattern detected, using auto-generated initial camera matrix")
        
        # Add USE_INTRINSIC_GUESS flag if initial matrix provided
        if initial_camera_matrix is not None and not (calibration_flags & cv2.CALIB_USE_INTRINSIC_GUESS):
            calibration_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        
        if verbose:
            print(f"Running calibration with {len(self.image_points)} point correspondences")
            print(f"Image size: {self.image_size}")
            print(f"Calibration flags: {calibration_flags}")
            if initial_camera_matrix is not None:
                print(f"Using initial camera matrix:")
                print(initial_camera_matrix)
        
        # Perform calibration
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, 
                self.image_points, 
                self.image_size, 
                initial_camera_matrix, 
                distCoeffs,
                flags=calibration_flags,
                criteria=criteria
            )
            
            if ret and mtx is not None:
                # Determine distortion model from flags
                distortion_model = 'standard'  # default
                if calibration_flags & cv2.CALIB_TILTED_MODEL:
                    distortion_model = 'tilted'
                elif calibration_flags & cv2.CALIB_THIN_PRISM_MODEL:
                    distortion_model = 'thin_prism'
                elif calibration_flags & cv2.CALIB_RATIONAL_MODEL:
                    distortion_model = 'rational'
                
                # Store results in class members
                self.camera_matrix = mtx
                self.distortion_coefficients = dist
                self.distortion_model = distortion_model
                self.rvecs = rvecs
                self.tvecs = tvecs
                self.rms_error = ret
                self.calibration_completed = True
                
                # Calculate per-image reprojection errors
                self.per_image_errors = []
                for obj_pts, img_pts, rvec, tvec in zip(self.object_points, self.image_points, rvecs, tvecs):
                    projected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
                    error = cv2.norm(img_pts, projected_pts, cv2.NORM_L2) / np.sqrt(len(projected_pts))
                    self.per_image_errors.append(error)
                
                if verbose:
                    print(f"✅ Calibration successful!")
                    print(f"RMS reprojection error: {ret:.4f} pixels")
                    print(f"Camera matrix:")
                    print(f"  fx: {mtx[0,0]:.2f}, fy: {mtx[1,1]:.2f}")
                    print(f"  cx: {mtx[0,2]:.2f}, cy: {mtx[1,2]:.2f}")
                    print(f"Distortion coefficients: {dist.flatten()}")
                    print(f"Per-image errors: {[f'{err:.4f}' for err in self.per_image_errors]}")
                
                return ret
            else:
                if verbose:
                    print("❌ Calibration failed - OpenCV returned invalid results")
                return 0.0
                
        except Exception as e:
            if verbose:
                print(f"❌ Calibration failed with exception: {e}")
            return 0.0
    
    # Getter methods for results
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """Get calibrated camera matrix."""
        return self.camera_matrix
    
    def get_distortion_coefficients(self) -> Optional[np.ndarray]:
        """Get calibrated distortion coefficients."""
        return self.distortion_coefficients
    
    def get_extrinsics(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """Get rotation and translation vectors for each image."""
        return self.rvecs, self.tvecs
    
    def get_reprojection_error(self) -> Tuple[Optional[float], Optional[List[float]]]:
        """Get overall and per-image reprojection errors."""
        return self.rms_error, self.per_image_errors
    
    def is_calibrated(self) -> bool:
        """Check if calibration has been completed successfully."""
        return self.calibration_completed
    
    # I/O methods for saving and loading calibration data
    def save_calibration(self, filepath: str, include_extrinsics: bool = True) -> None:
        """
        Save calibration results to JSON file.
        
        Args:
            filepath: Path to save the calibration data (should end with .json)
            include_extrinsics: Whether to include per-image extrinsics data
        """
        if not self.calibration_completed:
            raise ValueError("No calibration data to save. Run calibration first.")
        
        # Create calibration data dictionary
        calibration_data = {
            "calibration_info": {
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "image_count": len(self.image_points) if self.image_points else 0,
                "pattern_type": self.pattern_type,
                "distortion_model": self.distortion_model,
                "rms_error": float(self.rms_error)
            },
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "image_size": list(self.image_size) if self.image_size else None,
            "per_image_errors": [float(err) for err in self.per_image_errors] if self.per_image_errors else None
        }
        
        # Add pattern information if available
        if self.calibration_pattern:
            calibration_data["pattern_info"] = {
                "name": self.calibration_pattern.name,
                "is_planar": self.calibration_pattern.is_planar,
                "info": self.calibration_pattern.get_info()
            }
        
        # Add extrinsics if requested and available
        if include_extrinsics and self.rvecs is not None and self.tvecs is not None:
            calibration_data["extrinsics"] = {
                "rotation_vectors": [rvec.tolist() for rvec in self.rvecs],
                "translation_vectors": [tvec.tolist() for tvec in self.tvecs]
            }
            
            # Add image paths if available
            if self.image_paths:
                calibration_data["image_paths"] = self.image_paths
        
        # Save to file
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Calibration data saved to: {filepath}")
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration results from JSON file.
        
        Args:
            filepath: Path to the calibration data file
            
        Returns:
            bool: True if loaded successfully
        """
        if not os.path.exists(filepath):
            print(f"❌ Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            # Load required calibration results
            self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float32)
            self.distortion_coefficients = np.array(calibration_data["distortion_coefficients"], dtype=np.float32)
            self.rms_error = calibration_data["calibration_info"]["rms_error"]
            
            # Load optional data
            if "image_size" in calibration_data and calibration_data["image_size"]:
                self.image_size = tuple(calibration_data["image_size"])
            
            if "per_image_errors" in calibration_data and calibration_data["per_image_errors"]:
                self.per_image_errors = calibration_data["per_image_errors"]
            
            if "pattern_info" in calibration_data:
                self.pattern_type = calibration_data["pattern_info"]["name"]
            
            # Load extrinsics if available
            if "extrinsics" in calibration_data:
                extrinsics = calibration_data["extrinsics"]
                self.rvecs = [np.array(rvec, dtype=np.float32) for rvec in extrinsics["rotation_vectors"]]
                self.tvecs = [np.array(tvec, dtype=np.float32) for tvec in extrinsics["translation_vectors"]]
                
                if "image_paths" in calibration_data:
                    self.image_paths = calibration_data["image_paths"]
            
            self.calibration_completed = True
            
            print(f"✅ Calibration data loaded from: {filepath}")
            print(f"   RMS Error: {self.rms_error:.4f} pixels")
            print(f"   Image count: {calibration_data['calibration_info']['image_count']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load calibration data: {e}")
            return False
    
    def export_opencv_yaml(self, filepath: str) -> None:
        """
        Export calibration data in OpenCV YAML format for compatibility.
        
        Args:
            filepath: Path to save the YAML file (should end with .yml or .yaml)
        """
        if not self.calibration_completed:
            raise ValueError("No calibration data to export. Run calibration first.")
        
        import yaml
        
        # Create OpenCV-compatible data structure
        opencv_data = {
            "image_width": int(self.image_size[0]) if self.image_size else 0,
            "image_height": int(self.image_size[1]) if self.image_size else 0,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "dt": "d",
                "data": self.camera_matrix.flatten().tolist()
            },
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(self.distortion_coefficients),
                "dt": "d", 
                "data": self.distortion_coefficients.flatten().tolist()
            },
            "avg_reprojection_error": float(self.rms_error)
        }
        
        # Save to YAML file
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(opencv_data, f, default_flow_style=False)
        
        print(f"✅ OpenCV YAML data exported to: {filepath}")
    
    def draw_pattern_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw detected calibration patterns on original images.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.images or not self.image_points:
            raise ValueError("No images or detected points available. Run detection first.")
        
        debug_images = []
        
        for i, (img, corners) in enumerate(zip(self.images, self.image_points)):
            # Create copy of original image
            debug_img = img.copy()
            
            # Get point IDs for this image if available
            current_point_ids = None
            if self.point_ids and i < len(self.point_ids):
                current_point_ids = self.point_ids[i]
            
            # Draw pattern-specific visualization
            if hasattr(self.calibration_pattern, 'draw_corners'):
                debug_img = self.calibration_pattern.draw_corners(debug_img, corners, current_point_ids)
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
            raise ValueError("Calibration not completed. Run calibrate_camera() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
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
