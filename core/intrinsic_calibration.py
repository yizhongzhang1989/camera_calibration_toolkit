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
        self.object_points = None            # List of corresponding 3D object points
        self.image_size = None               # Image size (width, height)
        
        # Calibration pattern and related parameters (set as members)
        self.calibration_pattern = None      # CalibrationPattern instance
        self.pattern_type = None             # Pattern type string
        self.pattern_params = None           # Pattern-specific parameters dict
        
        # Output values and results (set as members)
        self.camera_matrix = None            # Calibrated camera matrix
        self.distortion_coefficients = None  # Calibrated distortion coefficients
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
        self.object_points = []
        successful_detections = 0
        
        if verbose:
            print(f"Detecting patterns in {len(self.images)} images...")
        
        for i, img in enumerate(self.images):
            success, img_pts, point_ids = self.calibration_pattern.detect_corners(img)
            
            if success:
                self.image_points.append(img_pts)
                
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
                # Store results in class members
                self.camera_matrix = mtx
                self.distortion_coefficients = dist
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
