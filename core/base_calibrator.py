"""
Base Calibration Module
=======================

This module provides the base class for all camera calibration types.
It contains common functionality shared across different calibration methods.

BaseCalibrator is an abstract base class that defines:
- Common data structures for images, patterns, and results
- Common methods for image handling, pattern detection, and visualization
- Abstract methods that must be implemented by specialized calibrators

This design eliminates code duplication and provides a consistent interface
for all calibration types while allowing specialized functionality.
"""

import os
import json
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from .calibration_patterns import CalibrationPattern


class BaseCalibrator(ABC):
    """
    Abstract base class for camera calibration operations.
    
    This class provides common functionality shared across different calibration types:
    - Image loading and management
    - Pattern detection and visualization
    - Results management and error calculation
    
    Specialized calibrators inherit from this class and implement specific calibration algorithms.
    """
    
    def __init__(self, images=None, image_paths=None, calibration_pattern=None, pattern_type=None):
        """
        Initialize BaseCalibrator with common parameters.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            pattern_type: Pattern type string for backwards compatibility or None
        """
        # Images and related parameters (common to all calibrators)
        self.images = None                    # List of image arrays
        self.image_paths = None              # List of image file paths
        self.image_points = None             # List of detected 2D points for each image
        self.point_ids = None                # List of detected point IDs for each image (for ChArUco etc.)
        self.object_points = None            # List of corresponding 3D object points
        self.image_size = None               # Image size (width, height)
        
        # Calibration pattern and related parameters (common to all calibrators)
        self.calibration_pattern = None      # CalibrationPattern instance
        self.pattern_type = None             # Pattern type string
        self.pattern_params = None           # Pattern-specific parameters dict
        
        # Common results and status (shared across calibration types)
        self.rvecs = None                    # Rotation vectors for each image
        self.tvecs = None                    # Translation vectors for each image
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
                    obj_pts = self.calibration_pattern.generate_object_points(point_ids)
                else:
                    obj_pts = self.calibration_pattern.generate_object_points()
                
                self.object_points.append(obj_pts)
                successful_detections += 1
                
                if verbose:
                    print(f"Image {i}: ✅ Detected {len(img_pts)} features")
            else:
                if verbose:
                    print(f"Image {i}: ❌ No pattern detected")
        
        if successful_detections < 3:
            print(f"Insufficient detections: need at least 3, got {successful_detections}")
            return False
        
        if verbose:
            print(f"Successfully detected pattern in {successful_detections}/{len(self.images)} images")
        
        return True
    
    def is_calibrated(self) -> bool:
        """Check if calibration has been completed successfully."""
        return self.calibration_completed
    
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
            
            # Get point IDs for this image if available
            current_point_ids = None
            if hasattr(self, 'point_ids') and self.point_ids and i < len(self.point_ids):
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
    
    def draw_axes_on_undistorted_images(self, axis_length: Optional[float] = None, 
                                      camera_matrix: Optional[np.ndarray] = None,
                                      distortion_coefficients: Optional[np.ndarray] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Draw 3D axes on undistorted images to verify calibration accuracy.
        
        Args:
            axis_length: Length of axes in world units. If None, calculates from pattern dimensions
            camera_matrix: Camera matrix to use for projection
            distortion_coefficients: Distortion coefficients to use for undistortion
            
        Returns:
            List of tuples (filename_without_extension, debug_image_array)
        """
        if not self.is_calibrated():
            raise ValueError("Calibration not completed. Run calibrate() first.")
        
        if not self.images or not self.object_points or not self.image_points:
            raise ValueError("No calibration data available.")
        
        if not self.rvecs or not self.tvecs:
            raise ValueError("No extrinsic parameters available. Ensure calibration completed successfully.")
        
        # Use provided camera parameters or try to get from calibrator
        if camera_matrix is None:
            camera_matrix = getattr(self, 'camera_matrix', None)
        if distortion_coefficients is None:
            distortion_coefficients = getattr(self, 'distortion_coefficients', None)
            
        if camera_matrix is None or distortion_coefficients is None:
            raise ValueError("Camera matrix and distortion coefficients must be provided or available from calibration")
        
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
            undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
            
            # Project 3D axis points to image plane
            axis_2d, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, camera_matrix, 
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
    
    # Abstract methods that must be implemented by specialized calibrators
    @abstractmethod
    def calibrate(self, **kwargs) -> float:
        """
        Perform calibration using the specific algorithm.
        Must be implemented by subclasses.
        
        Returns:
            float: RMS calibration error
        """
        pass
    
    @abstractmethod
    def save_results(self, save_directory: str) -> None:
        """
        Save calibration results to files.
        Must be implemented by subclasses.
        
        Args:
            save_directory: Directory to save results
        """
        pass
