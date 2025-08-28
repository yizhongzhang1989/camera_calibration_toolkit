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
from .utils import FilenameManager


class BaseCalibrator(ABC):
    """
    Abstract base class for camera calibration operations.
    
    This class provides common functionality shared across different calibration types:
    - Image loading and management
    - Pattern detection and visualization
    - Results management and error calculation
    
    Specialized calibrators inherit from this class and implement specific calibration algorithms.
    """
    
    def __init__(self, images=None, image_paths=None, calibration_pattern=None):
        """
        Initialize BaseCalibrator with common parameters.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
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
        self.pattern_params = None           # Pattern-specific parameters dict
        
        # Common results and status (shared across calibration types)
        self.rvecs = None                    # Rotation vectors for each image
        self.tvecs = None                    # Translation vectors for each image
        self.rms_error = None                # Overall RMS reprojection error
        self.per_image_errors = None         # RMS error for each image
        self.calibration_completed = False   # Whether calibration has been completed successfully
        
        # Filename management for systematic duplicate handling
        self.filename_manager = None         # FilenameManager instance for unique filename generation
        
        # Initialize with provided data using smart constructor
        if image_paths is not None:
            self.set_images_from_paths(image_paths)
        elif images is not None:
            self.set_images_from_arrays(images)
        
        if calibration_pattern is not None:
            self.set_calibration_pattern(calibration_pattern)
    
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
            
            # Initialize filename manager for systematic duplicate handling
            self.filename_manager = FilenameManager(image_paths)
                
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
    
    def set_calibration_pattern(self, pattern: CalibrationPattern, **pattern_params):
        """
        Set calibration pattern and related parameters.
        
        Args:
            pattern: CalibrationPattern instance
            **pattern_params: Additional pattern parameters
        """
        self.calibration_pattern = pattern
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
        """Check if calibration has been completed successfully."""
        return self.calibration_completed
    
    def draw_pattern_on_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Draw detected calibration patterns on original images.
        
        Returns:
            List of tuples (filename_without_extension, debug_image_array) for successfully detected images only
        """
        if not self.images or not self.image_points:
            raise ValueError("No images or detected points available. Run detect_pattern_points() first.")
        
        debug_images = []
        
        # Iterate through all images - arrays are now aligned
        for i, img in enumerate(self.images):
            # Skip images with no detection (None entries)
            if self.image_points[i] is None:
                continue
                
            # Get the detection results for this image
            corners = self.image_points[i]
            current_point_ids = self.point_ids[i]
            
            # Create copy of original image
            debug_img = img.copy()
            
            # Draw pattern-specific visualization
            if hasattr(self.calibration_pattern, 'draw_corners'):
                debug_img = self.calibration_pattern.draw_corners(debug_img, corners, current_point_ids)
            else:
                # Fallback: draw circles at corner locations
                corners_2d = corners.reshape(-1, 2).astype(int)
                for corner in corners_2d:
                    cv2.circle(debug_img, tuple(corner), 5, (0, 255, 0), 2)
            
            # Get unique filename from filename manager to avoid duplicates
            if self.filename_manager:
                filename = self.filename_manager.get_unique_filename(i)
            else:
                # Fallback for cases without filename manager
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, debug_img))
        
        return debug_images
    
    def draw_axes_on_undistorted_images(self, axis_length: Optional[float] = None, 
                                      camera_matrix: Optional[np.ndarray] = None,
                                      distortion_coefficients: Optional[np.ndarray] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Draw 3D axes and corner points on undistorted images to verify calibration accuracy.
        
        This function projects both the coordinate system axes and the 3D corner points 
        of the calibration pattern onto the undistorted images, providing a comprehensive
        visualization of the calibration results.
        
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
        
        for i, img in enumerate(self.images):
            # Skip images with no detection (None entries)
            if (self.image_points[i] is None or self.object_points[i] is None or 
                self.rvecs[i] is None or self.tvecs[i] is None):
                continue
                
            # Get detection results for this image
            objp = self.object_points[i]
            imgp = self.image_points[i]
            rvec = self.rvecs[i] 
            tvec = self.tvecs[i]
            
            # Undistort the image
            undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients)
            
            # Project 3D corner points to undistorted image plane
            corner_2d_undistorted, _ = cv2.projectPoints(
                objp, rvec, tvec, camera_matrix,
                np.zeros((4, 1))  # No distortion for undistorted image
            )
            corner_2d_undistorted = corner_2d_undistorted.reshape(-1, 2)
            
            # Draw corner points with 3D coordinates
            for j, (corner_2d, corner_3d) in enumerate(zip(corner_2d_undistorted, objp)):
                x, y = int(corner_2d[0]), int(corner_2d[1])
                
                # Draw corner point as a circle
                cv2.circle(undistorted_img, (x, y), 6, (255, 255, 0), 2)  # Cyan circles
            
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
            
            # Add legend in top-left corner
            legend_y_start = 30
            cv2.putText(undistorted_img, 'Legend:', (10, legend_y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(undistorted_img, 'Red: X-axis', (10, legend_y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(undistorted_img, 'Green: Y-axis', (10, legend_y_start + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(undistorted_img, 'Blue: Z-axis', (10, legend_y_start + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(undistorted_img, 'Cyan: Corner points', (10, legend_y_start + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add pattern information in bottom-right corner
            info_text = f"Pattern points: {len(objp)}"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            info_x = undistorted_img.shape[1] - text_size[0] - 10
            info_y = undistorted_img.shape[0] - 10
            cv2.putText(undistorted_img, info_text, (info_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Get unique filename from filename manager to avoid duplicates
            if self.filename_manager:
                filename = self.filename_manager.get_unique_filename(i)
            else:
                # Fallback for cases without filename manager
                filename = f"image_{i:03d}"
            
            debug_images.append((filename, undistorted_img))
        
        return debug_images
    
    # Abstract methods that must be implemented by specialized calibrators
    @abstractmethod
    def calibrate(self, **kwargs) -> bool:
        """
        Perform calibration using the specific algorithm.
        Must be implemented by subclasses.
        
        Returns:
            bool: True if calibration succeeded, False if failed
            
        Note:
            After successful calibration, use getter methods to access results:
            - Calibration quality metrics (RMS errors, etc.)
            - Calibrated parameters (camera matrix, transforms, etc.)
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
