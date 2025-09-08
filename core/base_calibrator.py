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
    
    def __init__(self, images=None, image_paths=None, calibration_pattern=None, verbose: bool = False):
        """
        Initialize BaseCalibrator with common parameters.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
            verbose: Whether to print progress information during initialization (default: False)
        """
        # Images and related parameters (common to all calibrators)
        self.images = None                   # List of image arrays
        self.image_paths = None              # List of image file paths
        self.image_points = None             # List of detected 2D points for each image
        self.point_ids = None                # List of detected point IDs for each image (for ChArUco etc.)
        self.object_points = None            # List of corresponding 3D object points
        self.image_size = None               # Image size (width, height)
        
        # Calibration pattern and related parameters (common to all calibrators)
        self.calibration_pattern = None      # CalibrationPattern instance
        
        # Intrinsic-specific attributes
        self.camera_matrix = None            # Calibrated camera matrix
        self.distortion_coefficients = None  # Calibrated distortion coefficients
        self.distortion_model = None         # Distortion model used for calibration

        # Common results and status (shared across calibration types)
        self.rvecs = None                    # Rotation vectors of calibration pattern for each image
        self.tvecs = None                    # Translation vectors  of calibration pattern for each image
        self.rms_error = None                # Overall RMS reprojection error
        self.per_image_errors = None         # RMS error for each image
        self.calibration_completed = False   # Whether calibration has been completed successfully
        
        # Filename management for systematic duplicate handling
        self.filename_manager = None         # FilenameManager instance for unique filename generation
        
        # Initialize with provided data using smart constructor
        if image_paths is not None:
            self.set_images_from_paths(image_paths, verbose=verbose)
        elif images is not None:
            self.set_images_from_arrays(images, verbose=verbose)
        
        if calibration_pattern is not None:
            self.set_calibration_pattern(calibration_pattern)
    
    @abstractmethod
    def calibrate(self, **kwargs) -> Optional[dict]:
        """
        Perform calibration using the specific algorithm.
        Must be implemented by subclasses.
        
        Returns:
            Optional[dict]: Dictionary containing calibration results if successful, None if failed.
            The dictionary should contain key results specific to the calibration type:
            - For intrinsic calibration: camera_matrix, distortion_coefficients, rms_error, etc.
            - For hand-eye calibration: cam2end_matrix, target2base_matrix, rms_error, etc.
            
        Note:
            This new return pattern provides immediate access to results while maintaining
            backward compatibility through class member variables for detailed results.
        """
        pass

    def to_json(self) -> dict:
        """
        Serialize calibrator state to JSON-compatible dictionary.
        
        Saves the following parameters if they are not None:
        - camera_matrix
        - distortion_coefficients  
        - image_size
        - distortion_model
        - calibration_pattern
        - rms_error
        - per_image_errors
        - rvecs
        - tvecs
        - image_paths
        - image_points
        - point_ids
        - object_points
        
        Returns:
            dict: JSON-compatible dictionary containing calibrator state
        """
        data = {}
        
        # Save camera matrix
        if self.camera_matrix is not None:
            data['camera_matrix'] = self.camera_matrix.tolist()
            
        # Save distortion coefficients
        if self.distortion_coefficients is not None:
            data['distortion_coefficients'] = self.distortion_coefficients.tolist()
            
        # Save image size
        if self.image_size is not None:
            data['image_size'] = self.image_size
            
        # Save distortion model
        if self.distortion_model is not None:
            data['distortion_model'] = self.distortion_model
            
        # Save calibration pattern
        if self.calibration_pattern is not None:
            # Save pattern as JSON if it has to_json method, otherwise save pattern_id
            if hasattr(self.calibration_pattern, 'to_json'):
                data['calibration_pattern'] = self.calibration_pattern.to_json()
            else:
                # Fallback: save basic pattern info
                data['calibration_pattern'] = {
                    'pattern_id': getattr(self.calibration_pattern, 'pattern_id', 'unknown'),
                    'pattern_type': type(self.calibration_pattern).__name__
                }
                
        # Save RMS error
        if self.rms_error is not None:
            data['rms_error'] = float(self.rms_error)
            
        # Save per-image errors
        if self.per_image_errors is not None:
            data['per_image_errors'] = []
            for err in self.per_image_errors:
                if err is not None:
                    data['per_image_errors'].append(float(err))
                else:
                    data['per_image_errors'].append(None)
            
        # Save rotation vectors
        if self.rvecs is not None:
            data['rvecs'] = []
            for rvec in self.rvecs:
                if rvec is not None:
                    data['rvecs'].append(rvec.tolist())
                else:
                    data['rvecs'].append(None)
                    
        # Save translation vectors
        if self.tvecs is not None:
            data['tvecs'] = []
            for tvec in self.tvecs:
                if tvec is not None:
                    data['tvecs'].append(tvec.tolist())
                else:
                    data['tvecs'].append(None)
                    
        # Save image paths
        if self.image_paths is not None:
            data['image_paths'] = self.image_paths
            
        # Save image points
        if self.image_points is not None:
            data['image_points'] = []
            for img_pts in self.image_points:
                if img_pts is not None:
                    data['image_points'].append(img_pts.tolist())
                else:
                    data['image_points'].append(None)
                    
        # Save point IDs
        if self.point_ids is not None:
            data['point_ids'] = []
            for pt_ids in self.point_ids:
                if pt_ids is not None:
                    if isinstance(pt_ids, np.ndarray):
                        data['point_ids'].append(pt_ids.tolist())
                    else:
                        data['point_ids'].append(pt_ids)
                else:
                    data['point_ids'].append(None)
                    
        # Save object points
        if self.object_points is not None:
            data['object_points'] = []
            for obj_pts in self.object_points:
                if obj_pts is not None:
                    data['object_points'].append(obj_pts.tolist())
                else:
                    data['object_points'].append(None)
        
        return data
    
    def from_json(self, data: dict) -> None:
        """
        Deserialize calibrator state from JSON-compatible dictionary.
        
        Loads the following parameters if they exist in the data:
        - camera_matrix
        - distortion_coefficients
        - image_size
        - distortion_model
        - calibration_pattern
        - rms_error
        - per_image_errors
        - rvecs
        - tvecs
        - image_paths
        - image_points
        - point_ids
        - object_points
        
        Args:
            data: JSON-compatible dictionary containing calibrator state
        """
        # Load camera matrix
        if 'camera_matrix' in data:
            self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
            
        # Load distortion coefficients
        if 'distortion_coefficients' in data:
            self.distortion_coefficients = np.array(data['distortion_coefficients'], dtype=np.float32)
            
        # Load image size
        if 'image_size' in data:
            self.image_size = tuple(data['image_size'])
            
        # Load distortion model
        if 'distortion_model' in data:
            self.distortion_model = data['distortion_model']
            
        # Load calibration pattern
        if 'calibration_pattern' in data:
            pattern_data = data['calibration_pattern']
            if isinstance(pattern_data, dict):
                # Try to reconstruct pattern from JSON data
                try:
                    from .calibration_patterns import load_pattern_from_json
                    self.calibration_pattern = load_pattern_from_json(pattern_data)
                except Exception as e:
                    print(f"Warning: Could not load calibration pattern from JSON: {e}")
                    self.calibration_pattern = None
            else:
                print(f"Warning: Invalid calibration pattern data format")
                
        # Load RMS error
        if 'rms_error' in data:
            self.rms_error = float(data['rms_error'])
            
        # Load per-image errors
        if 'per_image_errors' in data:
            self.per_image_errors = []
            for err in data['per_image_errors']:
                if err is not None:
                    self.per_image_errors.append(float(err))
                else:
                    self.per_image_errors.append(None)
            
        # Load rotation vectors
        if 'rvecs' in data:
            self.rvecs = []
            for rvec_data in data['rvecs']:
                if rvec_data is not None:
                    self.rvecs.append(np.array(rvec_data, dtype=np.float32))
                else:
                    self.rvecs.append(None)
                    
        # Load translation vectors
        if 'tvecs' in data:
            self.tvecs = []
            for tvec_data in data['tvecs']:
                if tvec_data is not None:
                    self.tvecs.append(np.array(tvec_data, dtype=np.float32))
                else:
                    self.tvecs.append(None)
                    
        # Load image paths
        if 'image_paths' in data:
            self.image_paths = data['image_paths']
            
        # Load image points
        if 'image_points' in data:
            self.image_points = []
            for img_pts_data in data['image_points']:
                if img_pts_data is not None:
                    self.image_points.append(np.array(img_pts_data, dtype=np.float32))
                else:
                    self.image_points.append(None)
                    
        # Load point IDs
        if 'point_ids' in data:
            self.point_ids = []
            for pt_ids_data in data['point_ids']:
                if pt_ids_data is not None:
                    if isinstance(pt_ids_data, list) and len(pt_ids_data) > 0:
                        # Convert to numpy array if it looks like numeric data
                        try:
                            self.point_ids.append(np.array(pt_ids_data))
                        except:
                            # Keep as list if conversion fails
                            self.point_ids.append(pt_ids_data)
                    else:
                        self.point_ids.append(pt_ids_data)
                else:
                    self.point_ids.append(None)
                    
        # Load object points
        if 'object_points' in data:
            self.object_points = []
            for obj_pts_data in data['object_points']:
                if obj_pts_data is not None:
                    self.object_points.append(np.array(obj_pts_data, dtype=np.float32))
                else:
                    self.object_points.append(None)
        
        # Update calibration completion status if we have camera matrix
        if self.camera_matrix is not None and self.distortion_coefficients is not None:
            self.calibration_completed = True
    
    def generate_calibration_report(self, output_dir: str, verbose: bool = False, **kwargs) -> Optional[dict]:
        """
        Generate comprehensive calibration report with JSON data, debug images, and HTML viewer.
        
        Creates a complete calibration analysis package including:
        - JSON file with all calibration parameters and results
        - Debug images in 4 categories: original, pattern detection, undistorted, reprojection
        - Interactive HTML report for viewing and analyzing results
        
        Args:
            output_dir: Directory to save all report files
            verbose: Whether to print progress information (default: True)
            **kwargs: Additional options for report generation:
                - overwrite (bool): If True, overwrite existing files (default: True)
                - html_filename (str): Custom name for HTML report (default: "calibration_report.html")
                - json_filename (str): Custom name for JSON data (default: "calibration_data.json")
            
        Returns:
            Optional[dict]: Dictionary with paths to generated files if successful, None if failed:
                {
                    'html_report': 'path/to/report.html',
                    'json_data': 'path/to/data.json',
                    'image_dirs': {
                        'original_images': 'path/to/original/',
                        'pattern_detection': 'path/to/pattern/',
                        'undistorted': 'path/to/undistorted/',
                        'reprojection': 'path/to/reprojection/'
                    }
                }
        """
        try:
            # Check calibration status
            if not self.is_calibrated():
                if verbose:
                    print("❌ Cannot generate report: Calibration not completed")
                return None
            
            # Parse options
            overwrite = kwargs.get('overwrite', True)
            html_filename = kwargs.get('html_filename', 'calibration_report.html')
            json_filename = kwargs.get('json_filename', 'calibration_data.json')
            
            # Create output directory and subdirectories
            os.makedirs(output_dir, exist_ok=True)
            
            subdirs = {
                'original_images': os.path.join(output_dir, 'original_images'),
                'pattern_detection': os.path.join(output_dir, 'pattern_detection'),
                'undistorted': os.path.join(output_dir, 'undistorted'),
                'reprojection': os.path.join(output_dir, 'reprojection'),
                'analysis': os.path.join(output_dir, 'analysis')
            }
            
            for subdir in subdirs.values():
                os.makedirs(subdir, exist_ok=True)
            
            if verbose:
                print(f"📁 Creating calibration report in: {output_dir}")
            
            # Generate and save JSON data
            json_path = os.path.join(output_dir, json_filename)
            calibration_data = self.to_json()
            with open(json_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            if verbose:
                print(f"💾 Saved calibration data: {json_filename}")
            
            # Initialize filename manager for consistent naming
            if not self.filename_manager:
                if self.image_paths:
                    self.filename_manager = FilenameManager(self.image_paths)
                else:
                    # Generate default names for loaded images
                    default_names = [f"image_{i:03d}" for i in range(len(self.images))]
                    self.filename_manager = FilenameManager(default_names)
            
            # Generate images for each category
            image_counts = {}
            
            # 1. Original images
            if verbose:
                print("📸 Copying original images...")
            original_files = self._save_original_images(subdirs['original_images'])
            image_counts['original_images'] = len(original_files)
            
            # 2. Pattern detection images
            if verbose:
                print("🔍 Generating pattern detection images...")
            pattern_images = self.draw_pattern_on_images()
            pattern_files = self._save_debug_images(pattern_images, subdirs['pattern_detection'])
            image_counts['pattern_detection'] = len(pattern_files)
            
            # 3. Undistorted images with axes
            if verbose:
                print("📐 Generating undistorted images with axes...")
            undistorted_images = self.draw_axes_on_undistorted_images()
            undistorted_files = self._save_debug_images(undistorted_images, subdirs['undistorted'])
            image_counts['undistorted'] = len(undistorted_files)
            
            # 4. Reprojection analysis images
            if verbose:
                print("📊 Generating reprojection analysis...")
            reprojection_images = self.draw_reprojection_on_images()
            # Filter out None values for saving (but keep original list for indexing)
            valid_reprojection_images = [img for img in reprojection_images if img is not None]
            reprojection_files = self._save_debug_images(valid_reprojection_images, subdirs['reprojection'])
            image_counts['reprojection'] = len(reprojection_files)
            
            # 5. Point distribution analysis
            if verbose:
                print("📈 Generating point distribution analysis...")
            try:
                vis_img = self.vis_image_points_distribution()
                analysis_path = os.path.join(subdirs['analysis'], 'point_distribution.jpg')
                cv2.imwrite(analysis_path, vis_img)
                image_counts['analysis'] = 1
                if verbose:
                    print(f"   ✅ Point distribution saved: {os.path.basename(analysis_path)}")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Failed to generate point distribution: {e}")
                image_counts['analysis'] = 0
            
            # Generate HTML report
            if verbose:
                print("🌐 Creating HTML report...")
            html_path = os.path.join(output_dir, html_filename)
            self._generate_html_report(html_path, json_filename, subdirs, image_counts)
            
            if verbose:
                print(f"✅ Calibration report generated successfully!")
                print(f"   📄 HTML Report: {html_filename}")
                print(f"   📊 JSON Data: {json_filename}")
                print(f"   🖼️  Images: {sum(image_counts.values())} total")
                for category, count in image_counts.items():
                    print(f"      - {category.replace('_', ' ').title()}: {count}")
            
            return {
                'html_report': html_path,
                'json_data': json_path,
                'image_dirs': subdirs
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Failed to generate calibration report: {e}")
            return None

    def set_images_from_paths(self, image_paths: List[str], verbose: bool = False) -> bool:
        """
        Set images from file paths.
        
        Args:
            image_paths: List of image file paths
            verbose: Whether to print progress information (default: True)
            
        Returns:
            bool: True if all images loaded successfully
        """
        self.image_paths = image_paths
        try:
            self.images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is None:
                    if verbose:
                        print(f"Warning: Could not load image {path}")
                    return False
                self.images.append(img)
                
            # Set image size from first image
            if self.images:
                h, w = self.images[0].shape[:2]
                self.image_size = (w, h)
            
            # Initialize filename manager for systematic duplicate handling
            self.filename_manager = FilenameManager(image_paths)
                
            if verbose:
                print(f"Successfully loaded {len(self.images)} images")
            return True
        except Exception as e:
            if verbose:
                print(f"Error loading images: {e}")
            return False
    
    def set_images_from_arrays(self, images: List[np.ndarray], verbose: bool = False) -> bool:
        """
        Set images from numpy arrays.
        
        Args:
            images: List of image arrays
            verbose: Whether to print progress information (default: True)
            
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
            **pattern_params: Additional pattern parameters
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
    
    def draw_reprojection_on_images(self, camera_matrix: Optional[np.ndarray] = None,
                                   distortion_coefficients: Optional[np.ndarray] = None,
                                   pattern2camera_matrices: Optional[List[np.ndarray]] = None) -> List[Optional[Tuple[str, np.ndarray]]]:
        """
        Draw reprojected calibration pattern points on original images.
        
        This function projects the 3D object points back onto the images using the calibrated
        camera parameters and compares them with the detected corner points. Shows both detected
        corners (green) and reprojected points (red) with per-image reprojection error.
        
        Args:
            camera_matrix: Camera matrix to use for projection. If None, uses self.camera_matrix
            distortion_coefficients: Distortion coefficients. If None, uses self.distortion_coefficients
            pattern2camera_matrices: Optional list of 4x4 pattern-to-camera transformation matrices.
                                   If provided and matches image length, rvec and tvec will be extracted
                                   from these matrices instead of using self.rvecs/self.tvecs.
                                   Used for hand-eye calibration to draw reprojection from robot matrix chain.
                                   Individual matrices can be None for images without valid poses.
            
        Returns:
            List with same length as input images. Each element is either:
            - (filename_without_extension, debug_image_array) for successfully processed images
            - None for images that couldn't be processed (no detection, invalid matrices, etc.)
        """
        if not self.is_calibrated():
            raise ValueError("Calibration not completed. Run calibrate() first.")
            
        if not self.images or not self.image_points or not self.object_points:
            raise ValueError("No images or detected points available. Run detect_pattern_points() first.")
        
        # Validate pattern2camera_matrices if provided
        use_external_matrices = False
        if pattern2camera_matrices is not None:
            if len(pattern2camera_matrices) != len(self.images):
                raise ValueError(f"pattern2camera_matrices length ({len(pattern2camera_matrices)}) must match images length ({len(self.images)})")
            use_external_matrices = True
        else:
            # Use internal rvecs/tvecs - check they exist
            if not self.rvecs or not self.tvecs:
                raise ValueError("No extrinsic parameters available. Ensure calibration completed successfully or provide pattern2camera_matrices.")
        
        # Use provided camera parameters or try to get from calibrator
        if camera_matrix is None:
            camera_matrix = getattr(self, 'camera_matrix', None)
        if distortion_coefficients is None:
            distortion_coefficients = getattr(self, 'distortion_coefficients', None)
            
        if camera_matrix is None or distortion_coefficients is None:
            raise ValueError("Camera matrix and distortion coefficients must be provided or available from calibration")
        
        debug_images = []
        
        # Iterate through all images - maintain same length as input
        for i, img in enumerate(self.images):
            # Check if we can process this image
            can_process = True
            rvec = None
            tvec = None
            
            # Skip images with no detection
            if self.image_points[i] is None or self.object_points[i] is None:
                can_process = False
            
            # Get rvec and tvec from appropriate source
            if can_process and use_external_matrices:
                # Check if matrix is None or invalid
                if pattern2camera_matrices[i] is None:
                    can_process = False
                else:
                    pattern2cam_matrix = pattern2camera_matrices[i]
                    
                    # Check if matrix has correct shape and is valid
                    if (pattern2cam_matrix is None or 
                        not isinstance(pattern2cam_matrix, np.ndarray) or 
                        pattern2cam_matrix.shape != (4, 4)):
                        can_process = False
                    else:
                        try:
                            # Extract rotation matrix and translation vector
                            rotation_matrix = pattern2cam_matrix[:3, :3]
                            translation_vector = pattern2cam_matrix[:3, 3]
                            
                            # Convert rotation matrix to rotation vector
                            rvec, _ = cv2.Rodrigues(rotation_matrix)
                            tvec = translation_vector.reshape(-1, 1)
                        except Exception:
                            can_process = False
            elif can_process:
                # Use internal rvecs/tvecs
                if self.rvecs[i] is None or self.tvecs[i] is None:
                    can_process = False
                else:
                    rvec = self.rvecs[i]
                    tvec = self.tvecs[i]
            
            # If we can't process this image, append None and continue
            if not can_process:
                debug_images.append(None)
                continue
                
            # Get the detection results for this image
            detected_corners = self.image_points[i]
            object_points = self.object_points[i]
            
            # Create copy of original image
            debug_img = img.copy()
            
            # Project 3D object points to image plane
            reprojected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, distortion_coefficients
            )
            reprojected_points = reprojected_points.reshape(-1, 2)
            
            # Calculate reprojection error for this image
            detected_corners_2d = detected_corners.reshape(-1, 2)
            diff = detected_corners_2d - reprojected_points
            per_point_errors = np.sqrt(np.sum(diff**2, axis=1))
            mean_error = np.mean(per_point_errors)
            
            # Draw detected corners (green circles)
            for corner in detected_corners_2d.astype(int):
                cv2.circle(debug_img, tuple(corner), 6, (0, 255, 0), 2)  # Green circles
            
            # Draw reprojected points (red crosses)
            for point in reprojected_points.astype(int):
                cv2.drawMarker(debug_img, tuple(point), (0, 0, 255), 
                             cv2.MARKER_CROSS, 10, 2)  # Red crosses
            
            # Calculate axis length based on pattern dimensions (marker size)
            if hasattr(self.calibration_pattern, 'square_size'):
                axis_length = self.calibration_pattern.square_size
            elif hasattr(self.calibration_pattern, 'marker_size'):
                axis_length = self.calibration_pattern.marker_size
            else:
                axis_length = 0.02  # Default 2cm
            
            # Define 3D axis points at origin of calibration pattern coordinate system
            axis_3d = np.float32([
                [0, 0, 0],                  # Origin
                [axis_length, 0, 0],        # X-axis (red)
                [0, axis_length, 0],        # Y-axis (green)
                [0, 0, -axis_length]        # Z-axis (blue) - negative to point up
            ]).reshape(-1, 3)
            
            # Project 3D axis points to image plane
            axis_2d, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, camera_matrix, distortion_coefficients
            )
            axis_2d = axis_2d.reshape(-1, 2).astype(int)
            
            # Draw axes at pattern origin
            origin = tuple(axis_2d[0])
            x_end = tuple(axis_2d[1])
            y_end = tuple(axis_2d[2])
            z_end = tuple(axis_2d[3])
            
            # Draw axis lines
            cv2.arrowedLine(debug_img, origin, x_end, (0, 0, 255), 3)    # X-axis: red
            cv2.arrowedLine(debug_img, origin, y_end, (0, 255, 0), 3)    # Y-axis: green
            cv2.arrowedLine(debug_img, origin, z_end, (255, 0, 0), 3)    # Z-axis: blue
            
            # Add axis labels
            cv2.putText(debug_img, 'X', (x_end[0] + 10, x_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(debug_img, 'Y', (y_end[0] + 10, y_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(debug_img, 'Z', (z_end[0] + 10, z_end[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Add legend in top-right corner
            img_height, img_width = debug_img.shape[:2]
            legend_x = img_width - 250
            legend_y_start = 30
            
            # Add semi-transparent background for legend
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (legend_x - 10, legend_y_start - 20), 
                         (img_width - 10, legend_y_start + 85), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, debug_img, 0.3, 0, debug_img)
            
            cv2.putText(debug_img, 'Reprojection Analysis:', (legend_x, legend_y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_img, 'Green: Detected corners', (legend_x, legend_y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(debug_img, 'Red: Reprojected points', (legend_x, legend_y_start + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(debug_img, 'RGB Axes: Pattern origin', (legend_x, legend_y_start + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add per-image error in bottom-left corner
            error_text = f"RMS Error: {mean_error:.3f} pixels"
            text_size = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Add semi-transparent background for error text
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (5, img_height - text_size[1] - 15), 
                         (text_size[0] + 15, img_height - 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, debug_img, 0.2, 0, debug_img)
            
            cv2.putText(debug_img, error_text, (10, img_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
    
    def vis_image_points_distribution(self) -> np.ndarray:
        """
        Create a visualization showing the distribution of image points used for calibration.
        
        This function creates an image showing all detected image points from all images,
        with each image's points drawn in a different color to visualize the distribution
        and coverage of calibration points across the image plane.
        
        Returns:
            np.ndarray: Visualization image of the same size as image_size showing point distribution
            
        Raises:
            ValueError: If no images or image points are available, or if image_size is not set
        """
        if not self.images or not self.image_points:
            raise ValueError("No images or detected points available. Run detect_pattern_points() first.")
        
        if self.image_size is None:
            raise ValueError("Image size not available. Set image_size or load images first.")
        
        # Create blank image with white background
        vis_img = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
        
        # Count valid images (images with detected points)
        valid_images = []
        for i, points in enumerate(self.image_points):
            if points is not None:
                valid_images.append(i)
        
        # Generate colors for each valid image (smooth color transition)
        num_valid = len(valid_images)
        if num_valid == 0:
            # No valid images - return white image
            return vis_img
        
        # Generate colors using HSV for smooth transitions
        colors = []
        for i in range(num_valid):
            # Use HSV colorspace for smooth color transitions
            hue = int(180 * i / max(1, num_valid - 1))  # Spread across hue range (0-180 in OpenCV)
            hsv_color = np.uint8([[[hue, 255, 255]]])  # Full saturation and value
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr_color))
        
        # Draw points for each valid image
        color_idx = 0
        for i in valid_images:
            points = self.image_points[i]
            
            # Check if this image should be drawn in black (per_image_errors is None)
            if (self.per_image_errors is not None and 
                i < len(self.per_image_errors) and 
                self.per_image_errors[i] is not None):
                # Use assigned color for valid calibration data
                color = colors[color_idx]
                color_idx += 1
            else:
                # Use black for images not used in calibration
                color = (0, 0, 0)
            
            # Convert points to integer coordinates and draw circles
            points_2d = points.reshape(-1, 2).astype(int)
            for point in points_2d:
                # Check if point is within image bounds
                x, y = point[0], point[1]
                if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                    cv2.circle(vis_img, (x, y), 3, color, -1)  # Filled circles
                    cv2.circle(vis_img, (x, y), 4, (128, 128, 128), 1)  # Gray outline for visibility
        
        # Add title and legend
        title_text = f"Image Points Distribution ({num_valid} images)"
        cv2.putText(vis_img, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add color legend in top-right corner
        legend_x = self.image_size[0] - 200
        legend_y_start = 50
        
        cv2.putText(vis_img, 'Legend:', (legend_x, legend_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Show a few color samples
        sample_colors = min(5, num_valid)  # Show up to 5 color samples
        for i in range(sample_colors):
            y_pos = legend_y_start + 25 + i * 20
            if i < len(colors):
                color = colors[i]
                cv2.circle(vis_img, (legend_x + 10, y_pos), 5, color, -1)
                cv2.putText(vis_img, f'Image {valid_images[i] + 1}', (legend_x + 25, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add info about unused images (black points)
        if self.per_image_errors is not None:
            unused_count = sum(1 for i, err in enumerate(self.per_image_errors) 
                             if i < len(self.image_points) and self.image_points[i] is not None and err is None)
            if unused_count > 0:
                y_pos = legend_y_start + 25 + sample_colors * 20 + 10
                cv2.circle(vis_img, (legend_x + 10, y_pos), 5, (0, 0, 0), -1)
                cv2.putText(vis_img, f'Unused ({unused_count})', (legend_x + 25, y_pos + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add statistics in bottom-left corner
        total_points = sum(len(points.reshape(-1, 2)) for points in self.image_points if points is not None)
        stats_text = f"Total points: {total_points}"
        cv2.putText(vis_img, stats_text, (10, self.image_size[1] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        coverage_text = f"Image coverage: {num_valid}/{len(self.image_points)} images"
        cv2.putText(vis_img, coverage_text, (10, self.image_size[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis_img
    
    def _save_original_images(self, output_dir: str) -> List[str]:
        """
        Save original images to the specified directory.
        
        Args:
            output_dir: Directory to save original images
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if not self.images:
            return saved_files
        
        for i, img in enumerate(self.images):
            if img is None:
                continue
                
            # Get unique filename from filename manager
            if self.filename_manager:
                filename = self.filename_manager.get_unique_filename(i)
            else:
                filename = f"image_{i:03d}"
            
            output_path = os.path.join(output_dir, f"{filename}.jpg")
            
            # Save image
            cv2.imwrite(output_path, img)
            saved_files.append(output_path)
        
        return saved_files

    def _save_debug_images(self, debug_images: List[Tuple[str, np.ndarray]], output_dir: str) -> List[str]:
        """
        Save debug images to the specified directory.
        
        Args:
            debug_images: List of (filename, image_array) tuples
            output_dir: Directory to save images
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for filename, img in debug_images:
            output_path = os.path.join(output_dir, f"{filename}.jpg")
            cv2.imwrite(output_path, img)
            saved_files.append(output_path)
        
        return saved_files

    def _generate_html_report(self, html_path: str, json_filename: str, 
                             subdirs: dict, image_counts: dict):
        """
        Generate an interactive HTML report for viewing calibration results.
        
        Args:
            html_path: Path where HTML file should be saved
            json_filename: Name of the JSON data file (relative to HTML)
            subdirs: Dictionary mapping category names to directory paths
            image_counts: Dictionary mapping category names to image counts
        """
        # Get relative paths for HTML (relative to the HTML file location)
        html_dir = os.path.dirname(html_path)
        rel_subdirs = {}
        for category, abs_path in subdirs.items():
            rel_subdirs[category] = os.path.relpath(abs_path, html_dir)
        
        # Generate list of image filenames for each category
        image_lists = {}
        for category, subdir_path in subdirs.items():
            image_files = []
            if os.path.exists(subdir_path):
                for filename in sorted(os.listdir(subdir_path)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(filename)
            image_lists[category] = image_files
        
        # Get calibration summary data
        rms_error = getattr(self, 'rms_error', 'N/A')
        if isinstance(rms_error, (int, float)):
            rms_error = f"{rms_error:.4f}"
        
        pattern_name = "Unknown"
        if self.calibration_pattern:
            pattern_name = getattr(self.calibration_pattern, 'name', 'Unknown Pattern')
        
        total_images = len(self.images) if self.images else 0
        
        # Get current timestamp for report generation
        from datetime import datetime
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML template
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Calibration Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            width: 95%;
            min-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #495057;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .content {{
            padding: 30px;
            overflow-x: auto;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h3 {{
            color: #495057;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-item {{
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        .image-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .image-item img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            cursor: pointer;
        }}
        .image-item .caption {{
            padding: 15px;
            font-size: 0.9em;
            color: #6c757d;
            text-align: center;
        }}
        .image-comparison-table {{
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .image-comparison-table th {{
            background: #667eea;
            color: white;
            padding: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 0.9em;
            border-bottom: 2px solid #5a67d8;
        }}
        .image-comparison-table th:first-child {{
            width: 120px;
        }}
        .image-comparison-table th:not(:first-child) {{
            width: calc((100% - 120px) / 4);
        }}
        .image-comparison-table td {{
            padding: 3px;
            text-align: center;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }}
        .image-comparison-table tr:hover {{
            background: #f7fafc;
        }}
        .table-image {{
            width: 100%;
            max-width: 350px;
            height: auto;
            aspect-ratio: 4/3;
            object-fit: contain;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        .table-image:hover {{
            transform: scale(1.02);
        }}
        .image-id {{
            font-weight: bold;
            color: #4a5568;
            padding: 8px;
            background: #f8f9fa;
            border-right: 2px solid #e2e8f0;
            font-size: 0.85em;
            word-break: break-all;
            min-width: 80px;
            max-width: 120px;
        }}
        .image-unavailable {{
            color: #e53e3e;
            font-style: italic;
            padding: 40px 20px;
            background: #fed7d7;
            border: 2px dashed #e53e3e;
            border-radius: 8px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100px;
            font-weight: bold;
        }}
        .image-unavailable::before {{
            content: "⚠️";
            font-size: 2em;
            margin-bottom: 8px;
            display: block;
        }}
        .image-count-summary {{
            display: flex;
            justify-content: space-around;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        .count-item {{
            text-align: center;
        }}
        .count-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .count-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            width: 80%;
            max-width: 900px;
            max-height: 80%;
            margin-top: 50px;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #bbb;
        }}
        .download-btn {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 15px;
            transition: background 0.2s ease;
        }}
        .download-btn:hover {{
            background: #218838;
            text-decoration: none;
            color: white;
        }}
        .analysis-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .analysis-section p {{
            color: #6c757d;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        .analysis-image-container {{
            text-align: center;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
        }}
        .analysis-image {{
            max-width: 100%;
            max-height: 600px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            cursor: pointer;
            transition: transform 0.2s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .analysis-image:hover {{
            transform: scale(1.02);
        }}
        .analysis-legend {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 20px;
        }}
        .analysis-legend h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .analysis-legend ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .analysis-legend li {{
            margin: 8px 0;
            display: flex;
            align-items: center;
            color: #6c757d;
        }}
        .legend-color-sample {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 10px;
            border: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📷 Camera Calibration Report</h1>
            <p>Generated on {current_timestamp}</p>
        </div>
        
        <div class="summary">
            <h2>Calibration Summary</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{rms_error}</div>
                    <div class="stat-label">RMS Error (pixels)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_images}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{pattern_name}</div>
                    <div class="stat-label">Calibration Pattern</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{sum(image_counts.values())}</div>
                    <div class="stat-label">Generated Images</div>
                </div>
            </div>
            
            <a href="{json_filename}" class="download-btn" download>
                📄 Download Calibration Data (JSON)
            </a>
        </div>
        
        <div class="content">
            <div class="section">
                <h3>� Point Distribution Analysis</h3>
                
                <div class="analysis-section">
                    <p>This visualization shows the distribution of all detected calibration points across the image plane. 
                    Each image's points are displayed in a different color, helping to assess the coverage and quality of calibration data.</p>
                    
                    <div class="analysis-image-container">
                        <img src="{rel_subdirs.get('analysis', 'analysis')}/point_distribution.jpg" 
                             alt="Point Distribution Analysis" 
                             class="analysis-image" 
                             onclick="openModal(this.src)"
                             onerror="this.parentElement.innerHTML='<div class=\\'image-unavailable\\'>Analysis image not available</div>'">
                    </div>
                    
                    <div class="analysis-legend">
                        <h4>Color Legend:</h4>
                        <ul>
                            <li><span class="legend-color-sample" style="background: linear-gradient(90deg, red, orange, yellow, green, cyan, blue, purple);"></span> Different colors represent points from different images</li>
                            <li><span class="legend-color-sample" style="background: black;"></span> Black points indicate images detected but not used in calibration</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>�📸 Image Analysis</h3>
                
                <div class="image-count-summary">
                    <div class="count-item">
                        <div class="count-value">{image_counts.get('original_images', 0)}</div>
                        <div class="count-label">Original Images</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value">{image_counts.get('pattern_detection', 0)}</div>
                        <div class="count-label">Pattern Detection</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value">{image_counts.get('undistorted', 0)}</div>
                        <div class="count-label">Undistorted</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value">{image_counts.get('reprojection', 0)}</div>
                        <div class="count-label">Reprojection Analysis</div>
                    </div>
                </div>
                
                <table class="image-comparison-table">
                    <thead>
                        <tr>
                            <th>Image Filename</th>
                            <th>Original Image</th>
                            <th>Pattern Detection</th>
                            <th>Undistorted</th>
                            <th>Reprojection Analysis</th>
                        </tr>
                    </thead>
                    <tbody>
'''
        
        # Get all unique image filenames (base filenames without extensions) from original images
        all_image_filenames = set()
        
        # Use original images as the primary source for filenames
        original_files = image_lists.get('original_images', [])
        for img_file in original_files:
            # Extract base name without extension  
            base_name = img_file.split('.')[0]
            all_image_filenames.add(base_name)
        
        # Helper function for natural sorting (handles numbers correctly)
        import re
        def natural_sort_key(filename):
            """Convert a string into a list of strings and numbers for natural sorting"""
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
        
        # Sort image filenames naturally for consistent display
        sorted_image_filenames = sorted(list(all_image_filenames), key=natural_sort_key)
        
        # Generate table rows
        for image_filename in sorted_image_filenames:
            html_content += f'''
                        <tr>
                            <td class="image-id">{image_filename}</td>'''
            
            # For each category, find the matching image
            categories = [
                ('original_images', 'original_images'),
                ('pattern_detection', 'pattern_detection'),
                ('undistorted', 'undistorted'),
                ('reprojection', 'reprojection')
            ]
            
            for category_key, subdir_key in categories:
                found_image = None
                category_files = image_lists.get(category_key, [])
                
                # Look for matching file with various possible suffixes
                possible_suffixes = ['', '_pattern', '_undistorted', '_reprojection', '_debug']
                possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                
                for suffix in possible_suffixes:
                    for ext in possible_extensions:
                        candidate = f"{image_filename}{suffix}{ext}"
                        if candidate in category_files:
                            found_image = candidate
                            break
                    if found_image:
                        break
                
                if found_image:
                    img_path = f"{rel_subdirs[subdir_key]}/{found_image}"
                    html_content += f'''
                            <td>
                                <img src="{img_path}" alt="{found_image}" class="table-image" onclick="openModal(this.src)">
                            </td>'''
                else:
                    html_content += '''
                            <td>
                                <div class="image-unavailable">Detection Failed</div>
                            </td>'''
            
            html_content += '''
                        </tr>'''
        
        html_content += '''
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- Modal for image viewing -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }
        
        function closeModal() {
            const modal = document.getElementById('imageModal');
            modal.style.display = 'none';
        }
        
        // Close modal when clicking outside the image
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>'''
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
