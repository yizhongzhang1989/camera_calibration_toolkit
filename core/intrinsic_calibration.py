"""
Intrinsic Camera Calibration Module
===================================

This module handles single camera intrinsic parameter calibration.
It can calibrate camera matrix and distortion coefficients from calibration images.
Supports multiple calibration pattern types through abstraction.

Key Methods in IntrinsicCalibrator class:
- Smart constructor: Initialize with images and patterns directly
- calibrate(): Perform intrinsic calibration with organized member variables
- detect_pattern_points(): Pattern detection with automatic point collection (inherited from BaseCalibrator)

The new interface provides clean separation of data and processing:
    # Smart constructor approach
    calibrator = IntrinsicCalibrator(
        image_paths=paths,
        calibration_pattern=pattern
    )
    result = calibrator.calibrate()
    if result:
        camera_matrix = result['camera_matrix']
        rms_error = result['rms_error']
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
from .base_calibrator import BaseCalibrator
from .calibration_patterns import CalibrationPattern


class IntrinsicCalibrator(BaseCalibrator):
    """
    Camera intrinsic parameter calibration class.
    
    This class inherits common functionality from BaseCalibrator and specializes in
    single camera intrinsic parameter calibration following OpenCV's calibrateCamera interface.
    
    Specialized attributes:
    - camera_matrix: Calibrated camera intrinsic matrix
    - distortion_coefficients: Calibrated distortion coefficients
    - distortion_model: Distortion model used for calibration
    """
    
    def __init__(self, images=None, image_paths=None, calibration_pattern=None):
        """
        Initialize IntrinsicCalibrator with smart constructor arguments.
        
        Args:
            images: List of image arrays (numpy arrays) or None
            image_paths: List of image file paths or None
            calibration_pattern: CalibrationPattern instance or None
        """
        # Initialize base class with common functionality
        super().__init__(images, image_paths, calibration_pattern)
            
    # Abstract method implementations
    def calibrate(self, 
                 cameraMatrix: Optional[np.ndarray] = None,
                 distCoeffs: Optional[np.ndarray] = None, 
                 flags: int = 0,
                 criteria: Optional[Tuple] = None,
                 verbose: bool = False) -> Optional[dict]:
        """
        Perform intrinsic camera calibration.
        
        This method uses the point correspondences stored in class members
        (image_points, object_points) to calibrate the camera.
        
        Args:
            cameraMatrix: Initial camera matrix (3x3). If None, will be estimated.
            distCoeffs: Initial distortion coefficients. If None, will be estimated.
            flags: Calibration flags (same as OpenCV's calibrateCamera)
            criteria: Termination criteria for iterative algorithms
            verbose: Whether to print detailed progress
            
        Returns:
            Optional[dict]: Dictionary containing calibration results if successful, None if failed.
            Result dictionary contains:
            - 'camera_matrix': np.ndarray - Calibrated camera intrinsic matrix
            - 'distortion_coefficients': np.ndarray - Distortion coefficients
            - 'rms_error': float - Overall RMS reprojection error
            
        Note:
            Before calling this method, you must:
            1. Set images: set_images_from_paths() or set_images_from_arrays()
            2. Set pattern: set_calibration_pattern()  
            3. Detect points: detect_pattern_points() (or let this method do it automatically)
            
            After successful calibration, class member variables are also available:
            - self.camera_matrix, self.distortion_coefficients
            - self.rms_error, self.per_image_errors
            
        Distortion Model Flags:
            Use cv2.CALIB_* flags to control distortion models:
            - Standard (5 coeff): no additional flags 
            - Rational (8 coeff): cv2.CALIB_RATIONAL_MODEL
            - Thin Prism (12 coeff): cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
            - Tilted (14 coeff): cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
        """
        if self.image_points is None or self.object_points is None:
            if verbose:
                print("Image and object points not calculated yet. Running pattern detection...")
            
            # Automatically detect pattern points if not done yet
            if not self.detect_pattern_points(verbose=verbose):
                raise ValueError("Pattern detection failed. Cannot proceed with calibration.")
        
        # Count successful detections (non-None entries)
        successful_count = sum(1 for pts in self.image_points if pts is not None)
        
        if successful_count < 3:
            raise ValueError(f"Insufficient point correspondences: need at least 3, got {successful_count}")
        
        if self.image_size is None:
            raise ValueError("Image size not set")
        
        # Create temporary arrays for OpenCV (only successful detections)
        temp_object_points = [pts for pts in self.object_points if pts is not None]
        temp_image_points = [pts for pts in self.image_points if pts is not None]
        
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
            print(f"Running calibration with {successful_count} point correspondences")
            print(f"Image size: {self.image_size}")
            print(f"Calibration flags: {calibration_flags}")
            if initial_camera_matrix is not None:
                print(f"Using initial camera matrix:")
                print(initial_camera_matrix)
        
        # Perform calibration using temporary arrays
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                temp_object_points, 
                temp_image_points, 
                self.image_size, 
                initial_camera_matrix, 
                distCoeffs,
                flags=calibration_flags,
                criteria=criteria
            )
            
            if ret and mtx is not None:
                # Restore alignment: expand rvecs/tvecs back to match original image count
                self.rvecs = [None] * len(self.images)
                self.tvecs = [None] * len(self.images)
                
                # Map successful results back to original image indices
                success_idx = 0
                for i in range(len(self.images)):
                    if self.image_points[i] is not None:
                        self.rvecs[i] = rvecs[success_idx]
                        self.tvecs[i] = tvecs[success_idx]
                        success_idx += 1
                    # else: rvecs[i] and tvecs[i] remain None for failed detections
                
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
                self.distortion_coefficients = dist.flatten()  # Flatten to 1D array
                self.distortion_model = distortion_model
                self.rms_error = ret
                self.calibration_completed = True
                
                # Calculate per-image reprojection errors (only for successful detections)
                self.per_image_errors = [None] * len(self.images)  # Aligned with images
                temp_errors = []
                for temp_obj_pts, temp_img_pts, rvec, tvec in zip(temp_object_points, temp_image_points, rvecs, tvecs):
                    projected_pts, _ = cv2.projectPoints(temp_obj_pts, rvec, tvec, mtx, dist)
                    error = cv2.norm(temp_img_pts, projected_pts, cv2.NORM_L2) / np.sqrt(len(projected_pts))
                    temp_errors.append(error)
                
                # Map errors back to original image indices
                success_idx = 0
                for i in range(len(self.images)):
                    if self.image_points[i] is not None:
                        self.per_image_errors[i] = temp_errors[success_idx]
                        success_idx += 1
                    # else: per_image_errors[i] remains None for failed detections
                
                if verbose:
                    print(f"✅ Calibration successful!")
                    print(f"RMS reprojection error: {ret:.4f} pixels")
                    print(f"Camera matrix:")
                    print(f"  fx: {mtx[0,0]:.2f}, fy: {mtx[1,1]:.2f}")
                    print(f"  cx: {mtx[0,2]:.2f}, cy: {mtx[1,2]:.2f}")
                    print(f"Distortion coefficients: {dist.flatten()}")
                    # Only show errors for successful detections
                    successful_errors = [f'{err:.4f}' for err in self.per_image_errors if err is not None]
                    print(f"Per-image errors: {successful_errors}")
                
                # Return simplified result dictionary
                return {
                    'camera_matrix': mtx.copy(),
                    'distortion_coefficients': dist.flatten(),  # Flatten to 1D array
                    'rms_error': float(ret)
                }
            else:
                if verbose:
                    print("❌ Calibration failed - OpenCV returned invalid results")
                return None
                
        except Exception as e:
            if verbose:
                print(f"❌ Calibration failed with exception: {e}")
            return None

    def save_results(self, save_directory: str) -> None:
        """
        Save calibration results (wrapper for save_calibration).
        
        Args:
            save_directory: Directory to save results
        """
        if not self.is_calibrated():
            raise ValueError("No calibration results to save. Run calibration first.")
        
        import os
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, "intrinsic_calibration_results.json")
        self.save_calibration(filepath)

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
                "image_count": len([pts for pts in self.image_points if pts is not None]) if self.image_points else 0,
                "pattern_type": self.calibration_pattern.pattern_id if self.calibration_pattern else "unknown",
                "distortion_model": self.distortion_model,
                "rms_error": float(self.rms_error)
            },
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "image_size": list(self.image_size) if self.image_size else None,
            "per_image_errors": [float(err) if err is not None else None for err in self.per_image_errors] if self.per_image_errors else None
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
            # Only include extrinsics for successful detections (non-None entries)
            valid_rvecs = [rvec.tolist() for rvec in self.rvecs if rvec is not None]
            valid_tvecs = [tvec.tolist() for tvec in self.tvecs if tvec is not None]
            
            calibration_data["extrinsics"] = {
                "rotation_vectors": valid_rvecs,
                "translation_vectors": valid_tvecs
            }
            
            # Add image paths if available
            if self.image_paths:
                calibration_data["image_paths"] = self.image_paths
        
        # Save to file
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Calibration data saved to: {filepath}")

    def get_rms_error(self) -> Optional[float]:
        """Get overall RMS reprojection error (lower is better)."""
        return self.rms_error
    
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
    
    def get_per_image_errors(self) -> Optional[List[float]]:
        """Get per-image reprojection errors (None entries for failed detections)."""
        return self.per_image_errors
        
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
            
