"""
Intrinsic Camera Calibration Module
===================================

This module handles single camera intrinsic parameter calibration.
It can calibrate camera matrix and distortion coefficients from chessboard images.
Supports multiple chessboard pattern types through abstraction.
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
    
    This class provides methods to calibrate camera intrinsic parameters
    (camera matrix and distortion coefficients) from chessboard calibration images.
    """
    
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.image_size = None
        self.calibration_completed = False
        self.rvecs = None  # Rotation vectors for each image (extrinsics)
        self.tvecs = None  # Translation vectors for each image (extrinsics)
        self.valid_image_paths = None  # Paths of images with successful corner detection
    
    def calibrate_from_images(self, image_paths: List[str], XX: int, YY: int, L: float,
                            distortion_model: str = 'standard', verbose: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Calibrate camera intrinsic parameters from a list of chessboard images.
        
        Args:
            image_paths: List of calibration image file paths
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            distortion_model: Distortion model to use ('standard', 'rational', 'thin_prism', 'tilted')
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (success, camera_matrix, distortion_coefficients)
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        
        # Generate 3D object points
        objpoints = get_objpoints(len(image_paths), XX, YY, L)
        
        # Find chessboard corners in all images
        imgpoints = []
        valid_objpoints = []
        valid_image_paths = []
        
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.image_size is None:
                self.image_size = gray.shape[::-1]
            
            find_corners_ret, corners = find_chessboard_corners(gray, XX, YY)
            
            if find_corners_ret:
                imgpoints.append(corners)
                valid_objpoints.append(objpoints[i])
                valid_image_paths.append(image_path)
                
                if verbose:
                    print(f"Found corners in {image_path}")
            else:
                print(f"Cannot find chessboard corners in {image_path}, skip.")
        
        if len(imgpoints) < 3:
            raise ValueError(f"Need at least 3 images with detected corners, got {len(imgpoints)}")
        
        # Set calibration flags based on distortion model
        flags = 0
        if distortion_model == 'rational':
            flags = cv2.CALIB_RATIONAL_MODEL
        elif distortion_model == 'thin_prism':
            flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        elif distortion_model == 'tilted':
            flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
        # 'standard' uses flags = 0 (default)
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            valid_objpoints, imgpoints, self.image_size, None, None, flags=flags)
        
        if ret:
            self.camera_matrix = mtx
            self.distortion_coefficients = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.valid_image_paths = valid_image_paths
            self.calibration_completed = True
            
            if verbose:
                print(f"Calibration successful with {len(imgpoints)} images")
                print(f"Camera matrix:\n{mtx}")
                print(f"Distortion coefficients:\n{dist}")
        
        return ret, mtx, dist
    
    def calibrate_from_directory(self, directory_path: str, XX: int, YY: int, L: float,
                               selected_indices: Optional[List[int]] = None,
                               distortion_model: str = 'standard', verbose: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Calibrate camera from all images in a directory.
        
        Args:
            directory_path: Path to directory containing calibration images
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            selected_indices: Optional list of image indices to use (0-based)
            distortion_model: Distortion model to use ('standard', 'rational', 'thin_prism', 'tilted')
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (success, camera_matrix, distortion_coefficients)
        """
        image_paths = load_images_from_directory(directory_path)
        
        if selected_indices is not None:
            image_paths = [image_paths[i] for i in selected_indices if i < len(image_paths)]
        
        return self.calibrate_from_images(image_paths, XX, YY, L, distortion_model, verbose)
    
    def save_parameters(self, save_directory: str) -> None:
        """
        Save calibrated intrinsic parameters to JSON files.
        
        Args:
            save_directory: Directory to save the parameter files
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save camera matrix
        mtx_dict = {"camera_matrix": self.camera_matrix.tolist()}
        with open(os.path.join(save_directory, 'mtx.json'), 'w', encoding='utf-8') as f:
            json.dump(mtx_dict, f, indent=4, ensure_ascii=False)
        
        # Save distortion coefficients
        dist_dict = {"distortion_coefficients": self.distortion_coefficients.tolist()}
        with open(os.path.join(save_directory, 'dist.json'), 'w', encoding='utf-8') as f:
            json.dump(dist_dict, f, indent=4, ensure_ascii=False)
        
        # Save extrinsic parameters (pose of each image relative to chessboard)
        if self.rvecs is not None and self.tvecs is not None and self.valid_image_paths is not None:
            extrinsics_dict = {}
            for i, (rvec, tvec, image_path) in enumerate(zip(self.rvecs, self.tvecs, self.valid_image_paths)):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                # Convert rotation vector to rotation matrix for easier interpretation
                rmat, _ = cv2.Rodrigues(rvec)
                extrinsics_dict[image_name] = {
                    "rotation_vector": rvec.flatten().tolist(),
                    "translation_vector": tvec.flatten().tolist(),
                    "rotation_matrix": rmat.tolist(),
                    "image_path": image_path
                }
            
            with open(os.path.join(save_directory, 'extrinsics.json'), 'w', encoding='utf-8') as f:
                json.dump(extrinsics_dict, f, indent=4, ensure_ascii=False)
        
        print(f"Camera intrinsic parameters saved to: {save_directory}")
    
    def load_parameters(self, save_directory: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load calibrated intrinsic parameters from JSON files.
        
        Args:
            save_directory: Directory containing the parameter files
            
        Returns:
            Tuple of (camera_matrix, distortion_coefficients)
        """
        camera_matrix_path = os.path.join(save_directory, 'mtx.json')
        distortion_path = os.path.join(save_directory, 'dist.json')
        
        if not os.path.exists(camera_matrix_path):
            raise FileNotFoundError(f"Camera matrix file not found: {camera_matrix_path}")
        
        if not os.path.exists(distortion_path):
            raise FileNotFoundError(f"Distortion coefficients file not found: {distortion_path}")
        
        # Load camera matrix
        with open(camera_matrix_path, 'r', encoding='utf-8') as f:
            mtx_dict = json.load(f)
            camera_matrix = np.array(mtx_dict["camera_matrix"])
        
        # Load distortion coefficients
        with open(distortion_path, 'r', encoding='utf-8') as f:
            dist_dict = json.load(f)
            distortion_coefficients = np.array(dist_dict["distortion_coefficients"])
        
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.calibration_completed = True
        
        return camera_matrix, distortion_coefficients
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the calibrated camera parameters.
        
        Returns:
            Tuple of (camera_matrix, distortion_coefficients)
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        return self.camera_matrix, self.distortion_coefficients
    
    def undistort_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Undistort an image using the calibrated parameters.
        
        Args:
            image_path: Path to the input image
            output_path: Optional path to save the undistorted image
            
        Returns:
            Undistorted image
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        undistorted = cv2.undistort(img, self.camera_matrix, self.distortion_coefficients, None, None)
        
        if output_path:
            cv2.imwrite(output_path, undistorted)
            
        return undistorted
    
    def calibrate_with_pattern(self, image_paths: List[str], 
                              calibration_pattern: CalibrationPattern,
                              distortion_model: str = 'standard', 
                              verbose: bool = False) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Calibrate camera using a specific calibration pattern abstraction.
        
        Args:
            image_paths: List of calibration image file paths
            calibration_pattern: CalibrationPattern instance defining the pattern
            distortion_model: Distortion model to use ('standard', 'rational', 'thin_prism', 'tilted')
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (success, camera_matrix, distortion_coefficients)
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        
        # Collect corner detections and object points
        imgpoints = []
        objpoints = []
        valid_image_paths = []
        
        # Generate base object points for this pattern
        if calibration_pattern.is_planar:
            # For planar patterns, generate once
            pattern_objpoints = calibration_pattern.generate_object_points()
        
        if verbose:
            print(f"Using pattern: {calibration_pattern.name}")
            print(f"Pattern info: {calibration_pattern.get_info()}")
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                if verbose:
                    print(f"Could not load image: {image_path}")
                continue
            
            if self.image_size is None:
                if len(img.shape) == 3:
                    self.image_size = (img.shape[1], img.shape[0])
                else:
                    self.image_size = (img.shape[1], img.shape[0])
            
            # Detect corners/features using the pattern
            success, corners, point_ids = calibration_pattern.detect_corners(img)
            
            if success and corners is not None:
                imgpoints.append(corners)
                
                # Generate corresponding object points
                if calibration_pattern.is_planar:
                    # For planar patterns, use the same object points for all images
                    if point_ids is not None:
                        # For patterns with non-sequential detection (e.g., ChArUco)
                        objp = calibration_pattern.generate_object_points(point_ids)
                    else:
                        # For patterns with sequential detection (e.g., standard chessboard)
                        objp = pattern_objpoints
                else:
                    # For 3D patterns, generate object points based on detected features
                    objp = calibration_pattern.generate_object_points(point_ids)
                
                objpoints.append(objp)
                valid_image_paths.append(image_path)
                
                if verbose:
                    print(f"Found corners in {image_path}")
            else:
                if verbose:
                    print(f"Cannot find pattern corners in {image_path}, skip.")
        
        if len(imgpoints) < 3:
            raise ValueError(f"Need at least 3 images with detected corners, got {len(imgpoints)}")
        
        # Set calibration flags based on distortion model
        flags = 0
        if distortion_model == 'rational':
            flags = cv2.CALIB_RATIONAL_MODEL
        elif distortion_model == 'thin_prism':
            flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        elif distortion_model == 'tilted':
            flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None, flags=flags)
        
        if ret:
            self.camera_matrix = mtx
            self.distortion_coefficients = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.valid_image_paths = valid_image_paths
            self.calibration_completed = True
            
            if verbose:
                print(f"Calibration successful with {len(imgpoints)} images")
                print(f"Camera matrix:\n{mtx}")
                print(f"Distortion coefficients:\n{dist}")
        
        return ret, mtx, dist
