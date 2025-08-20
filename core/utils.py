"""
Utility functions for camera calibration
========================================

This module contains general utility functions used across different calibration methods.
Extracted from the original duco_camera_calibrateOPT_cmd.py for modularity.
"""

import os
import numpy as np
from typing import Sequence, Tuple, List
from numpy.typing import ArrayLike
import cv2
from glob import glob


def get_objpoints(num_images: int, XX: int, YY: int, L: float) -> Sequence[ArrayLike]:
    """
    Generate 3D coordinates of chessboard corners for all images.
    
    Args:
        num_images: Number of calibration images
        XX: Number of corners along x-axis
        YY: Number of corners along y-axis  
        L: Square size in meters
        
    Returns:
        List of 3D object points for each image
    """
    objpoints = []
    for i in range(num_images):
        objp = np.zeros((XX * YY, 3), np.float32)
        objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
        objp *= L
        objpoints.append(objp)
    
    return objpoints


def find_chessboard_corners(gray: np.array, XX: int, YY: int, 
                          flags: int = None, 
                          criteria: Tuple[int, int, float] = None, 
                          winsize: Tuple[int, int] = None):
    """
    Find chessboard corners in a grayscale image.
    
    Args:
        gray: Grayscale image
        XX: Number of corners along x-axis
        YY: Number of corners along y-axis
        flags: Corner detection flags
        criteria: Corner refinement criteria
        winsize: Window size for corner refinement
        
    Returns:
        (success, corners) tuple
    """
    if flags is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    if winsize is None:
        winsize = (11, 11)

    ret, corners = cv2.findChessboardCorners(gray, (XX, YY), flags)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, winsize, (-1, -1), criteria)
        return True, corners2
    else:
        return False, []


def get_chessboard_corners(image_paths, XX, YY):
    """
    Detect corners in multiple images.
    
    Args:
        image_paths: List of image file paths
        XX: Number of corners along x-axis
        YY: Number of corners along y-axis
        
    Returns:
        List of detected corner points
    """
    imgpoints = []
    
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        find_corners_ret, corners = find_chessboard_corners(gray, XX, YY)

        if find_corners_ret:
            imgpoints.append(corners)
        else:
            print(f"Cannot find chessboard corners in {image_path}, skip.")

    return imgpoints


def load_images_from_directory(directory_path, extensions=None):
    """
    Load all images from a directory.
    
    Args:
        directory_path: Path to the directory containing images
        extensions: List of image extensions to look for
        
    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(directory_path, ext)))
        image_paths.extend(glob(os.path.join(directory_path, ext.upper())))
    
    # Remove duplicates (case-insensitive filesystems can cause duplicates)
    image_paths = list(dict.fromkeys(image_paths))
    
    # Sort by numeric value in filename
    image_paths.sort(key=lambda x: int(os.path.split(x)[-1].split('.')[0]) 
                    if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
    
    return image_paths


# Mathematical utility functions
def rpy_to_matrix(coords):
    """
    Calculate rotation matrix from roll-pitch-yaw angles (radians).
    
    Args:
        coords: Array of [roll, pitch, yaw] in radians
        
    Returns:
        3x3 rotation matrix
    """
    coords = np.asanyarray(coords, dtype=np.float64)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ], dtype=np.float64)


def xyz_rpy_to_matrix(xyz_rpy):
    """
    Calculate 4x4 transformation matrix from xyz positions and rpy angles.
    
    Args:
        xyz_rpy: Array of [x, y, z, roll, pitch, yaw]
        
    Returns:
        4x4 transformation matrix
    """
    R = rpy_to_matrix(xyz_rpy[3:])  # 3x3
    t = xyz_rpy[:3]  # 3x1
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R
    matrix[:3, 3] = t
    return matrix


def inverse_transform_matrix(T):
    """
    Calculate the inverse of a 4x4 transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Inverse transformation matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -np.dot(R_inv, t)

    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def matrix_to_xyz_rpy(matrix):
    """
    Calculate xyz positions and rpy angles from 4x4 transformation matrix.
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Tuple of (x, y, z, roll, pitch, yaw)
    """
    x, y, z = matrix[:3, 3]
    
    R = matrix[:3, :3]
    
    # Calculate Euler angles
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return x, y, z, roll, pitch, yaw
