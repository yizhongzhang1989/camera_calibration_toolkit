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


def calculate_reproject_error_fast(imgpoints, objpoints, 
                                   rvecs_target2cam, tvecs_target2cam, mtx, dist, 
                                   image_paths=None, XX=None, YY=None, vis=False, 
                                   save_fig=False, save_dir='./output',
                                   verbose=False):
    """
    Calculate the average reprojection error for all images.
    
    Args:
        imgpoints: List of detected corner points in images
        objpoints: List of 3D object points
        rvecs_target2cam: Rotation vectors from target to camera
        tvecs_target2cam: Translation vectors from target to camera
        mtx: Camera intrinsic matrix
        dist: Distortion coefficients
        image_paths: List of image file paths (optional)
        XX: Chessboard corners in x direction (for visualization)
        YY: Chessboard corners in y direction (for visualization)
        vis: Whether to show visualization
        save_fig: Whether to save visualization images
        save_dir: Directory to save visualization images
        verbose: Whether to print detailed information
        
    Returns:
        Average reprojection error
    """
    num_images = len(imgpoints)
    mean_error = 0

    for image_idx in range(num_images):
        imgpoints2, _ = cv2.projectPoints(objpoints[image_idx], 
                                        rvecs_target2cam[image_idx], 
                                        tvecs_target2cam[image_idx], 
                                        mtx, dist)
        error = cv2.norm(imgpoints[image_idx], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

        if verbose:
            print(f"{image_idx} error: {error}")

        mean_error += error

        # Visualization or saving
        if (vis or save_fig) and image_paths is not None and XX is not None and YY is not None:
            imagei = cv2.imread(image_paths[image_idx])            
            img_draw = cv2.drawChessboardCorners(imagei, (XX, YY), imgpoints2, True)

            if save_fig:
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(f'{save_dir}/{image_idx}_reproject_corners.png', img_draw)

            if vis:
                cv2.imshow('img_draw', img_draw)
                cv2.waitKey(0)

    if verbose:
        print(f"Total error: {mean_error/len(objpoints)}")

    return mean_error/len(objpoints)


def calculate_single_image_reprojection_error(image_path, rvec_target2cam, tvec_target2cam, 
                                            mtx, dist, XX, YY, L, verbose=False):
    """
    Calculate the reprojection error for a single image.
    
    Args:
        image_path: Path to the image
        rvec_target2cam: Rotation vector from target to camera
        tvec_target2cam: Translation vector from target to camera
        mtx: Camera intrinsic matrix
        dist: Distortion coefficients
        XX: Number of corners along x-axis
        YY: Number of corners along y-axis
        L: Square size in meters
        verbose: Whether to print detailed information
        
    Returns:
        Reprojection error for the image
    """
    imgpoints = get_chessboard_corners([image_path], XX, YY)
    
    if len(imgpoints) == 0:
        print(f"Could not find chessboard corners in {image_path}")
        return float('inf')
    
    objpoints = get_objpoints(1, XX, YY, L)
    
    reprojected_imgpoints, _ = cv2.projectPoints(objpoints[0], rvec_target2cam, 
                                               tvec_target2cam, mtx, dist)
    
    error = cv2.norm(imgpoints[0], reprojected_imgpoints, cv2.NORM_L2) / len(reprojected_imgpoints)

    if verbose:
        print(f"{image_path} reprojection error: {error}")

    return error


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


def count_calibration_images(directory_path):
    """
    Count the number of images in the calibration data directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Number of images found
    """
    image_paths = load_images_from_directory(directory_path)
    return len(image_paths)
