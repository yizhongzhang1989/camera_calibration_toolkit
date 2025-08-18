"""
Visualization Utilities for Camera Calibration
==============================================

Shared functions for generating calibration visualizations 
including corner detection and undistorted images.
"""

import os
import cv2
import numpy as np
from flask import url_for
from core.utils import find_chessboard_corners


def draw_coordinate_axes(img, camera_matrix, dist_coeffs, rvec, tvec, XX, YY, L):
    """
    Draw coordinate axes on the image to visualize the chessboard pose.
    
    Args:
        img: Input image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector (chessboard to camera)
        tvec: Translation vector (chessboard to camera)
        XX: Number of chessboard corners along x-axis
        YY: Number of chessboard corners along y-axis
        L: Size of chessboard squares
        
    Returns:
        Image with coordinate axes drawn
    """
    # Create 3D points for coordinate axes
    # Origin at (0, 0, 0), with axes lengths matching chessboard dimensions
    x_length = (XX - 1) * L  # X-axis length matches chessboard width
    y_length = (YY - 1) * L  # Y-axis length matches chessboard height
    z_length = L             # Z-axis length equals one square size
    
    axes_points = np.array([
        [0, 0, 0],           # Origin
        [x_length, 0, 0],    # X-axis end (red)
        [0, y_length, 0],    # Y-axis end (green)  
        [0, 0, z_length]     # Z-axis end (blue)
    ], dtype=np.float32)
    
    # Project 3D points to image plane
    projected_points, _ = cv2.projectPoints(
        axes_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    # Convert to integer pixel coordinates
    projected_points = projected_points.reshape(-1, 2).astype(int)
    
    img_with_axes = img.copy()
    
    # Draw coordinate axes with different colors
    # X-axis in red
    cv2.arrowedLine(img_with_axes, 
                   tuple(projected_points[0]), 
                   tuple(projected_points[1]),
                   (0, 0, 255), 5, tipLength=0.1)
    
    # Y-axis in green  
    cv2.arrowedLine(img_with_axes,
                   tuple(projected_points[0]),
                   tuple(projected_points[2]),
                   (0, 255, 0), 5, tipLength=0.1)
    
    # Z-axis in blue
    cv2.arrowedLine(img_with_axes,
                   tuple(projected_points[0]),
                   tuple(projected_points[3]),
                   (255, 0, 0), 5, tipLength=0.1)
    
    # Add text labels near the axes endpoints
    cv2.putText(img_with_axes, 'X', tuple(projected_points[1] + [10, -10]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_with_axes, 'Y', tuple(projected_points[2] + [10, -10]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_with_axes, 'Z', tuple(projected_points[3] + [10, -10]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return img_with_axes


def trim_distortion_coefficients(dist_coeffs, distortion_model='standard'):
    """
    Trim distortion coefficients based on the distortion model to remove trailing zeros.
    
    Args:
        dist_coeffs: Distortion coefficients array
        distortion_model: Type of distortion model used
        
    Returns:
        Trimmed distortion coefficients array
    """
    # Flatten in case it's 2D
    flat_coeffs = dist_coeffs.flatten() if hasattr(dist_coeffs, 'flatten') else np.array(dist_coeffs).flatten()
    
    # Expected coefficient counts for each model
    expected_counts = {
        'standard': 5,    # k1, k2, p1, p2, k3
        'rational': 8,    # k1-k6, p1, p2  
        'thin_prism': 12, # k1-k6, p1, p2, s1-s4
        'tilted': 14      # k1-k6, p1, p2, s1-s4, τx, τy
    }
    
    expected_count = expected_counts.get(distortion_model, 5)
    
    # For rational model, specifically check if coefficients beyond index 7 are all zeros
    if distortion_model == 'rational' and len(flat_coeffs) > 8:
        has_non_zero_after_8 = np.any(np.abs(flat_coeffs[8:]) > 1e-10)
        if not has_non_zero_after_8:
            return flat_coeffs[:8]
    
    # General trailing zero removal for all models
    coeffs_list = flat_coeffs.tolist()
    while len(coeffs_list) > expected_count and abs(coeffs_list[-1]) < 1e-10:
        coeffs_list.pop()
    
    return np.array(coeffs_list)


def generate_calibration_visualizations(session_id, image_paths, selected_indices, 
                                       camera_matrix, dist_coeffs, XX, YY, 
                                       results_folder, calibration_type='intrinsic',
                                       L=1.0, rvecs=None, tvecs=None, pattern_type=None, pattern=None):
    """
    Generate corner detection and undistorted images for calibration results.
    
    Args:
        session_id: Session identifier for URL generation
        image_paths: List of image file paths
        selected_indices: List of indices corresponding to selected images  
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients array
        XX: Number of chessboard corners along x-axis
        YY: Number of chessboard corners along y-axis
        results_folder: Base folder for saving results
        calibration_type: Type of calibration ('intrinsic' or 'eye_in_hand')
        rvecs: Rotation vectors for each image (optional)
        tvecs: Translation vectors for each image (optional)
        
    Returns:
        Tuple of (corner_images, undistorted_images) lists
    """
    corner_images = []
    undistorted_images = []
    
    # Create visualization directories
    corner_viz_dir = os.path.join(results_folder, 'corner_visualizations')
    if calibration_type == 'intrinsic':
        undistorted_dir = os.path.join(results_folder, 'undistorted')
    else:
        undistorted_dir = os.path.join(results_folder, 'undistorted_images')
    
    os.makedirs(corner_viz_dir, exist_ok=True)
    os.makedirs(undistorted_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        original_index = selected_indices[i] if i < len(selected_indices) else i
        
        # Extract original filename (without extension) from the image path
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Corner detection visualization - handle different pattern types
        find_corners_ret = False
        corners = None
        
        if pattern_type == 'charuco' and pattern is not None:
            # ChArUco corner detection
            try:
                # Use the pattern object's detection method
                find_corners_ret, corners = pattern.detect_corners(gray)
                
                if find_corners_ret and corners is not None and len(corners) > 0:
                    # Draw corners on the image
                    img_with_corners = img.copy()
                    
                    # For ChArUco, corners are already in the right format
                    # Draw each corner point
                    for corner in corners:
                        center = tuple(corner[0].astype(int))
                        cv2.circle(img_with_corners, center, 8, (0, 255, 0), 2)
                        
                    corner_filename = f"{original_filename}_corners.jpg"
                    corner_path = os.path.join(corner_viz_dir, corner_filename)
                    cv2.imwrite(corner_path, img_with_corners)
                    
                    corner_images.append({
                        'name': corner_filename,
                        'path': corner_path,
                        'url': url_for('get_corner_image', session_id=session_id, filename=corner_filename),
                        'index': original_index,
                        'original_name': original_filename
                    })
                    
            except Exception as e:
                print(f"ChArUco corner detection failed for {original_filename}: {e}")
        else:
            # Standard chessboard corner detection
            find_corners_ret, corners = find_chessboard_corners(gray, XX, YY)
            
            if find_corners_ret:
                # Draw corners on the image
                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, (XX, YY), corners, find_corners_ret)
                
                corner_filename = f"{original_filename}_corners.jpg"
                corner_path = os.path.join(corner_viz_dir, corner_filename)
                cv2.imwrite(corner_path, img_with_corners)
                
                corner_images.append({
                    'name': corner_filename,
                    'path': corner_path,
                    'url': url_for('get_corner_image', session_id=session_id, filename=corner_filename),
                    'index': original_index,
                    'original_name': original_filename
                })
        
        # Generate undistorted image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        
        # Draw coordinate axes on undistorted image if rotation and translation vectors are provided
        if rvecs is not None and tvecs is not None and original_index < len(rvecs):
            undistorted_img = draw_coordinate_axes(
                undistorted_img, 
                camera_matrix, 
                np.zeros(5),  # No distortion for undistorted image
                rvecs[original_index], 
                tvecs[original_index], 
                XX, 
                YY, 
                L
            )
            
        undistorted_filename = f"{original_filename}_undistorted.jpg"
        undistorted_path = os.path.join(undistorted_dir, undistorted_filename)
        cv2.imwrite(undistorted_path, undistorted_img)
        
        undistorted_images.append({
            'name': undistorted_filename,
            'path': undistorted_path,
            'url': url_for('get_undistorted_image', session_id=session_id, filename=undistorted_filename),
            'index': original_index,
            'original_name': original_filename
        })
    
    return corner_images, undistorted_images


def generate_reprojection_visualizations(session_id, image_paths, selected_indices, results_folder):
    """
    Collect reprojection visualization images generated by the eye-in-hand calibrator.
    
    Args:
        session_id: Session identifier for URL generation
        image_paths: List of image file paths
        selected_indices: List of indices corresponding to selected images
        results_folder: Base folder containing results
        
    Returns:
        List of reprojected image dictionaries
    """
    reprojected_images = []
    reprojection_dir = os.path.join(results_folder, 'visualizations')
    
    if not os.path.exists(reprojection_dir):
        return reprojected_images
    
    for i, image_path in enumerate(image_paths):
        original_index = selected_indices[i] if i < len(selected_indices) else i
        
        # Extract original filename (without extension) from the image path
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Look for reprojected visualization files
        # Backend generates files with pattern: optimized_reproject_originalname.png
        possible_filenames = [
            f"optimized_reproject_{original_filename}.png",  # New pattern from calibrator
            f"{i}_optimized.jpg",  # Fallback to old index-based naming
            f"{original_filename}_optimized.jpg"  # Alternative filename-based naming
        ]
        
        for reprojected_filename in possible_filenames:
            reprojected_path = os.path.join(reprojection_dir, reprojected_filename)
            if os.path.exists(reprojected_path):
                reprojected_images.append({
                    'name': reprojected_filename,
                    'path': reprojected_path,
                    'url': url_for('get_visualization_image', session_id=session_id, filename=reprojected_filename),
                    'index': original_index,
                    'original_name': original_filename
                })
                break  # Use the first match found
    
    return reprojected_images


def create_calibration_results(calibration_type, session_id, image_paths, selected_indices,
                              camera_matrix, dist_coeffs, results_folder, XX, YY, L=1.0, **kwargs):
    """
    Create standardized calibration results with visualizations.
    
    Args:
        calibration_type: 'intrinsic' or 'eye_in_hand'
        session_id: Session identifier
        image_paths: List of image file paths
        selected_indices: List of selected image indices
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        results_folder: Results storage folder
        XX, YY: Chessboard dimensions
        **kwargs: Additional calibration-specific results
        
    Returns:
        Dictionary containing standardized results
    """
    # Generate common visualizations
    corner_images, undistorted_images = generate_calibration_visualizations(
        session_id, image_paths, selected_indices, camera_matrix, dist_coeffs, 
        XX, YY, results_folder, calibration_type, L,
        rvecs=kwargs.get('rvecs'), tvecs=kwargs.get('tvecs'),
        pattern_type=kwargs.get('pattern_type'), pattern=kwargs.get('pattern')
    )
    
    # Base result structure
    results = {
        'success': True,
        'calibration_type': calibration_type,
        'images_used': len(image_paths),
        'corner_images': corner_images,
        'undistorted_images': undistorted_images,
    }
    
    # Add calibration-specific results
    if calibration_type == 'intrinsic':
        # Trim distortion coefficients based on the model
        distortion_model = kwargs.get('distortion_model', 'standard')
        trimmed_dist_coeffs = trim_distortion_coefficients(dist_coeffs, distortion_model)
        
        results.update({
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': trimmed_dist_coeffs.tolist(),
            'distortion_model': distortion_model,
            'rms_error': kwargs.get('rms_error', 0.0),
            'message': kwargs.get('message', f'Intrinsic calibration completed successfully using {len(image_paths)} images')
        })
        
    elif calibration_type == 'eye_in_hand':
        # Generate reprojection visualizations for eye-in-hand
        reprojected_images = generate_reprojection_visualizations(
            session_id, image_paths, selected_indices, results_folder
        )
        
        results.update({
            'handeye_transform': kwargs.get('handeye_transform', []).tolist() if hasattr(kwargs.get('handeye_transform', []), 'tolist') else kwargs.get('handeye_transform', []),
            'cam2end_matrix': kwargs.get('cam2end_matrix', []).tolist() if hasattr(kwargs.get('cam2end_matrix', []), 'tolist') else kwargs.get('cam2end_matrix', []),
            'reprojection_error': kwargs.get('reprojection_error', 0.0),
            'reprojected_images': reprojected_images,
            'initial_reprojection_errors': kwargs.get('initial_reprojection_errors', []),
            'optimized_reprojection_errors': kwargs.get('optimized_reprojection_errors', []),
            'initial_mean_error': kwargs.get('initial_mean_error', 0.0),
            'optimized_mean_error': kwargs.get('optimized_mean_error', 0.0),
            'improvement_percentage': kwargs.get('improvement_percentage', 0.0),
            'message': kwargs.get('message', 'Eye-in-hand calibration completed successfully')
        })
    
    return results
