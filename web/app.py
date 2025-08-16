"""
Flask Web Application for Camera Calibration
============================================

This module provides a web interface for camera calibration operations.
It includes image upload, parameter configuration, calibration execution,
and result visualization.
"""

import os
import json
import shutil
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from datetime import datetime
import zipfile
import tempfile
import base64

# Import core calibration modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
ALLOWED_POSE_EXTENSIONS = {'json'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global calibration objects
intrinsic_calibrator = IntrinsicCalibrator()
eye_in_hand_calibrator = EyeInHandCalibrator()

# Session data storage (in production, use proper session management)
session_data = {}


def allowed_file(filename, extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def get_session_folder(session_id):
    """Get or create session-specific folder."""
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    return session_folder


def image_to_base64(image_path):
    """Convert image to base64 for web display."""
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        # Determine image format
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            format_str = 'jpeg'
        elif ext == '.png':
            format_str = 'png'
        else:
            format_str = 'jpeg'  # default
        
        return f"data:image/{format_str};base64,{img_base64}"
    return None


@app.route('/')
def index():
    """Main calibration type selection page."""
    return render_template('index.html')


@app.route('/intrinsic')
def intrinsic_calibration():
    """Intrinsic calibration interface."""
    return render_template('intrinsic.html')


@app.route('/eye_in_hand')
def eye_in_hand_calibration():
    """Eye-in-hand calibration interface."""
    return render_template('eye_in_hand.html')


@app.route('/api/upload_images', methods=['POST'])
def upload_images():
    """Upload calibration images."""
    try:
        session_id = request.form.get('session_id', 'default')
        calibration_type = request.form.get('calibration_type', 'eye_in_hand')
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        session_folder = get_session_folder(session_id)
        images_folder = os.path.join(session_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(images_folder, filename)
                file.save(file_path)
                uploaded_files.append({
                    'name': filename,
                    'path': file_path,
                    'url': url_for('get_image', session_id=session_id, filename=filename)
                })
        
        # Update session data
        if session_id not in session_data:
            session_data[session_id] = {}
        
        # Append new images to existing ones (if any)
        existing_images = session_data[session_id].get('images', [])
        all_images = existing_images + uploaded_files
        
        session_data[session_id]['images'] = all_images
        session_data[session_id]['calibration_type'] = calibration_type
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} new images. Total: {len(all_images)} images',
            'files': all_images
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_poses', methods=['POST'])
def upload_poses():
    """Upload robot pose files for eye-in-hand calibration."""
    try:
        session_id = request.form.get('session_id', 'default')
        
        if 'files' not in request.files:
            return jsonify({'error': 'No pose files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No pose files selected'}), 400
        
        session_folder = get_session_folder(session_id)
        poses_folder = os.path.join(session_folder, 'poses')
        os.makedirs(poses_folder, exist_ok=True)
        
        uploaded_poses = []
        for file in files:
            if file and allowed_file(file.filename, ALLOWED_POSE_EXTENSIONS):
                filename = secure_filename(file.filename)
                file_path = os.path.join(poses_folder, filename)
                file.save(file_path)
                
                # Validate JSON format
                try:
                    with open(file_path, 'r') as f:
                        pose_data = json.load(f)
                    uploaded_poses.append({
                        'name': filename,
                        'path': file_path
                    })
                except json.JSONDecodeError:
                    os.remove(file_path)  # Remove invalid file
                    return jsonify({'error': f'Invalid JSON format in {filename}'}), 400
        
        # Update session data
        if session_id not in session_data:
            session_data[session_id] = {}
        session_data[session_id]['poses'] = uploaded_poses
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_poses)} pose files',
            'files': uploaded_poses
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set_parameters', methods=['POST'])
def set_parameters():
    """Set calibration parameters."""
    try:
        session_id = request.json.get('session_id', 'default')
        parameters = request.json.get('parameters', {})
        
        # Validate parameters
        required_params = ['chessboard_x', 'chessboard_y', 'square_size']
        for param in required_params:
            if param not in parameters:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Update session data
        if session_id not in session_data:
            session_data[session_id] = {}
        session_data[session_id]['parameters'] = parameters
        
        return jsonify({'message': 'Parameters set successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Run calibration process."""
    try:
        session_id = request.json.get('session_id', 'default')
        
        if session_id not in session_data:
            return jsonify({'error': 'No session data found'}), 400
        
        session_info = session_data[session_id]
        
        # Check required data
        if 'images' not in session_info or 'parameters' not in session_info:
            return jsonify({'error': 'Missing images or parameters'}), 400
        
        images = session_info['images']
        parameters = session_info['parameters']
        calibration_type = session_info.get('calibration_type', 'intrinsic')
        selected_indices = request.json.get('selected_indices', list(range(len(images))))
        
        # Extract parameters
        XX = int(parameters['chessboard_x'])
        YY = int(parameters['chessboard_y'])
        L = float(parameters['square_size'])
        distortion_model = parameters.get('distortion_model', 'standard')
        
        # Get image paths - only selected ones
        image_paths = [images[i]['path'] for i in selected_indices if i < len(images)]
        
        if len(image_paths) < 3:
            return jsonify({'error': 'Need at least 3 selected images for calibration'}), 400
        
        results = {}
        
        if calibration_type == 'intrinsic':
            # Intrinsic calibration only
            ret, camera_matrix, dist_coeffs = intrinsic_calibrator.calibrate_from_images(
                image_paths, XX, YY, L, distortion_model, verbose=True)
            
            if ret:
                # Save results
                results_folder = os.path.join(RESULTS_FOLDER, session_id)
                intrinsic_calibrator.save_parameters(results_folder)
                
                # Generate corner detection and undistorted images
                corner_images = []
                undistorted_images = []
                
                # Create visualization directories
                corner_viz_dir = os.path.join(results_folder, 'corner_detection')
                undistorted_dir = os.path.join(results_folder, 'undistorted')
                os.makedirs(corner_viz_dir, exist_ok=True)
                os.makedirs(undistorted_dir, exist_ok=True)
                
                for i, img_path in enumerate(image_paths):
                    original_index = selected_indices[i]
                    
                    # Generate corner detection visualization
                    img = cv2.imread(img_path)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Use the same function as in the intrinsic calibrator
                        from core.utils import find_chessboard_corners
                        find_corners_ret, corners = find_chessboard_corners(gray, XX, YY)
                        
                        if find_corners_ret:
                            # Draw corners on the image
                            img_with_corners = img.copy()
                            cv2.drawChessboardCorners(img_with_corners, (XX, YY), corners, find_corners_ret)
                            
                            corner_filename = f"{original_index}.jpg"
                            corner_path = os.path.join(corner_viz_dir, corner_filename)
                            cv2.imwrite(corner_path, img_with_corners)
                            
                            corner_images.append({
                                'name': corner_filename,
                                'path': corner_path,
                                'url': url_for('get_corner_image', session_id=session_id, filename=corner_filename),
                                'index': original_index
                            })
                        
                        # Generate undistorted image
                        undistorted_img = intrinsic_calibrator.undistort_image(img_path)
                        undistorted_filename = f"{original_index}.jpg"
                        undistorted_path = os.path.join(undistorted_dir, undistorted_filename)
                        cv2.imwrite(undistorted_path, undistorted_img)
                        
                        undistorted_images.append({
                            'name': undistorted_filename,
                            'path': undistorted_path,
                            'url': url_for('get_undistorted_image', session_id=session_id, filename=undistorted_filename),
                            'index': original_index
                        })
                
                results = {
                    'success': True,
                    'calibration_type': 'intrinsic',
                    'camera_matrix': camera_matrix.tolist(),
                    'distortion_coefficients': dist_coeffs.tolist(),
                    'images_used': len(image_paths),
                    'corner_images': corner_images,
                    'undistorted_images': undistorted_images,
                    'message': f'Intrinsic calibration completed successfully using {len(image_paths)} images'
                }
            else:
                results = {
                    'success': False,
                    'error': 'Intrinsic calibration failed'
                }
        
        elif calibration_type == 'eye_in_hand':
            # Eye-in-hand calibration
            if 'poses' not in session_info:
                return jsonify({'error': 'Pose files required for eye-in-hand calibration'}), 400
            
            # First do intrinsic calibration if not already done
            if not intrinsic_calibrator.calibration_completed:
                ret, camera_matrix, dist_coeffs = intrinsic_calibrator.calibrate_from_images(
                    image_paths, XX, YY, L, distortion_model, verbose=True)
                if not ret:
                    return jsonify({'error': 'Intrinsic calibration failed'}), 500
            
            # Load camera intrinsics into eye-in-hand calibrator
            camera_matrix, dist_coeffs = intrinsic_calibrator.get_parameters()
            eye_in_hand_calibrator.load_camera_intrinsics(camera_matrix, dist_coeffs)
            
            # Load pose data - manually handle separate images and poses folders
            session_folder = get_session_folder(session_id)
            poses_folder = os.path.join(session_folder, 'poses')
            
            try:
                # Use the already loaded image paths (from the images we processed for intrinsic calibration)
                image_paths_sorted = image_paths
                
                # Load poses from poses folder
                pose_json_paths = []
                import glob
                pose_json_paths = glob.glob(os.path.join(poses_folder, "*.json"))
                pose_json_paths = sorted(pose_json_paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0])
                                       if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
                
                if len(image_paths_sorted) != len(pose_json_paths):
                    return jsonify({'error': f'Number of images ({len(image_paths_sorted)}) does not match number of pose files ({len(pose_json_paths)})'}), 400
                
                # Load end-effector poses
                from core.utils import xyz_rpy_to_matrix, inverse_transform_matrix
                end2base_matrices = []
                for pose_path in pose_json_paths:
                    with open(pose_path, 'r', encoding='utf-8') as f:
                        pose_data = json.load(f)
                        end_xyzrpy_dict = pose_data["end_xyzrpy"]
                        xyzrpy = np.array([
                            end_xyzrpy_dict["x"],
                            end_xyzrpy_dict["y"], 
                            end_xyzrpy_dict["z"],
                            end_xyzrpy_dict["rx"],
                            end_xyzrpy_dict["ry"],
                            end_xyzrpy_dict["rz"]
                        ])
                    
                    end2base_matrix = xyz_rpy_to_matrix(xyzrpy)
                    end2base_matrices.append(end2base_matrix)

                base2end_matrices = [inverse_transform_matrix(matrix) for matrix in end2base_matrices]
                
                print(f"Successfully loaded {len(image_paths_sorted)} calibration images and poses")
                
                # Run eye-in-hand calibration
                cam2end_R, cam2end_t, cam2end_4x4, rvecs, tvecs = eye_in_hand_calibrator.calibrate(
                    image_paths_sorted, end2base_matrices, XX, YY, L, verbose=True)
                
                # Calculate reprojection errors
                errors, target2base_matrices = eye_in_hand_calibrator.calculate_reprojection_errors(
                    image_paths_sorted, base2end_matrices, end2base_matrices, 
                    rvecs, tvecs, XX, YY, L, vis=True, 
                    save_dir=os.path.join(RESULTS_FOLDER, session_id, 'visualizations'))
                
                initial_mean_error = float(np.mean(errors))
                print(f"Initial mean reprojection error: {initial_mean_error:.6f} pixels")
                
                # Add optimization step (like in the legacy script)
                print("Starting calibration optimization...")
                optimized_cam2end, optimized_target2base = eye_in_hand_calibrator.optimize_calibration(
                    image_paths_sorted, rvecs, tvecs, end2base_matrices, base2end_matrices,
                    XX, YY, L, iterations=5, ftol_rel=1e-6)
                
                # Calculate final optimized errors
                end2cam_4x4 = np.linalg.inv(optimized_cam2end)
                final_errors = []
                
                for i, base2end_matrix in enumerate(base2end_matrices):
                    target2cam_4x4 = end2cam_4x4 @ base2end_matrix @ optimized_target2base
                    
                    # Calculate reprojection error for this image
                    from core.utils import calculate_single_image_reprojection_error
                    
                    error = calculate_single_image_reprojection_error(
                        image_paths_sorted[i], target2cam_4x4[:3, :3], target2cam_4x4[:3, 3],
                        camera_matrix, dist_coeffs, XX, YY, L)
                    final_errors.append(error)
                    
                    # Generate optimized visualization
                    eye_in_hand_calibrator._generate_reprojection_visualization(
                        image_paths_sorted[i], target2cam_4x4, XX, YY, L, 
                        os.path.join(RESULTS_FOLDER, session_id, 'visualizations'), suffix="optimized")
                
                final_errors = np.array(final_errors)
                optimized_mean_error = float(np.mean(final_errors))
                print(f"Optimized mean reprojection error: {optimized_mean_error:.6f} pixels")
                
                # Update the calibrator with optimized results
                eye_in_hand_calibrator.cam2end_matrix = optimized_cam2end
                eye_in_hand_calibrator.target2base_matrix = optimized_target2base
                eye_in_hand_calibrator.optimization_completed = True
                
                # Save results
                results_folder = os.path.join(RESULTS_FOLDER, session_id)
                eye_in_hand_calibrator.save_results(results_folder)
                
                results = {
                    'success': True,
                    'calibration_type': 'eye_in_hand',
                    'cam2end_matrix': optimized_cam2end.tolist(),
                    'initial_reprojection_errors': errors.tolist(),
                    'optimized_reprojection_errors': final_errors.tolist(),
                    'initial_mean_error': initial_mean_error,
                    'optimized_mean_error': optimized_mean_error,
                    'improvement_percentage': float((initial_mean_error - optimized_mean_error) / initial_mean_error * 100),
                    'message': f'Eye-in-hand calibration completed with optimization. Initial error: {initial_mean_error:.4f} pixels, Optimized error: {optimized_mean_error:.4f} pixels ({(initial_mean_error - optimized_mean_error) / initial_mean_error * 100:.1f}% improvement)'
                }
                
            except Exception as e:
                return jsonify({'error': f'Eye-in-hand calibration failed: {str(e)}'}), 500
        
        # Store results in session
        session_data[session_id]['results'] = results
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_results/<session_id>')
def get_results(session_id):
    """Get calibration results and visualization images."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        session_info = session_data[session_id]
        
        if 'results' not in session_info:
            return jsonify({'error': 'No calibration results found'}), 404
        
        results = session_info['results'].copy()
        
        # Add visualization images if available
        viz_folder = os.path.join(RESULTS_FOLDER, session_id, 'visualizations')
        if os.path.exists(viz_folder):
            viz_images = []
            for filename in os.listdir(viz_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(viz_folder, filename)
                    img_base64 = image_to_base64(img_path)
                    if img_base64:
                        viz_images.append({
                            'name': filename,
                            'data': img_base64
                        })
            results['visualization_images'] = viz_images
        
        # Add original images for comparison
        if 'images' in session_info:
            original_images = []
            for img_info in session_info['images']:
                img_base64 = image_to_base64(img_info['path'])
                if img_base64:
                    original_images.append({
                        'name': img_info['name'],
                        'data': img_base64
                    })
            results['original_images'] = original_images
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_results/<session_id>')
def export_results(session_id):
    """Export calibration results as a ZIP file."""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        if not os.path.exists(results_folder):
            return jsonify({'error': 'No results found for this session'}), 404
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(results_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, results_folder)
                    zipf.write(file_path, archive_path)
        
        return send_file(temp_zip.name, as_attachment=True, 
                        download_name=f'calibration_results_{session_id}.zip',
                        mimetype='application/zip')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<session_id>/<filename>')
def get_image(session_id, filename):
    """Serve uploaded images."""
    try:
        session_folder = get_session_folder(session_id)
        image_path = os.path.join(session_folder, 'images', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/corner_image/<session_id>/<filename>')
def get_corner_image(session_id, filename):
    """Serve corner detection visualization images."""
    try:
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        image_path = os.path.join(results_folder, 'corner_detection', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Corner image not found'}), 404
        
        return send_file(image_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/undistorted_image/<session_id>/<filename>')
def get_undistorted_image(session_id, filename):
    """Serve undistorted images."""
    try:
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        image_path = os.path.join(results_folder, 'undistorted', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Undistorted image not found'}), 404
        
        return send_file(image_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_session/<session_id>', methods=['POST'])
def clear_session(session_id):
    """Clear session data and files."""
    try:
        # Remove session data
        if session_id in session_data:
            del session_data[session_id]
        
        # Remove session files
        session_folder = get_session_folder(session_id)
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)
        
        # Remove results
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        
        return jsonify({'message': 'Session cleared successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
