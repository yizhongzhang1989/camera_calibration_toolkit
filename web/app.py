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
import sys

# Import core calibration modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import create_pattern_from_json, get_pattern_type_configurations
from core.eye_in_hand_calibration import EyeInHandCalibrator
from web.visualization_utils import trim_distortion_coefficients

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Template helper for shared components
@app.context_processor
def utility_processor():
    def include_shared_template(template_name):
        """Include a shared template component"""
        try:
            return render_template(f'shared/{template_name}')
        except:
            return f'<!-- Shared template {template_name} not found -->'
    
    return dict(include_shared_template=include_shared_template)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
ALLOWED_POSE_EXTENSIONS = {'json'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Session data storage (in production, use proper session management)
session_data = {}

# Console output storage
console_outputs = {}

class ConsoleCapture:
    def __init__(self, session_id):
        self.session_id = session_id
        self.original_stdout = sys.stdout
        if session_id not in console_outputs:
            console_outputs[session_id] = []
    
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            console_outputs[self.session_id].append(f"[{timestamp}] {text.strip()}")
            if len(console_outputs[self.session_id]) > 100:
                console_outputs[self.session_id] = console_outputs[self.session_id][-100:]
    
    def flush(self):
        self.original_stdout.flush()
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout


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
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
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
            
            result = f"data:image/{format_str};base64,{img_base64}"
            print(f"✅ Converted image to base64: {image_path} ({len(img_data)} bytes -> {len(result)} chars)")
            return result
    except Exception as e:
        print(f"❌ Failed to convert image to base64: {image_path}, error: {e}")
        return None


def create_pattern_from_parameters(parameters):
    """Create calibration pattern from session parameters using JSON system."""
    pattern_type = parameters.get('pattern_type', 'standard')
    pattern_params = parameters.get('pattern_parameters', {})
    
    if pattern_type == 'standard':
        width = pattern_params.get('width', parameters.get('chessboard_x', 11))
        height = pattern_params.get('height', parameters.get('chessboard_y', 8))
        square_size = pattern_params.get('square_size', parameters.get('square_size', 0.02))
        
        pattern_json = {
            'pattern_id': 'standard_chessboard',
            'name': 'Standard Chessboard',
            'description': 'Standard chessboard calibration pattern',
            'parameters': {
                'width': width,
                'height': height,
                'square_size': square_size
            }
        }
        
    elif pattern_type == 'charuco':
        width = pattern_params.get('width', parameters.get('chessboard_x', 8))
        height = pattern_params.get('height', parameters.get('chessboard_y', 6))
        square_size = pattern_params.get('square_size', parameters.get('square_size', 0.040))
        marker_size = pattern_params.get('marker_size', parameters.get('marker_size', 0.020))
        dict_id_raw = pattern_params.get('dictionary_id', parameters.get('dictionary_id', cv2.aruco.DICT_6X6_250))
        dictionary_id = int(dict_id_raw) if isinstance(dict_id_raw, (str, float)) else dict_id_raw
        
        pattern_json = {
            'pattern_id': 'charuco_board',
            'name': 'ChArUco Board',
            'description': 'ChArUco board calibration pattern',
            'parameters': {
                'width': width,
                'height': height,
                'square_size': square_size,
                'marker_size': marker_size,
                'dictionary_id': dictionary_id
            }
        }
        
    else:
        raise ValueError(f'Unsupported pattern type: {pattern_type}')
    
    return create_pattern_from_json(pattern_json)


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


@app.route('/api/console/<session_id>')
def get_console_output(session_id):
    output = console_outputs.get(session_id, [])
    return jsonify({'console_output': output})


@app.route('/api/console/<session_id>/clear')
def clear_console_output(session_id):
    if session_id in console_outputs:
        console_outputs[session_id] = []
    return jsonify({'success': True})


@app.route('/api/pattern_configurations')
def get_pattern_configurations():
    """Get available pattern type configurations."""
    try:
        configurations = get_pattern_type_configurations()
        return jsonify({'success': True, 'configurations': configurations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/chessboard-test')
def chessboard_test():
    """Chessboard configuration test page."""
    return render_template('chessboard-test.html')


@app.route('/simple-test')
def simple_test():
    """Simple chessboard configuration test page."""
    return render_template('simple-test.html')


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
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        # Extract parameters - handle both new JSON pattern and legacy formats
        parameters = {}
        
        # Pattern JSON (new format)
        if 'pattern_json' in data:
            pattern_json = data['pattern_json']
            # Store the complete JSON pattern for use with unified pattern system
            parameters['pattern_json'] = pattern_json
            
            # For backward compatibility, also extract individual parameters
            pattern_params = pattern_json.get('parameters', {})
            parameters['pattern_type'] = pattern_json.get('pattern_id', 'standard')
            parameters['chessboard_x'] = pattern_params.get('width', 11)
            parameters['chessboard_y'] = pattern_params.get('height', 8) 
            parameters['square_size'] = pattern_params.get('square_size', 0.02)
            
            # Add ChArUco-specific parameters if present
            if 'marker_size' in pattern_params:
                parameters['marker_size'] = pattern_params['marker_size']
            if 'dictionary_id' in pattern_params:
                parameters['dictionary_id'] = pattern_params['dictionary_id']
                
            print(f'Using JSON pattern format: {pattern_json}')
        elif 'pattern_config' in data:
            # Legacy pattern_config format (for backward compatibility)
            pattern_config = data['pattern_config']
            parameters['pattern_type'] = pattern_config.get('patternType', 'standard')
            parameters['pattern_parameters'] = pattern_config.get('parameters', {})
            
            # For backward compatibility, also set legacy parameters
            if parameters['pattern_type'] == 'standard':
                parameters['chessboard_x'] = pattern_config.get('parameters', {}).get('width', 11)
                parameters['chessboard_y'] = pattern_config.get('parameters', {}).get('height', 8)
                parameters['square_size'] = pattern_config.get('parameters', {}).get('square_size', 0.02)
            elif parameters['pattern_type'] == 'charuco':
                parameters['chessboard_x'] = pattern_config.get('parameters', {}).get('width', 8)  # ChArUco squares
                parameters['chessboard_y'] = pattern_config.get('parameters', {}).get('height', 6)
                parameters['square_size'] = pattern_config.get('parameters', {}).get('square_size', 0.040)
                parameters['marker_size'] = pattern_config.get('parameters', {}).get('marker_size', 0.020)
                # Convert dictionary_id to integer to fix OpenCV error
                dict_id_raw = pattern_config.get('parameters', {}).get('dictionary_id', cv2.aruco.DICT_6X6_250)
                parameters['dictionary_id'] = int(dict_id_raw) if isinstance(dict_id_raw, (str, float)) else dict_id_raw
        else:
            # Legacy format - direct parameter extraction
            parameters['pattern_type'] = 'standard'  # Default to standard for legacy
            parameters['chessboard_x'] = data.get('chessboard_x')
            parameters['chessboard_y'] = data.get('chessboard_y')
            parameters['square_size'] = data.get('square_size')
        
        parameters['distortion_model'] = data.get('distortion_model', 'standard')
        
        # Add eye-in-hand specific parameters
        if data.get('handeye_method'):
            parameters['handeye_method'] = data.get('handeye_method')
        if data.get('camera_matrix_source'):
            parameters['camera_matrix_source'] = data.get('camera_matrix_source')
            
        # Add manual camera parameters if provided
        if data.get('fx') is not None:
            parameters['fx'] = data.get('fx')
        if data.get('fy') is not None:
            parameters['fy'] = data.get('fy')
        if data.get('cx') is not None:
            parameters['cx'] = data.get('cx')
        if data.get('cy') is not None:
            parameters['cy'] = data.get('cy')
        if data.get('distortion_coefficients') is not None:
            parameters['distortion_coefficients'] = data.get('distortion_coefficients')
        
        # Validate required parameters
        required_params = ['chessboard_x', 'chessboard_y', 'square_size']
        for param in required_params:
            if parameters[param] is None:
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
    import time
    start_time = time.time()
    
    try:
        session_id = request.json.get('session_id', 'default')
        
        # Initialize console
        console_outputs[session_id] = []
        
        with ConsoleCapture(session_id):
            print("=" * 60)
            print(f"🚀 STARTING CALIBRATION FOR SESSION: {session_id}")
            print(f"�?Start time: {time.strftime('%H:%M:%S')}")
            print("=" * 60)
            
            if session_id not in session_data:
                print("�?ERROR: No session data found")
                return jsonify({'error': 'No session data found'}), 400
            
            session_info = session_data[session_id]
            
            # Check required data
            if 'images' not in session_info or 'parameters' not in session_info:
                print("�?ERROR: Missing images or parameters")
                return jsonify({'error': 'Missing images or parameters'}), 400
            
            print(f"�?Session data validated - images: {len(session_info.get('images', []))}")
            
            images = session_info['images']
            parameters = session_info['parameters']
            calibration_type = request.json.get('calibration_type', session_info.get('calibration_type', 'intrinsic'))
            selected_indices = request.json.get('selected_indices', list(range(len(images))))
            
            print(f"📋 Calibration type: {calibration_type}")
            print(f"🖼�? Selected {len(selected_indices)} images for processing")
            
            # Extract basic parameters
            distortion_model = parameters.get('distortion_model', 'standard')
            
            # For legacy compatibility and logging, try to get pattern dimensions
            pattern_type = parameters.get('pattern_type', 'standard')
            if pattern_type == 'standard':
                pattern_params = parameters.get('pattern_parameters', {})
                XX = pattern_params.get('width', parameters.get('chessboard_x', 11))
                YY = pattern_params.get('height', parameters.get('chessboard_y', 8))
                L = pattern_params.get('square_size', parameters.get('square_size', 0.02))
                print(f"🏁 Standard Chessboard: {XX}x{YY} corners, Square size: {L}m")
            elif pattern_type == 'charuco':
                pattern_params = parameters.get('pattern_parameters', {})
                XX = pattern_params.get('width', parameters.get('chessboard_x', 8))
                YY = pattern_params.get('height', parameters.get('chessboard_y', 6))
                L = pattern_params.get('square_size', parameters.get('square_size', 0.040))
                marker_size = pattern_params.get('marker_size', parameters.get('marker_size', 0.020))
                print(f"🎯 ChArUco Board: {XX}x{YY} squares, Square size: {L}m, Marker size: {marker_size}m")
            else:
                # Fallback to legacy parameters for unknown types
                XX = parameters.get('chessboard_x', 11)
                YY = parameters.get('chessboard_y', 8)
                L = parameters.get('square_size', 0.02)
            
            print(f"📐 Distortion model: {distortion_model}")
            print("-" * 50)
            
            # Get image paths - only selected ones
            image_paths = [images[i]['path'] for i in selected_indices if i < len(images)]
            
            if len(image_paths) < 3:
                print("�?ERROR: Need at least 3 selected images for calibration")
                return jsonify({'error': 'Need at least 3 selected images for calibration'}), 400
            
            print(f"📂 Processing image files:")
            for i, path in enumerate(image_paths):
                filename = os.path.basename(path)
                print(f"   {i+1}. {filename}")
            print()
            
            results = {}
            
            if calibration_type == 'intrinsic':
                print("🔧 INTRINSIC CALIBRATION SETUP")
                print("-" * 30)
                
                # Get pattern configuration
                pattern_type = parameters.get('pattern_type', 'standard')
                pattern_params = parameters.get('pattern_parameters', {})
                
                # Create calibration pattern using unified JSON system
                try:
                    if 'pattern_json' in parameters:
                        # Use the JSON pattern directly
                        pattern_json = parameters['pattern_json']
                        pattern = create_pattern_from_json(pattern_json)
                        print(f"✅ Created pattern from JSON: {pattern_json['pattern_id']}")
                    else:
                        # Fallback to create from individual parameters
                        pattern = create_pattern_from_parameters(parameters)
                        print(f"✅ Created pattern using unified JSON system")
                except Exception as e:
                    print(f"❌ Error creating calibration pattern: {str(e)}")
                    return jsonify({'error': f'Error creating calibration pattern: {str(e)}'}), 400
                
                print()
                
                # Create new calibrator instance for this session
                # Create new calibrator instance for this session
                calibrator = IntrinsicCalibrator(
                    image_paths=image_paths,
                    calibration_pattern=pattern,
                    pattern_type=pattern_type
                )
                
                print("🎯 PATTERN DETECTION PHASE")
                print("-" * 30)
                
                # Run pattern detection
                detection_success = calibrator.detect_pattern_points(verbose=True)
                    
                if not detection_success:
                    print("�?Pattern detection failed. Check your images and parameters.")
                    return jsonify({'error': 'Pattern detection failed. Check your images and parameters.'}), 400
                
                print()
                print("🧮 CAMERA CALIBRATION PHASE")
                print("-" * 30)
                print("Running OpenCV calibrateCamera...")
                print()
                
                # Run calibration
                rms_error = calibrator.calibrate_camera(verbose=True)
                
                if rms_error > 0:
                    print()
                    print("💾 SAVING CALIBRATION RESULTS")
                    print("-" * 30)
                    
                    # Get results
                    camera_matrix = calibrator.get_camera_matrix()
                    dist_coeffs = calibrator.get_distortion_coefficients()
                    
                    # Save results
                    results_folder = os.path.join(RESULTS_FOLDER, session_id)
                    os.makedirs(results_folder, exist_ok=True)
                    
                    print(f"📁 Results folder: {os.path.basename(results_folder)}")
                    
                    calibrator.save_calibration(
                        os.path.join(results_folder, 'calibration_results.json'),
                        include_extrinsics=True
                    )
                    
                    print("�?Saved main calibration results to calibration_results.json")
                    
                    # Save additional JSON with trimmed distortion coefficients for backward compatibility
                    trimmed_dist_coeffs = trim_distortion_coefficients(dist_coeffs, distortion_model)
                    
                    # Save additional legacy format for backward compatibility
                    legacy_calibration_data = {
                        "camera_matrix": camera_matrix.tolist(),
                        "distortion_coefficients": trimmed_dist_coeffs.tolist(),
                        "distortion_model": distortion_model,
                        "rms_error": float(rms_error),
                        "images_used": len(image_paths)
                    }
                    
                    with open(os.path.join(results_folder, 'legacy_calibration_results.json'), 'w') as f:
                        json.dump(legacy_calibration_data, f, indent=4)
                    
                    print("�?Saved legacy format to legacy_calibration_results.json")
                    
                    # Also save trimmed distortion coefficients separately
                    trimmed_dist_dict = {
                        "distortion_coefficients": trimmed_dist_coeffs.tolist(),
                        "distortion_model": distortion_model,
                        "note": f"Trimmed to {len(trimmed_dist_coeffs)} coefficients for {distortion_model} model"
                    }
                    with open(os.path.join(results_folder, 'dist_trimmed.json'), 'w') as f:
                        json.dump(trimmed_dist_dict, f, indent=4)
                    
                    print("�?Saved trimmed distortion coefficients to dist_trimmed.json")
                    print()
                    print("🖼�? GENERATING VISUALIZATION IMAGES")
                    print("-" * 30)
                    
                    # Use the calibrator's built-in visualization methods
                    # Generate corner detection images
                    corner_viz_dir = os.path.join(results_folder, 'corner_visualizations')
                    os.makedirs(corner_viz_dir, exist_ok=True)
                    
                    pattern_images = calibrator.draw_pattern_on_images()
                    corner_images = []
                    
                    for filename, debug_img in pattern_images:
                        corner_filename = f"{filename}_corners.jpg"
                        corner_path = os.path.join(corner_viz_dir, corner_filename)
                        cv2.imwrite(corner_path, debug_img)
                        
                        corner_images.append({
                            'name': corner_filename,
                            'path': corner_path,
                            'url': url_for('get_corner_image', session_id=session_id, filename=corner_filename),
                            'index': len(corner_images),  # Use sequential index
                            'original_name': filename
                        })
                    
                    # Generate undistorted images with 3D axes
                    undistorted_dir = os.path.join(results_folder, 'undistorted')
                    os.makedirs(undistorted_dir, exist_ok=True)
                    
                    axes_images = calibrator.draw_axes_on_undistorted_images()
                    undistorted_images = []
                    
                    for filename, debug_img in axes_images:
                        undistorted_filename = f"{filename}_undistorted.jpg"
                        undistorted_path = os.path.join(undistorted_dir, undistorted_filename)
                        cv2.imwrite(undistorted_path, debug_img)
                        
                        undistorted_images.append({
                            'name': undistorted_filename,
                            'path': undistorted_path,
                            'url': url_for('get_undistorted_image', session_id=session_id, filename=undistorted_filename),
                            'index': len(undistorted_images),  # Use sequential index
                            'original_name': filename
                        })
                    
                    # Create results dictionary directly
                    results = {
                        'success': True,
                        'calibration_type': 'intrinsic',
                        'images_used': len(image_paths),
                        'corner_images': corner_images,
                        'undistorted_images': undistorted_images,
                        'camera_matrix': camera_matrix.tolist(),
                        'distortion_coefficients': trimmed_dist_coeffs.tolist(),
                        'distortion_model': distortion_model,
                        'rms_error': float(rms_error),
                        'message': f'Intrinsic calibration completed successfully using {len(image_paths)} images'
                    }
                        
                    print("�?Generated corner detection and undistorted images")
                    print()
                    calibration_file_path = os.path.join(results_folder, 'calibration_results.json')
                    print(f"�?Calibration data saved to: {calibration_file_path}")
                    
                    # Calculate and display total time
                    total_time = time.time() - start_time
                    print(f"⏱️  Total calibration time: {total_time:.2f} seconds")
                    print()
                    print("🎉 CALIBRATION COMPLETED SUCCESSFULLY!")
                    print("=" * 60)
                else:
                    total_time = time.time() - start_time
                    print("�?CALIBRATION FAILED")
                    print("-" * 20)
                    print("Possible causes:")
                    print("�?Insufficient pattern detections")
                    print("�?Poor image quality")
                    print("�?Incorrect chessboard parameters")
                    print("�?Images too similar (lack of variety)")
                    print(f"⏱️  Time elapsed: {total_time:.2f} seconds")
                    print("=" * 60)
                    results = {
                        'success': False,
                        'error': 'Intrinsic calibration failed'
                    }
            
            elif calibration_type == 'eye_in_hand':
                # Eye-in-hand calibration
                if 'poses' not in session_info:
                    print("�?ERROR: Pose files required for eye-in-hand calibration")
                    return jsonify({'error': 'Pose files required for eye-in-hand calibration'}), 400
                
                # Handle camera intrinsics - either from previous calibration or manual input
                camera_matrix_source = parameters.get('camera_matrix_source', 'intrinsic')
                
                if camera_matrix_source == 'manual':
                    # Use manual camera parameters
                    fx = float(parameters.get('fx', 800))
                    fy = float(parameters.get('fy', 800))
                    cx = float(parameters.get('cx', 320))
                    cy = float(parameters.get('cy', 240))
                    dist_coeffs = np.array(parameters.get('distortion_coefficients', [0, 0, 0, 0, 0]), dtype=np.float32)
                    
                    camera_matrix = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    
                    print(f"Using manual camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                    
                else:
                    # Use intrinsic calibration results
                    # Create a new calibrator for intrinsic calculation if needed
                    
                    # Create calibration pattern using unified JSON system
                    try:
                        if 'pattern_json' in parameters:
                            # Use the JSON pattern directly
                            pattern_json = parameters['pattern_json']
                            pattern = create_pattern_from_json(pattern_json)
                            print(f"✅ Created pattern from JSON: {pattern_json['pattern_id']}")
                        else:
                            # Fallback to create from individual parameters
                            pattern = create_pattern_from_parameters(parameters)
                            print(f"✅ Created pattern using unified JSON system")
                    except Exception as e:
                        print(f"❌ Error creating calibration pattern: {str(e)}")
                        return jsonify({'error': f'Error creating calibration pattern: {str(e)}'}), 400
                    
                    intrinsic_cal = IntrinsicCalibrator(
                        image_paths=image_paths,
                        calibration_pattern=pattern,
                        pattern_type=parameters.get('pattern_type', 'standard')
                    )
                    
                    if not intrinsic_cal.detect_pattern_points(verbose=True):
                        print("�?Pattern detection failed for eye-in-hand calibration")
                        return jsonify({'error': 'Pattern detection failed for eye-in-hand calibration'}), 400
                    
                    rms_error = intrinsic_cal.calibrate_camera(verbose=True)
                    if rms_error <= 0:
                        print("�?Intrinsic calibration failed")
                        return jsonify({'error': 'Intrinsic calibration failed'}), 500
                    
                    camera_matrix = intrinsic_cal.get_camera_matrix()
                    dist_coeffs = intrinsic_cal.get_distortion_coefficients()
                
                # Create eye-in-hand calibrator instance with modern API
                # Reuse the same pattern that was used for intrinsic calibration
                eye_in_hand_calibrator = EyeInHandCalibrator(
                    camera_matrix=camera_matrix,
                    distortion_coefficients=dist_coeffs,
                    calibration_pattern=pattern,  # Use same pattern from intrinsic calibration
                    pattern_type=parameters.get('pattern_type', 'standard')
                )
                
                print(f"✅ Eye-in-hand calibrator initialized with modern API")
                
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
                        print(f"�?ERROR: Number of images ({len(image_paths_sorted)}) does not match number of pose files ({len(pose_json_paths)})")
                        return jsonify({'error': f'Number of images ({len(image_paths_sorted)}) does not match number of pose files ({len(pose_json_paths)})'}), 400
                    
                    # Load end-effector poses and convert to format expected by modern API
                    from core.utils import xyz_rpy_to_matrix, inverse_transform_matrix
                    robot_poses = []
                    for pose_path in pose_json_paths:
                        with open(pose_path, 'r', encoding='utf-8') as f:
                            pose_data = json.load(f)
                            # Modern API expects the full pose_data dict structure
                            robot_poses.append(pose_data)

                    print(f"Successfully loaded {len(image_paths_sorted)} calibration images and poses")
                    
                    # Load data into calibrator using modern member-based API
                    # Set images from paths
                    eye_in_hand_calibrator.set_images_from_paths(image_paths_sorted)
                    
                    # Set robot poses using the original JSON data format
                    eye_in_hand_calibrator.set_robot_poses(robot_poses)
                    
                    # Detect pattern points using member-based API
                    print("🎯 Detecting calibration patterns...")
                    if not eye_in_hand_calibrator.detect_pattern_points():
                        print("❌ Failed to detect calibration patterns")
                        return jsonify({'error': 'Failed to detect calibration patterns in images'}), 500
                    
                    print(f"✅ Pattern detection completed: {len(eye_in_hand_calibrator.image_points)} images")
                    
                    # Run eye-in-hand calibration using modern API with method comparison
                    print("🔧 Performing hand-eye calibration with method comparison...")
                    
                    methods = [
                        (cv2.CALIB_HAND_EYE_TSAI, "TSAI"),
                        (cv2.CALIB_HAND_EYE_PARK, "PARK"), 
                        (cv2.CALIB_HAND_EYE_HORAUD, "HORAUD"),
                        (cv2.CALIB_HAND_EYE_ANDREFF, "ANDREFF"),
                        (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS")
                    ]
                    
                    best_method = None
                    best_error = float('inf')
                    
                    for method_const, method_name in methods:
                        try:
                            # Reset calibrator state for fair comparison
                            eye_in_hand_calibrator.cam2end_matrix = None
                            eye_in_hand_calibrator.calibration_completed = False
                            eye_in_hand_calibrator.rms_error = None
                            
                            # Test this method
                            rms_error = eye_in_hand_calibrator.calibrate(method=method_const, verbose=False)
                            
                            if rms_error > 0 and rms_error < best_error:
                                best_error = rms_error
                                best_method = method_name
                                print(f"✅ {method_name}: {rms_error:.4f} pixels")
                            else:
                                print(f"❌ {method_name}: Failed or poor result")
                                
                        except Exception as e:
                            print(f"❌ {method_name}: Exception - {str(e)}")
                    
                    if best_method:
                        print(f"🏆 Best method: {best_method} with {best_error:.4f} pixels RMS error")
                        initial_mean_error = float(best_error)
                    else:
                        print("❌ All hand-eye calibration methods failed")
                        return jsonify({'error': 'All hand-eye calibration methods failed'}), 500
                    
                    # Add optimization step using modern API
                    print("🔍 Starting calibration optimization...")
                    optimized_error = eye_in_hand_calibrator.optimize_calibration(
                        iterations=5,
                        ftol_rel=1e-6,
                        verbose=True
                    )
                    
                    if optimized_error > 0:
                        final_mean_error = float(optimized_error)
                        improvement_percentage = float((initial_mean_error - final_mean_error) / initial_mean_error * 100)
                        print(f"Optimized RMS reprojection error: {final_mean_error:.6f} pixels")
                        print(f"Improvement: {improvement_percentage:.1f}%")
                    else:
                        final_mean_error = initial_mean_error
                        improvement_percentage = 0.0
                        print("⚠️ Optimization completed with no improvement")
                    
                    # Save results
                    results_folder = os.path.join(RESULTS_FOLDER, session_id)
                    eye_in_hand_calibrator.save_results(results_folder)
                    
                    try:
                        # Generate visualization images using modern API
                        print("📸 Generating visualization images...")
                        
                        # Save debug images to separate visualization folders
                        viz_base_folder = os.path.join(RESULTS_FOLDER, session_id, 'visualizations')
                        corners_folder = os.path.join(viz_base_folder, 'corner_detection')
                        axes_folder = os.path.join(viz_base_folder, 'undistorted_axes')
                        reprojection_folder = os.path.join(viz_base_folder, 'reprojection')
                        
                        os.makedirs(corners_folder, exist_ok=True)
                        os.makedirs(axes_folder, exist_ok=True)
                        os.makedirs(reprojection_folder, exist_ok=True)
                        
                        # Generate and save corner detection images
                        corner_detection_images = eye_in_hand_calibrator.draw_pattern_on_images()
                        corner_detection_paths = []
                        for filename, debug_img in corner_detection_images:
                            # Use original filename without extra suffix
                            output_path = os.path.join(corners_folder, f"{filename}.jpg")
                            cv2.imwrite(output_path, debug_img)
                            corner_detection_paths.append(f"visualizations/corner_detection/{filename}.jpg")
                        
                        # Generate and save undistorted images with axes
                        undistorted_images = eye_in_hand_calibrator.draw_axes_on_undistorted_images()
                        undistorted_paths = []
                        for filename, debug_img in undistorted_images:
                            # Use original filename without extra suffix
                            output_path = os.path.join(axes_folder, f"{filename}.jpg")
                            cv2.imwrite(output_path, debug_img)
                            undistorted_paths.append(f"visualizations/undistorted_axes/{filename}.jpg")
                        
                        # Generate and save reprojection images
                        reprojection_images = eye_in_hand_calibrator.draw_reprojection_on_images()
                        reprojection_paths = []
                        for filename, debug_img in reprojection_images:
                            # Use original filename without extra suffix
                            output_path = os.path.join(reprojection_folder, f"{filename}.jpg")
                            cv2.imwrite(output_path, debug_img)
                            reprojection_paths.append(f"visualizations/reprojection/{filename}.jpg")
                        
                        print(f"✅ Saved {len(corner_detection_paths)} corner detection images to {corners_folder}")
                        print(f"✅ Saved {len(undistorted_paths)} undistorted axes images to {axes_folder}") 
                        print(f"✅ Saved {len(reprojection_paths)} reprojection images to {reprojection_folder}")
                        print(f"✅ Saved {len(undistorted_paths)} undistorted axes images") 
                        print(f"✅ Saved {len(reprojection_paths)} reprojection images")
                        
                    except Exception as viz_error:
                        print(f"⚠️ Warning: Visualization generation failed: {viz_error}")
                        # Use empty lists if visualization fails
                        corner_detection_paths = []
                        undistorted_paths = []
                        reprojection_paths = []
                    
                    # Ensure all matrices are converted to lists for JSON serialization
                    cam2end_matrix = eye_in_hand_calibrator.cam2end_matrix
                    if cam2end_matrix is not None:
                        cam2end_list = cam2end_matrix.tolist()
                    else:
                        cam2end_list = None
                        print("⚠️ Warning: cam2end_matrix is None")
                    
                    results = {
                        'success': True,
                        'calibration_type': 'eye_in_hand',
                        'session_id': session_id,
                        'camera_matrix': camera_matrix.tolist(),
                        'distortion_coefficients': dist_coeffs.tolist(),
                        'handeye_transform': cam2end_list,
                        'cam2end_matrix': cam2end_list,
                        'reprojection_error': float(final_mean_error),
                        'initial_mean_error': float(initial_mean_error),
                        'optimized_mean_error': float(final_mean_error),
                        'improvement_percentage': float(improvement_percentage),
                        'best_calibration_method': str(best_method) if best_method else "Unknown",
                        'corner_detection_images': corner_detection_paths,
                        'undistorted_images': undistorted_paths,
                        'reprojection_images': reprojection_paths,
                        'pattern_info': {
                            'width': int(XX),
                            'height': int(YY),
                            'square_size': float(L)
                        },
                        'message': f'Eye-in-hand calibration completed successfully using {len(image_paths)} images. Improved error from {initial_mean_error:.4f} to {final_mean_error:.4f} pixels ({improvement_percentage:.1f}% improvement)'
                    }
                    
                except Exception as e:
                    print(f"�?Eye-in-hand calibration failed: {str(e)}")
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
        viz_base_folder = os.path.join(RESULTS_FOLDER, session_id, 'visualizations')
        
        # Collect all visualization images from the organized directory structure
        visualization_images = []
        
        # Add corner detection images
        if 'corner_detection_images' in results:
            for img_path in results['corner_detection_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Corner Detection - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'corner_detection'
                    })
        
        # Add undistorted images with axes
        if 'undistorted_images' in results:
            for img_path in results['undistorted_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Undistorted Axes - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'undistorted_axes'
                    })
        
        # Add reprojection images
        if 'reprojection_images' in results:
            for img_path in results['reprojection_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Reprojection - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'reprojection'
                    })
        
        # Fallback: check for images in the old structure or any other location
        if len(visualization_images) == 0:
            # Check the old single visualization folder
            old_viz_folder = os.path.join(RESULTS_FOLDER, session_id, 'visualizations')
            if os.path.exists(old_viz_folder):
                for filename in os.listdir(old_viz_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(old_viz_folder, filename)
                        img_base64 = image_to_base64(img_path)
                        if img_base64:
                            # Determine type from filename
                            img_type = 'other'
                            if '_corners' in filename:
                                img_type = 'corner_detection'
                            elif '_axes' in filename:
                                img_type = 'undistorted_axes'
                            elif '_reprojection' in filename:
                                img_type = 'reprojection'
                            
                            visualization_images.append({
                                'name': filename,
                                'data': img_base64,
                                'type': img_type
                            })
        
        results['visualization_images'] = visualization_images
        print(f"✅ Added {len(visualization_images)} visualization images to results")
        
        # Debug: Print details about the images we're returning
        if len(visualization_images) > 0:
            print("📊 Visualization Images Debug:")
            for i, img in enumerate(visualization_images):
                data_size = len(img['data']) if img['data'] else 0
                print(f"   {i+1}. {img['name']} (type: {img['type']}, data: {data_size} chars)")
        
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
        # Try both naming schemes for backward compatibility
        paths_to_try = [
            os.path.join(results_folder, 'corner_visualizations', filename),
            os.path.join(results_folder, 'corner_detection', filename)
        ]
        
        for image_path in paths_to_try:
            if os.path.exists(image_path):
                return send_file(image_path)
        
        return jsonify({'error': 'Corner image not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/undistorted_image/<session_id>/<filename>')
def get_undistorted_image(session_id, filename):
    """Serve undistorted images."""
    try:
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        # Try both naming schemes for backward compatibility
        paths_to_try = [
            os.path.join(results_folder, 'undistorted_images', filename),
            os.path.join(results_folder, 'undistorted', filename)
        ]
        
        for image_path in paths_to_try:
            if os.path.exists(image_path):
                return send_file(image_path)
        
        return jsonify({'error': 'Undistorted image not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualization_image/<session_id>/<filename>')
def get_visualization_image(session_id, filename):
    """Serve visualization/reprojection images."""
    try:
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        image_path = os.path.join(results_folder, 'visualizations', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Visualization image not found'}), 404
        
        return send_file(image_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pattern_image')
def get_pattern_image():
    """Generate and serve calibration pattern images."""
    try:
        # Get pattern parameters from query string
        pattern_type = request.args.get('pattern_type', 'standard')
        
        # Handle both legacy (corner_x/corner_y) and new (width/height) parameter names
        width = int(request.args.get('width', request.args.get('corner_x', 11)))
        height = int(request.args.get('height', request.args.get('corner_y', 8)))
        square_size = float(request.args.get('square_size', 0.02))
        
        # Simplified parameters
        pixel_per_square = int(request.args.get('pixel_per_square', 100))
        border_pixels = int(request.args.get('border_pixels', 0))
        
        # ChArUco specific parameters
        marker_size = float(request.args.get('marker_size', 0.0125))
        dictionary_id = int(request.args.get('dictionary_id', cv2.aruco.DICT_6X6_250))
        
        print(f"API: Creating pattern {pattern_type} with width={width}, height={height}, square_size={square_size}")
        
        # Create pattern instance
        if pattern_type == 'standard':
            pattern_json = {
                'pattern_id': 'standard_chessboard',
                'parameters': {
                    'width': width,
                    'height': height,
                    'square_size': square_size
                }
            }
            pattern = create_pattern_from_json(pattern_json)
        elif pattern_type == 'charuco':
            # For ChArUco, width and height represent squares, not corners
            print(f"API: Creating ChArUco with width={width}, height={height}, marker_size={marker_size}, dict={dictionary_id}")
            pattern_json = {
                'pattern_id': 'charuco_board',
                'parameters': {
                    'width': width,
                    'height': height,
                    'square_size': square_size,
                    'marker_size': marker_size,
                    'dictionary_id': dictionary_id
                }
            }
            pattern = create_pattern_from_json(pattern_json)
        else:
            return jsonify({'error': f'Unsupported pattern type: {pattern_type}'}), 400
        
        # Generate pattern image using simplified parameters
        pattern_image = pattern.generate_pattern_image(
            pixel_per_square=pixel_per_square,
            border_pixels=border_pixels
        )
        
        # Convert to bytes for HTTP response
        _, buffer = cv2.imencode('.png', pattern_image)
        
        # Create temporary file to serve
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.write(buffer.tobytes())
        temp_file.close()
        
        def remove_file(response):
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return response
        
        response = send_file(temp_file.name, mimetype='image/png')
        response.call_on_close(remove_file)
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pattern_description')
def get_pattern_description():
    """Get pattern description text."""
    try:
        # Get pattern parameters from query string
        pattern_type = request.args.get('pattern_type', 'standard')
        corner_x = int(request.args.get('corner_x', 11))
        corner_y = int(request.args.get('corner_y', 8))
        square_size = float(request.args.get('square_size', 0.02))
        
        # ChArUco specific parameters
        marker_size = float(request.args.get('marker_size', 0.0125))
        dictionary_id = int(request.args.get('dictionary_id', cv2.aruco.DICT_6X6_250))
        
        # Create pattern instance
        if pattern_type == 'standard':
            pattern_json = {
                'pattern_id': 'standard_chessboard',
                'parameters': {
                    'width': corner_x,
                    'height': corner_y,
                    'square_size': square_size
                }
            }
            pattern = create_pattern_from_json(pattern_json)
        elif pattern_type == 'charuco':
            pattern_json = {
                'pattern_id': 'charuco_board',
                'parameters': {
                    'width': corner_x,
                    'height': corner_y,
                    'square_size': square_size,
                    'marker_size': marker_size,
                    'dictionary_id': dictionary_id
                }
            }
            pattern = create_pattern_from_json(pattern_json)
        else:
            return jsonify({'error': f'Unsupported pattern type: {pattern_type}'}), 400
        
        # Get pattern information
        info = pattern.get_info()
        description = pattern.get_pattern_description()
        
        return jsonify({
            'description': description,
            'pattern_name': info['name'],
            'pattern_info': info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_images', methods=['POST'])
def clear_images():
    """Clear uploaded images for a session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
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
        
        return jsonify({'message': 'Images cleared successfully'})
        
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
