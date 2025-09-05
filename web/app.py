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
        print(f"�?Image not found: {image_path}")
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
            print(f"�?Converted image to base64: {image_path} ({len(img_data)} bytes -> {len(result)} chars)")
            return result
    except Exception as e:
        print(f"�?Failed to convert image to base64: {image_path}, error: {e}")
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


@app.route('/api/default_pattern_config')
def get_default_pattern_config():
    """Get default pattern configuration JSON."""
    try:
        configurations = get_pattern_type_configurations()
        
        # Return the first available pattern as default
        if configurations:
            # Get the first pattern
            first_pattern_id = list(configurations.keys())[0]
            pattern_config = configurations[first_pattern_id]
            
            # Build default parameters
            default_parameters = {}
            if 'parameters' in pattern_config:
                # Handle both list and dict formats
                if isinstance(pattern_config['parameters'], list):
                    for param in pattern_config['parameters']:
                        default_parameters[param['name']] = param.get('default', 0)
                else:
                    for param_name, param_config in pattern_config['parameters'].items():
                        default_parameters[param_name] = param_config.get('default', 0)
            
            # Create the JSON configuration
            default_json = {
                'pattern_id': pattern_config.get('id', first_pattern_id),
                'name': pattern_config.get('name', first_pattern_id),
                'description': pattern_config.get('description', f'{first_pattern_id} calibration pattern'),
                'is_planar': True,
                'parameters': default_parameters
            }
            
            return jsonify({'success': True, 'pattern_config': default_json})
        else:
            return jsonify({'success': False, 'error': 'No pattern configurations available'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


@app.route('/api/upload_paired_files', methods=['POST'])
def upload_paired_files():
    """Upload image and JSON files together, ensuring each image has a corresponding JSON file."""
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
        poses_folder = os.path.join(session_folder, 'poses')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(poses_folder, exist_ok=True)
        
        # Separate image and JSON files
        image_files = []
        json_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                image_files.append(file)
            elif file and allowed_file(file.filename, ALLOWED_POSE_EXTENSIONS):
                json_files.append(file)
        
        # Validate pairing - each image should have a corresponding JSON file
        image_base_names = {os.path.splitext(secure_filename(f.filename))[0] for f in image_files}
        json_base_names = {os.path.splitext(secure_filename(f.filename))[0] for f in json_files}
        
        missing_json = image_base_names - json_base_names
        missing_images = json_base_names - image_base_names
        
        if missing_json:
            return jsonify({
                'error': f'Missing JSON files for images: {", ".join(missing_json)}.jpg (expected: {", ".join(missing_json)}.json)'
            }), 400
        
        if missing_images:
            return jsonify({
                'error': f'Missing image files for JSON: {", ".join(missing_images)}.json (expected: {", ".join(missing_images)}.jpg)'
            }), 400
        
        # Process paired files
        uploaded_images = []
        uploaded_poses = []
        
        for image_file in image_files:
            image_filename = secure_filename(image_file.filename)
            base_name = os.path.splitext(image_filename)[0]
            
            # Find corresponding JSON file
            json_file = next((f for f in json_files if os.path.splitext(secure_filename(f.filename))[0] == base_name), None)
            
            if json_file:
                # Save image
                image_path = os.path.join(images_folder, image_filename)
                image_file.save(image_path)
                
                # Save and validate JSON
                json_filename = secure_filename(json_file.filename)
                json_path = os.path.join(poses_folder, json_filename)
                json_file.save(json_path)
                
                # Validate JSON format
                try:
                    with open(json_path, 'r') as f:
                        pose_data = json.load(f)
                    
                    # Check for required 'end2base' key
                    if 'end2base' not in pose_data:
                        os.remove(image_path)
                        os.remove(json_path)
                        return jsonify({'error': f'JSON file {json_filename} missing required "end2base" key'}), 400
                    
                    uploaded_images.append({
                        'name': image_filename,
                        'path': image_path,
                        'url': url_for('get_image', session_id=session_id, filename=image_filename)
                    })
                    
                    uploaded_poses.append({
                        'name': json_filename,
                        'path': json_path
                    })
                    
                except json.JSONDecodeError:
                    os.remove(image_path)
                    os.remove(json_path)
                    return jsonify({'error': f'Invalid JSON format in {json_filename}'}), 400
        
        # Update session data
        if session_id not in session_data:
            session_data[session_id] = {}
        
        # Append new files to existing ones (if any)
        existing_images = session_data[session_id].get('images', [])
        existing_poses = session_data[session_id].get('poses', [])
        
        all_images = existing_images + uploaded_images
        all_poses = existing_poses + uploaded_poses
        
        session_data[session_id]['images'] = all_images
        session_data[session_id]['poses'] = all_poses
        session_data[session_id]['calibration_type'] = calibration_type
        
        print(f"📁 DEBUG: Upload completed for session {session_id}")
        print(f"📁 DEBUG: Session now has {len(all_images)} images and {len(all_poses)} poses")
        print(f"📁 DEBUG: Session keys after upload: {list(session_data[session_id].keys())}")
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_images)} paired image-pose files. Total: {len(all_images)} images with poses',
            'images': all_images,
            'poses': all_poses
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
        
        # Extract parameters - modern JSON pattern format only
        parameters = {}
        
        # Pattern JSON (modern format)
        if 'pattern_json' in data:
            pattern_json = data['pattern_json']
            # Store the complete JSON pattern for use with unified pattern system
            parameters['pattern_json'] = pattern_json
            
            # Extract individual parameters for compatibility
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
                
            print(f'�?Using modern JSON pattern format: {pattern_json}')
        else:
            return jsonify({'error': 'pattern_json parameter is required'}), 400
        
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
        
        print(f"⚙️ DEBUG: Parameters set for session {session_id}")
        print(f"⚙️ DEBUG: Parameter keys: {list(parameters.keys())}")
        print(f"⚙️ DEBUG: Session keys after parameters: {list(session_data[session_id].keys())}")
        
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
            
            # Debug: Print current session data structure
            print(f"�?DEBUG: Session data keys: {list(session_info.keys())}")
            print(f"�?DEBUG: Has images: {'images' in session_info}")
            print(f"�?DEBUG: Has parameters: {'parameters' in session_info}")
            if 'images' in session_info:
                print(f"�?DEBUG: Number of images: {len(session_info['images'])}")
            if 'parameters' in session_info:
                print(f"�?DEBUG: Parameters keys: {list(session_info['parameters'].keys())}")
            
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
            print(f"🖼�?Selected {len(selected_indices)} images for processing")
            
            # Extract basic parameters
            distortion_model = parameters.get('distortion_model', 'standard')
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
                    # Use the JSON pattern directly (always available since set_parameters requires it)
                    pattern_json = parameters['pattern_json']
                    pattern = create_pattern_from_json(pattern_json)
                    print(f"�?Created pattern from JSON: {pattern_json['pattern_id']}")
                    
                    # Log pattern information
                    pattern_info = pattern.get_info()
                    print(f"📋 Pattern: {pattern.name}")
                    print(f"   Info: {pattern_info}")
                    
                except Exception as e:
                    print(f"�?Error creating calibration pattern: {str(e)}")
                    return jsonify({'error': f'Error creating calibration pattern: {str(e)}'}), 400
                
                print()
                
                # Create new calibrator instance for this session
                calibrator = IntrinsicCalibrator(
                    image_paths=image_paths,
                    calibration_pattern=pattern
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
                
                # Calculate calibration flags based on distortion model
                calibration_flags = 0
                if distortion_model == 'rational':
                    calibration_flags = cv2.CALIB_RATIONAL_MODEL
                elif distortion_model == 'thin_prism':
                    calibration_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
                elif distortion_model == 'tilted':
                    calibration_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
                # 'standard' uses default flags (0)
                
                # Run calibration
                success = calibrator.calibrate(
                    flags=calibration_flags,
                    verbose=True
                )
                
                if success:
                    # Get calibration results 
                    rms_error = calibrator.get_rms_error()
                    
                    # Check RMS error threshold - consider calibration failed if > 0.5
                    if rms_error > 0.5:
                        print(f"❌ Calibration failed - RMS error too high: {rms_error:.4f} pixels (threshold: 0.5)")
                        return jsonify({
                            'error': f'Calibration failed - RMS error too high: {rms_error:.4f} pixels (threshold: 0.5)',
                            'rms_error': rms_error
                        }), 400
                    
                    print(f"✅ Calibration successful - RMS error: {rms_error:.4f} pixels")
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
                    
                    print("�?Saved calibration results to calibration_results.json")
                    print()
                    print("🖼�?GENERATING VISUALIZATION IMAGES")
                    print("-" * 30)
                    
                    # Use the calibrator's built-in visualization methods
                    # Generate corner detection images with aligned index tracking
                    corner_viz_dir = os.path.join(results_folder, 'corner_visualizations')
                    os.makedirs(corner_viz_dir, exist_ok=True)
                    
                    # Create separate original images directory
                    original_dir = os.path.join(results_folder, 'original')
                    os.makedirs(original_dir, exist_ok=True)
                    
                    pattern_images = calibrator.draw_pattern_on_images()
                    
                    # Use the calibrator's filename manager for systematic duplicate handling
                    filename_to_index = calibrator.filename_manager.get_mapping_dict() if calibrator.filename_manager else {}
                    
                    # Initialize corner_images with None placeholders for proper alignment
                    corner_images = [None] * len(calibrator.image_paths)
                    
                    # Save corner detection images 
                    for filename, debug_img in pattern_images:
                        # Use original filename without extra suffixes, with .jpg extension
                        corner_filename = f"{filename}.jpg"
                        corner_path = os.path.join(corner_viz_dir, corner_filename)
                        cv2.imwrite(corner_path, debug_img)
                        
                        # Find the original index for data alignment
                        original_index = filename_to_index.get(filename, -1)
                        if original_index >= 0 and original_index < len(calibrator.image_paths):
                            corner_images[original_index] = {
                                'name': corner_filename,
                                'path': corner_path,
                                'url': url_for('get_corner_image', session_id=session_id, filename=corner_filename),
                                'index': original_index,
                                'original_name': filename
                            }
                    
                    # Save original images separately with original filenames (using FilenameManager for duplicates)
                    for i, image_path in enumerate(calibrator.image_paths):
                        if i < len(calibrator.images) and calibrator.images[i] is not None:
                            # Get original filename from path
                            original_basename = os.path.splitext(os.path.basename(image_path))[0]
                            
                            # Use FilenameManager to get the proper filename (handles duplicates)
                            if calibrator.filename_manager:
                                managed_filename = calibrator.filename_manager.get_unique_filename(i)
                                # If it's different from original, use managed name, otherwise use original
                                if managed_filename != original_basename:
                                    original_filename = f"{managed_filename}.jpg"
                                else:
                                    original_filename = f"{original_basename}.jpg"
                            else:
                                original_filename = f"{original_basename}.jpg"
                            
                            original_path = os.path.join(original_dir, original_filename)
                            cv2.imwrite(original_path, calibrator.images[i])
                    
                    # Generate undistorted images with 3D axes with aligned index tracking
                    undistorted_dir = os.path.join(results_folder, 'undistorted')
                    os.makedirs(undistorted_dir, exist_ok=True)
                    
                    axes_images = calibrator.draw_axes_on_undistorted_images()
                    
                    # Initialize undistorted_images with None placeholders for proper alignment
                    undistorted_images = [None] * len(calibrator.image_paths)
                    
                    # Save undistorted images (original images already saved in separate 'original' directory)
                    for filename, debug_img in axes_images:
                        # Use original filename without extra suffixes, with .jpg extension
                        undistorted_filename = f"{filename}.jpg"
                        undistorted_path = os.path.join(undistorted_dir, undistorted_filename)
                        cv2.imwrite(undistorted_path, debug_img)
                        
                        # Find the original index for data alignment
                        original_index = filename_to_index.get(filename, -1)
                        if original_index >= 0 and original_index < len(calibrator.image_paths):
                            undistorted_images[original_index] = {
                                'name': undistorted_filename,
                                'path': undistorted_path,
                                'url': url_for('get_undistorted_image', session_id=session_id, filename=undistorted_filename),
                                'index': original_index,
                                'original_name': filename
                            }
                    
                    # Create results dictionary with aligned data
                    successful_detections = sum(1 for pts in calibrator.image_points if pts is not None)
                    
                    results = {
                        'success': True,
                        'calibration_type': 'intrinsic',
                        'total_images': len(image_paths),
                        'images_used': successful_detections,
                        'corner_images': corner_images,
                        'undistorted_images': undistorted_images,
                        'per_image_errors': calibrator.per_image_errors,  # Aligned array with None for failed detections
                        'camera_matrix': camera_matrix.tolist(),
                        'distortion_coefficients': dist_coeffs.tolist(),
                        'distortion_model': distortion_model,
                        'rms_error': float(rms_error),
                        'message': f'Intrinsic calibration completed successfully using {successful_detections} out of {len(image_paths)} images'
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
                    # Calibration completely failed
                    total_time = time.time() - start_time
                    print("❌ CALIBRATION FAILED")
                    print("-" * 20)
                    print("Possible causes:")
                    print("• Insufficient pattern detections")
                    print("• Poor image quality")
                    print("• Incorrect pattern parameters")
                    print("• Images too similar (lack of variety)")
                    print(f"⏱️  Time elapsed: {total_time:.2f} seconds")
                    print("=" * 60)
                    return jsonify({
                        'success': False,
                        'error': 'Calibration failed - could not compute camera parameters'
                    }), 400
            
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
                        # Use the JSON pattern directly (always available since set_parameters requires it)
                        pattern_json = parameters['pattern_json']
                        pattern = create_pattern_from_json(pattern_json)
                        print(f"✅ Created pattern from JSON: {pattern_json['pattern_id']}")
                        print(f"✅ Pattern parameters: {pattern_json.get('parameters', {})}")
                        print(f"✅ Pattern info: {pattern.get_info()}")
                    except Exception as e:
                        print(f"❌ Error creating calibration pattern: {str(e)}")
                        print(f"❌ Pattern JSON: {parameters.get('pattern_json', 'NOT FOUND')}")
                        return jsonify({'error': f'Error creating calibration pattern: {str(e)}'}), 400
                    
                    intrinsic_cal = IntrinsicCalibrator(
                        image_paths=image_paths,
                        calibration_pattern=pattern
                    )
                    
                    if not intrinsic_cal.detect_pattern_points(verbose=True):
                        print("Pattern detection failed for eye-in-hand calibration")
                        return jsonify({'error': 'Pattern detection failed for eye-in-hand calibration'}), 400
                    
                    success = intrinsic_cal.calibrate(verbose=True)
                    if not success:
                        print("Intrinsic calibration failed")
                        return jsonify({'error': 'Intrinsic calibration failed'}), 500
                    
                    # Get calibration results
                    camera_matrix = intrinsic_cal.get_camera_matrix()
                    dist_coeffs = intrinsic_cal.get_distortion_coefficients()
                    rms_error = intrinsic_cal.get_rms_error()
                    
                    # Ensure distortion coefficients are 1D array
                    if dist_coeffs is not None:
                        dist_coeffs = dist_coeffs.flatten()
                    
                    # Check RMS error threshold - consider calibration failed if > 0.5
                    if rms_error > 0.5:
                        print(f"Intrinsic calibration failed - RMS error too high: {rms_error:.4f} pixels")
                        return jsonify({'error': f'Intrinsic calibration failed - RMS error too high: {rms_error:.4f} pixels (threshold: 0.5)'}), 500
                
                # Create eye-in-hand calibrator instance with correct constructor approach
                # Following the pattern from examples/eye_in_hand_calibration_example.py
                
                # Add required import
                from core.utils import xyz_rpy_to_matrix
                
                # Load images as cv2 image arrays
                session_folder = get_session_folder(session_id)
                images_folder = os.path.join(session_folder, 'images')
                poses_folder = os.path.join(session_folder, 'poses')
                
                try:
                    # Load images as cv2 arrays (not paths)
                    loaded_images = []
                    end2base_matrices = []
                    
                    # Get sorted list of image files
                    import glob
                    image_files = glob.glob(os.path.join(images_folder, "*.jpg"))
                    image_files.extend(glob.glob(os.path.join(images_folder, "*.jpeg")))
                    image_files.extend(glob.glob(os.path.join(images_folder, "*.png")))
                    image_files = sorted(image_files, key=lambda x: int(os.path.split(x)[-1].split('.')[0])
                                       if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
                    
                    pose_json_paths = glob.glob(os.path.join(poses_folder, "*.json"))
                    pose_json_paths = sorted(pose_json_paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0])
                                           if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
                    
                    if len(image_files) != len(pose_json_paths):
                        print(f"❌ ERROR: Number of images ({len(image_files)}) does not match number of pose files ({len(pose_json_paths)})")
                        return jsonify({'error': f'Number of images ({len(image_files)}) does not match number of pose files ({len(pose_json_paths)})'}), 400
                    
                    # Load images as cv2 arrays and poses as matrices (following example pattern)
                    for i, (img_path, pose_path) in enumerate(zip(image_files, pose_json_paths)):
                        # Load image as cv2 array
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"❌ Failed to load image: {img_path}")
                            continue
                            
                        # Load pose and convert to transformation matrix
                        with open(pose_path, 'r', encoding='utf-8') as f:
                            pose_data = json.load(f)
                            
                        # Use pre-computed end2base matrix if available, otherwise convert from xyz_rpy
                        if 'end2base' in pose_data:
                            end2base_matrix = np.array(pose_data['end2base'], dtype=np.float32)
                        elif 'end_xyzrpy' in pose_data:
                            pose_xyzrpy = pose_data['end_xyzrpy']
                            end2base_matrix = xyz_rpy_to_matrix(
                                pose_xyzrpy['x'], pose_xyzrpy['y'], pose_xyzrpy['z'],
                                pose_xyzrpy['rx'], pose_xyzrpy['ry'], pose_xyzrpy['rz']
                            )
                        else:
                            raise ValueError(f"Unknown pose format in {pose_path}")
                            
                        loaded_images.append(image)
                        end2base_matrices.append(end2base_matrix)
                        print(f"✅ Loaded image-pose pair {i+1}: {os.path.basename(img_path)}")

                    print(f"Successfully loaded {len(loaded_images)} image-pose pairs for eye-in-hand calibration")
                    
                    # Initialize EyeInHandCalibrator using constructor approach (like the example)
                    eye_in_hand_calibrator = EyeInHandCalibrator(
                        images=loaded_images,
                        end2base_matrices=end2base_matrices,
                        image_paths=None,  # Set to None as we provide images directly
                        calibration_pattern=pattern,
                        camera_matrix=camera_matrix,
                        distortion_coefficients=dist_coeffs.flatten()  # Flatten to 1D array
                    )
                    
                    print(f"✅ EyeInHandCalibrator initialized with constructor approach")
                    
                    # Perform eye-in-hand calibration (following example pattern)
                    print("🤖 Performing eye-in-hand calibration...")
                    calibration_result = eye_in_hand_calibrator.calibrate(method=None, verbose=True)
                    
                    if calibration_result is not None:
                        rms_error = calibration_result['rms_error']
                        print(f"✅ Eye-in-hand calibration completed successfully!")
                        print(f"   • RMS error: {rms_error:.4f} pixels")
                        print(f"   • Camera-to-end transformation matrix shape: {calibration_result['cam2end_matrix'].shape}")
                        print(f"   • Target-to-base transformation matrix shape: {calibration_result['target2base_matrix'].shape}")
                        
                        # Generate visualization images
                        print("📸 Generating visualization images...")
                        
                        # Set up output directory for debug images
                        results_folder = os.path.join(RESULTS_FOLDER, session_id)
                        viz_base_folder = os.path.join(results_folder, 'visualizations')
                        corners_folder = os.path.join(viz_base_folder, 'corner_detection')
                        axes_folder = os.path.join(viz_base_folder, 'undistorted_axes')
                        reprojection_folder = os.path.join(viz_base_folder, 'reprojection')
                        
                        os.makedirs(corners_folder, exist_ok=True)
                        os.makedirs(axes_folder, exist_ok=True)
                        os.makedirs(reprojection_folder, exist_ok=True)
                        
                        # Generate visualization images
                        corner_detection_paths = []
                        undistorted_paths = []
                        reprojection_paths = []
                        
                        try:
                            # Generate and save corner detection images
                            print("🎨 Generating corner detection images...")
                            corner_detection_images = eye_in_hand_calibrator.draw_pattern_on_images()
                            print(f"📷 Generated {len(corner_detection_images)} corner detection images")
                            for i, (filename, debug_img) in enumerate(corner_detection_images):
                                output_path = os.path.join(corners_folder, f"{filename}.jpg")
                                success = cv2.imwrite(output_path, debug_img)
                                if success:
                                    corner_detection_paths.append(f"visualizations/corner_detection/{filename}.jpg")
                                    print(f"✅ Saved corner detection: {output_path}")
                                else:
                                    print(f"❌ Failed to save corner detection: {output_path}")
                            
                            print(f"✅ Generated {len(corner_detection_paths)} corner detection images")
                            
                            # Generate and save undistorted images with axes
                            print("🎨 Generating undistorted axes images...")
                            undistorted_images = eye_in_hand_calibrator.draw_axes_on_undistorted_images()
                            print(f"📷 Generated {len(undistorted_images)} undistorted images")
                            for i, (filename, debug_img) in enumerate(undistorted_images):
                                output_path = os.path.join(axes_folder, f"{filename}.jpg")
                                success = cv2.imwrite(output_path, debug_img)
                                if success:
                                    undistorted_paths.append(f"visualizations/undistorted_axes/{filename}.jpg")
                                    print(f"✅ Saved undistorted axes: {output_path}")
                                else:
                                    print(f"❌ Failed to save undistorted axes: {output_path}")
                            
                            print(f"✅ Generated {len(undistorted_paths)} undistorted axes images")
                            
                            # Generate and save reprojection images
                            print("🎨 Generating reprojection images...")
                            reprojection_images = eye_in_hand_calibrator.draw_reprojection_on_images()
                            print(f"📷 Generated {len(reprojection_images)} reprojection images")
                            for i, (filename, debug_img) in enumerate(reprojection_images):
                                output_path = os.path.join(reprojection_folder, f"{filename}.jpg")
                                success = cv2.imwrite(output_path, debug_img)
                                if success:
                                    reprojection_paths.append(f"visualizations/reprojection/{filename}.jpg")
                                    print(f"✅ Saved reprojection: {output_path}")
                                else:
                                    print(f"❌ Failed to save reprojection: {output_path}")
                            
                            print(f"✅ Generated {len(reprojection_paths)} reprojection images")
                            
                        except Exception as viz_error:
                            print(f"⚠️ Warning: Visualization generation failed: {viz_error}")
                            # Use empty lists if visualization fails
                            corner_detection_paths = []
                            undistorted_paths = []
                            reprojection_paths = []
                        
                        # Return results in the expected format for the web interface
                        results = {
                            'message': f'Eye-in-hand calibration completed successfully! RMS error: {rms_error:.4f} pixels',
                            'success': True,
                            'calibration_type': 'eye_in_hand',
                            'session_id': session_id,
                            'rms_error': rms_error,
                            'reprojection_error': rms_error,  # Frontend expects this name
                            'handeye_transform': calibration_result['cam2end_matrix'].tolist(),  # Frontend expects this name
                            'cam2end_matrix': calibration_result['cam2end_matrix'].tolist(),
                            'target2base_matrix': calibration_result['target2base_matrix'].tolist(),
                            'camera_matrix': camera_matrix.tolist(),
                            'distortion_coefficients': dist_coeffs.flatten().tolist(),
                            'corner_detection_images': corner_detection_paths,
                            'undistorted_images': undistorted_paths,
                            'reprojection_images': reprojection_paths
                        }
                        
                        # Store results in session data for web interface
                        session_data[session_id]['results'] = results
                        
                        return jsonify(results)
                    
                    else:
                        print("❌ Eye-in-hand calibration failed")
                        return jsonify({'error': 'Eye-in-hand calibration failed'}), 500
                
                except Exception as e:
                    print(f"❌ Eye-in-hand calibration failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Eye-in-hand calibration failed: {str(e)}'}), 500
            
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
            print(f"🔍 Looking for {len(results['corner_detection_images'])} corner detection images")
            for img_path in results['corner_detection_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                print(f"   Checking: {full_path}")
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Corner Detection - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'corner_detection'
                    })
                    print(f"   ✅ Added corner detection image: {os.path.basename(img_path)}")
                else:
                    print(f"   ❌ Failed to load corner detection image: {full_path}")
        
        # Add undistorted images with axes
        if 'undistorted_images' in results:
            print(f"🔍 Looking for {len(results['undistorted_images'])} undistorted images")
            for img_path in results['undistorted_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                print(f"   Checking: {full_path}")
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Undistorted Axes - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'undistorted_axes'
                    })
                    print(f"   ✅ Added undistorted image: {os.path.basename(img_path)}")
                else:
                    print(f"   ❌ Failed to load undistorted image: {full_path}")
        
        # Add reprojection images
        if 'reprojection_images' in results:
            print(f"🔍 Looking for {len(results['reprojection_images'])} reprojection images")
            for img_path in results['reprojection_images']:
                full_path = os.path.join(RESULTS_FOLDER, session_id, img_path)
                print(f"   Checking: {full_path}")
                img_base64 = image_to_base64(full_path)
                if img_base64:
                    visualization_images.append({
                        'name': f"Reprojection - {os.path.basename(img_path)}",
                        'data': img_base64,
                        'type': 'reprojection'
                    })
                    print(f"   ✅ Added reprojection image: {os.path.basename(img_path)}")
                else:
                    print(f"   ❌ Failed to load reprojection image: {full_path}")
        
        print(f"📊 Final visualization images count: {len(visualization_images)}")
        for i, img in enumerate(visualization_images):
            print(f"   {i+1}. {img['name']} ({img['type']})")
        
        results['visualization_images'] = visualization_images
        print(f"�?Added {len(visualization_images)} visualization images to results")
        
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


@app.route('/api/original_image/<session_id>/<filename>')
def get_original_image(session_id, filename):
    """Serve original images from the original directory."""
    try:
        results_folder = os.path.join(RESULTS_FOLDER, session_id)
        image_path = os.path.join(results_folder, 'original', filename)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Original image not found'}), 404
        
        return send_file(image_path)
        
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


@app.route('/api/pattern_image', methods=['GET', 'POST'])
def get_pattern_image():
    """Generate and serve calibration pattern images."""
    try:
        # Get pattern JSON from different sources based on request method
        if request.method == 'POST':
            if request.is_json:
                # Direct JSON body (from JavaScript)
                pattern_json = request.get_json()
            else:
                # Form data
                pattern_json_str = request.form.get('pattern_json')
                if not pattern_json_str:
                    return jsonify({'error': 'pattern_json parameter is required'}), 400
                try:
                    pattern_json = json.loads(pattern_json_str)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON format for pattern_json parameter'}), 400
        else:
            # GET request with query string
            pattern_json_str = request.args.get('pattern_json')
            if not pattern_json_str:
                return jsonify({'error': 'pattern_json parameter is required'}), 400
            try:
                pattern_json = json.loads(pattern_json_str)
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid JSON format for pattern_json parameter'}), 400
        
        if not pattern_json:
            return jsonify({'error': 'pattern_json parameter is required'}), 400
        
        # Pattern generation parameters
        pixel_per_square = int(request.args.get('pixel_per_square', 100))
        border_pixels = int(request.args.get('border_pixels', 0))
        
        print(f"API: Pattern generation parameters:")
        print(f"  - pixel_per_square: {pixel_per_square}")
        print(f"  - border_pixels: {border_pixels}")
        print(f"API: Creating pattern from JSON: {pattern_json}")
        
        # Create pattern instance using unified JSON system
        try:
            pattern = create_pattern_from_json(pattern_json)
            print(f"�?Created pattern: {pattern.name}")
        except Exception as e:
            return jsonify({'error': f'Failed to create pattern: {str(e)}'}), 400
        
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
