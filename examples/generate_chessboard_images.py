#!/usr/bin/env python3
"""
Calibration Pattern Image Generator
===================================

This script generates specific calibration pattern images (chessboards, ChArUco, Grid Boards)
for printing and calibration purposes. Images are saved to data/results/chessboard_images/

Generated Patterns:
- Standard chessboard 11×8 (default settings)
- Standard chessboard 9×6 (200px squares, 200px border)
- ChArUco 8×6 (DICT_4X4_100, 20mm squares, 15mm markers)
- ChArUco 12×9 (DICT_6X6_250, 20mm squares, 10mm markers, 50px border)
- ArUco Grid Board 1×1 (DICT_4X4_50, 40mm markers, 10mm separation)
- ArUco Grid Board 5×4 (DICT_4X4_50, 40mm markers, 10mm separation)

Usage:
    conda activate camcalib
    python examples/generate_chessboard_images.py

Output:
    Images saved to data/results/chessboard_images/
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.calibration_patterns import load_pattern_from_json


def ensure_output_directory():
    """Ensure the output directory exists."""
    output_dir = os.path.join('data', 'results', 'chessboard_images')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_patterns():
    """Generate the specific requested patterns."""
    print("🎨 Calibration Pattern Image Generator")
    print("=" * 45)
    
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = ensure_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize pattern system once to avoid repeated discovery messages
    print("\n🔍 Initializing pattern system...")
    from core.calibration_patterns import load_pattern_from_json
    _ = load_pattern_from_json({
        "pattern_id": "standard_chessboard",
        "name": "Test",
        "description": "Test",
        "is_planar": True,
        "parameters": {"width": 8, "height": 6, "square_size": 0.02}
    })
    print("✅ Pattern system ready")
    print()
    
    # Define the specific patterns to generate
    patterns = [
        {
            'name': 'chessboard_11x8_default',
            'description': 'Standard 11×8 chessboard with default settings',
            'generate_params': {},  # Use defaults: 100px squares, 0px border
            'pattern_config': {
                "pattern_id": "standard_chessboard",
                "name": "Standard Chessboard",
                "description": "Traditional black and white checkerboard pattern",
                "is_planar": True,
                "parameters": {
                    "width": 11,
                    "height": 8,
                    "square_size": 0.025
                }
            }
        },
        {
            'name': 'chessboard_9x6_200px_border200',
            'description': 'Standard 9×6 chessboard with 200px squares and 200px border',
            'generate_params': {'pixel_per_square': 200, 'border_pixels': 200},
            'pattern_config': {
                "pattern_id": "standard_chessboard",
                "name": "Standard Chessboard",
                "description": "Traditional black and white checkerboard pattern",
                "is_planar": True,
                "parameters": {
                    "width": 9,
                    "height": 6,
                    "square_size": 0.025
                }
            }
        },
        {
            'name': 'charuco_8x6_4x4_100',
            'description': 'ChArUco 8×6 with DICT_4X4_100, 20mm squares, 15mm markers',
            'generate_params': {},  # Use defaults: 100px squares, 0px border
            'pattern_config': {
                "pattern_id": "charuco_board",
                "name": "ChArUco Board",
                "description": "Chessboard pattern with ArUco markers for robust detection",
                "is_planar": True,
                "parameters": {
                    "width": 8,
                    "height": 6,
                    "square_size": 0.02,
                    "marker_size": 0.015,
                    "dictionary_id": cv2.aruco.DICT_4X4_100
                }
            }
        },
        {
            'name': 'charuco_12x9_6x6_250_border50',
            'description': 'ChArUco 12×9 with DICT_6X6_250, 20mm squares, 10mm markers, 50px border',
            'generate_params': {'border_pixels': 50},  # Use default 100px squares, add 50px border
            'pattern_config': {
                "pattern_id": "charuco_board",
                "name": "ChArUco Board",
                "description": "Chessboard pattern with ArUco markers for robust detection",
                "is_planar": True,
                "parameters": {
                    "width": 12,
                    "height": 9,
                    "square_size": 0.02,
                    "marker_size": 0.01,
                    "dictionary_id": cv2.aruco.DICT_6X6_250
                }
            }
        },
        {
            'name': 'gridboard_1x1_dict10',
            'description': 'ArUco Grid Board 1×1 with DICT_4X4_50, 40mm markers, 10mm separation',
            'generate_params': {'pixel_per_square': 150, 'border_pixels': 100},
            'pattern_config': {
                "pattern_id": "grid_board",
                "name": "Grid Board",
                "description": "ArUco marker grid board pattern",
                "is_planar": True,
                "parameters": {
                    "markers_x": 1,
                    "markers_y": 1,
                    "marker_size": 0.04,
                    "marker_separation": 0.01,
                    "dictionary_id": cv2.aruco.DICT_4X4_50
                }
            }
        },
        {
            'name': 'gridboard_5x4_dict10',
            'description': 'ArUco Grid Board 5×4 with DICT_4X4_50, 40mm markers, 10mm separation',
            'generate_params': {'pixel_per_square': 80, 'border_pixels': 50},
            'pattern_config': {
                "pattern_id": "grid_board",
                "name": "Grid Board",
                "description": "ArUco marker grid board pattern",
                "is_planar": True,
                "parameters": {
                    "markers_x": 5,
                    "markers_y": 4,
                    "marker_size": 0.04,
                    "marker_separation": 0.01,
                    "dictionary_id": cv2.aruco.DICT_4X4_50
                }
            }
        }
    ]
    
    generated_files = []
    failed_patterns = []
    
    for i, config in enumerate(patterns, 1):
        print(f"🔲 Pattern {i}/{len(patterns)}: {config['name']}")
        print(f"   📄 {config['description']}")
        
        try:
            # Create pattern using JSON configuration - load_pattern_from_json handles all types
            pattern = load_pattern_from_json(config['pattern_config'])
            
            # Generate image with specified parameters
            if config['generate_params']:
                image = pattern.generate_pattern_image(**config['generate_params'])
            else:
                image = pattern.generate_pattern_image()
            
            # Save image
            filename = f"{config['name']}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            
            # Verify the image was actually saved
            if not os.path.exists(filepath):
                raise Exception(f"Failed to save image file: {filepath}")
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_kb = file_size / 1024
            
            print(f"   💾 Generated: {filename} ({image.shape[1]}×{image.shape[0]}) - {file_size_kb:.1f} KB")
            
            # Serialize pattern and export JSON
            json_filename = f"{config['name']}.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            try:
                # Use the pattern's built-in to_json method
                pattern_dict = pattern.to_json()
                
                # Write JSON file
                with open(json_filepath, 'w') as f:
                    json.dump(pattern_dict, f, indent=2)
                
                print(f"   📋 Pattern JSON: {json_filename}")
                generated_files.append(json_filename)
                
            except Exception as json_error:
                print(f"   ⚠️  JSON serialization warning: {json_error}")
            
            # Generate info file
            info_filename = f"{config['name']}_info.txt"
            info_filepath = os.path.join(output_dir, info_filename)
            
            pattern_params = config['pattern_config']['parameters']
            pattern_id = config['pattern_config']['pattern_id']
            
            with open(info_filepath, 'w') as f:
                f.write(f"Pattern Information\n")
                f.write(f"==================\n\n")
                f.write(f"Name: {config['name']}\n")
                f.write(f"Description: {config['description']}\n")
                f.write(f"Type: {pattern_id.replace('_', ' ').title()}\n")
                
                if pattern_id == 'grid_board':
                    f.write(f"Grid Size: {pattern_params['markers_x']}×{pattern_params['markers_y']} markers\n")
                    f.write(f"Marker Size: {pattern_params['marker_size']*1000:.1f}mm\n")
                    f.write(f"Marker Separation: {pattern_params['marker_separation']*1000:.1f}mm\n")
                    dict_name = [name for name, val in vars(cv2.aruco).items() 
                               if name.startswith('DICT_') and val == pattern_params['dictionary_id']][0]
                    f.write(f"Dictionary: {dict_name}\n")
                else:
                    f.write(f"Dimensions: {pattern_params['width']}×{pattern_params['height']}\n")
                    f.write(f"Square Size: {pattern_params['square_size']*1000:.1f}mm\n")
                    
                    if pattern_id == 'charuco_board':
                        f.write(f"Marker Size: {pattern_params['marker_size']*1000:.1f}mm\n")
                        dict_name = [name for name, val in vars(cv2.aruco).items() 
                                   if name.startswith('DICT_') and val == pattern_params['dictionary_id']][0]
                        f.write(f"Dictionary: {dict_name}\n")
                
                f.write(f"\nImage Information:\n")
                f.write(f"Size: {image.shape[1]}×{image.shape[0]} pixels\n")
                f.write(f"File Size: {file_size_kb:.1f} KB\n")
                
                # Generation parameters
                params = config['generate_params']
                pixel_per_square = params.get('pixel_per_square', 100)
                border_pixels = params.get('border_pixels', 0)
                f.write(f"Pixel per square: {pixel_per_square}px\n")
                f.write(f"Border pixels: {border_pixels}px\n")
                
                f.write(f"\nGenerated: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            generated_files.extend([filename, json_filename, info_filename])
            
        except Exception as e:
            print(f"   ❌ Error creating {config['name']}: {e}")
            failed_patterns.append(config['name'])
        
        print()
    
    # Summary
    end_time = datetime.now()
    success_count = len(patterns) - len(failed_patterns)
    
    if failed_patterns:
        print(f"⚠️  Generation completed with {len(failed_patterns)} failures")
        print(f"   Failed patterns: {', '.join(failed_patterns)}")
    else:
        print("✅ All patterns generated successfully!")
    
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Generated {len(generated_files)} files ({success_count} patterns)")
    print(f"⏰ Completed in {(end_time - start_time).total_seconds():.1f}s")
    
    print("\n🎯 Usage Tips:")
    print("   • Print at 'Actual Size' for accurate physical dimensions")  
    print("   • Mount patterns on rigid, flat surfaces")
    print("   • Ensure good lighting and avoid reflections")
    print("   • Check .txt files for detailed specifications")
    
    # Return success/failure indication
    if failed_patterns:
        return len(failed_patterns)  # Return number of failures
    else:
        return 0  # Success


def main():
    """Main function with proper error handling."""
    try:
        result = generate_patterns()
        return result  # 0 for success, >0 for number of failures
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Pattern generation failed:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
