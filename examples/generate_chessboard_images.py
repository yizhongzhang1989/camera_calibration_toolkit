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
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.calibration_patterns import create_chessboard_pattern


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
    print()
    
    # Define the specific patterns to generate
    patterns = [
        {
            'type': 'standard',
            'name': 'chessboard_11x8_default',
            'width': 11, 
            'height': 8, 
            'square_size': 0.025,
            'description': 'Standard 11×8 chessboard with default settings',
            'generate_params': {}  # Use defaults: 100px squares, 0px border
        },
        {
            'type': 'standard',
            'name': 'chessboard_9x6_200px_border200',
            'width': 9, 
            'height': 6, 
            'square_size': 0.025,
            'description': 'Standard 9×6 chessboard with 200px squares and 200px border',
            'generate_params': {'pixel_per_square': 200, 'border_pixels': 200}
        },
        {
            'type': 'charuco',
            'name': 'charuco_8x6_4x4_100',
            'width': 8, 
            'height': 6, 
            'square_size': 0.02, 
            'marker_size': 0.015,
            'dictionary_id': cv2.aruco.DICT_4X4_100,
            'description': 'ChArUco 8×6 with DICT_4X4_100, 20mm squares, 15mm markers',
            'generate_params': {}  # Use defaults: 100px squares, 0px border
        },
        {
            'type': 'charuco',
            'name': 'charuco_12x9_6x6_250_border50',
            'width': 12, 
            'height': 9, 
            'square_size': 0.02, 
            'marker_size': 0.01,
            'dictionary_id': cv2.aruco.DICT_6X6_250,
            'description': 'ChArUco 12×9 with DICT_6X6_250, 20mm squares, 10mm markers, 50px border',
            'generate_params': {'border_pixels': 50}  # Use default 100px squares, add 50px border
        },
        {
            'type': 'grid',
            'name': 'gridboard_1x1_dict10',
            'markers_x': 1, 
            'markers_y': 1, 
            'marker_size': 0.04, 
            'marker_separation': 0.01,
            'dictionary_id': cv2.aruco.DICT_4X4_50,
            'description': 'ArUco Grid Board 1×1 with DICT_4X4_50, 40mm markers, 10mm separation',
            'generate_params': {'pixel_per_square': 150, 'border_pixels': 100}
        },
        {
            'type': 'grid',
            'name': 'gridboard_5x4_dict10',
            'markers_x': 5, 
            'markers_y': 4, 
            'marker_size': 0.04, 
            'marker_separation': 0.01,
            'dictionary_id': cv2.aruco.DICT_4X4_50,
            'description': 'ArUco Grid Board 5×4 with DICT_4X4_50, 40mm markers, 10mm separation',
            'generate_params': {'pixel_per_square': 80, 'border_pixels': 50}
        }
    ]
    
    generated_files = []
    
    for i, config in enumerate(patterns, 1):
        print(f"🔲 Pattern {i}/{len(patterns)}: {config['name']}")
        print(f"   📄 {config['description']}")
        
        try:
            # Create pattern
            if config['type'] == 'standard':
                pattern = create_chessboard_pattern(
                    'standard',
                    width=config['width'],
                    height=config['height'],
                    square_size=config['square_size']
                )
            elif config['type'] == 'charuco':
                pattern = create_chessboard_pattern(
                    'charuco',
                    width=config['width'],
                    height=config['height'],
                    square_size=config['square_size'],
                    marker_size=config['marker_size'],
                    dictionary_id=config['dictionary_id']
                )
            elif config['type'] == 'grid':
                # Import Grid Board pattern directly
                from core.calibration_patterns import GridBoard
                pattern = GridBoard(
                    markers_x=config['markers_x'],
                    markers_y=config['markers_y'],
                    marker_size=config['marker_size'],
                    marker_separation=config['marker_separation'],
                    dictionary_id=config['dictionary_id']
                )
            
            # Generate image with specified parameters
            if config['generate_params']:
                image = pattern.generate_pattern_image(**config['generate_params'])
            else:
                image = pattern.generate_pattern_image()
            
            # Save image
            filename = f"{config['name']}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_kb = file_size / 1024
            
            print(f"   💾 Saved: {filename} ({image.shape[1]}×{image.shape[0]}) - {file_size_kb:.1f} KB")
            
            # Generate info file
            info_filename = f"{config['name']}_info.txt"
            info_filepath = os.path.join(output_dir, info_filename)
            
            with open(info_filepath, 'w') as f:
                f.write(f"Pattern Information\n")
                f.write(f"==================\n\n")
                f.write(f"Name: {config['name']}\n")
                f.write(f"Description: {config['description']}\n")
                f.write(f"Type: {config['type'].title()}\n")
                
                if config['type'] == 'grid':
                    f.write(f"Grid Size: {config['markers_x']}×{config['markers_y']} markers\n")
                    f.write(f"Marker Size: {config['marker_size']*1000:.1f}mm\n")
                    f.write(f"Marker Separation: {config['marker_separation']*1000:.1f}mm\n")
                    dict_name = [name for name, val in vars(cv2.aruco).items() 
                               if name.startswith('DICT_') and val == config['dictionary_id']][0]
                    f.write(f"Dictionary: {dict_name}\n")
                else:
                    f.write(f"Dimensions: {config['width']}×{config['height']}\n")
                    f.write(f"Square Size: {config['square_size']*1000:.1f}mm\n")
                    
                    if config['type'] == 'charuco':
                        f.write(f"Marker Size: {config['marker_size']*1000:.1f}mm\n")
                        dict_name = [name for name, val in vars(cv2.aruco).items() 
                                   if name.startswith('DICT_') and val == config['dictionary_id']][0]
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
            
            generated_files.extend([filename, info_filename])
            print(f"   📝 Info saved: {info_filename}")
            
        except Exception as e:
            print(f"   ❌ Error creating {config['name']}: {e}")
        
        print()
    
    # Summary
    end_time = datetime.now()
    print("✅ Pattern generation complete!")
    print(f"📂 All images saved to: {output_dir}")
    print(f"⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if generated_files:
        print(f"\n📋 Generated Files ({len(generated_files)} files):")
        for i, filename in enumerate(generated_files, 1):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                file_type = "🖼️" if filename.endswith('.png') else "📝"
                print(f"    {i:2d}. {file_type} {filename:<45} ({file_size:,} bytes)")
    
    print("\n🎯 Usage Tips:")
    print("   • Use these patterns for camera calibration")
    print("   • Print at 'Actual Size' for accurate physical dimensions")  
    print("   • Mount patterns on rigid, flat surfaces")
    print("   • Ensure good lighting and avoid reflections")
    print("   • Check pattern info files for detailed specifications")


if __name__ == "__main__":
    generate_patterns()
