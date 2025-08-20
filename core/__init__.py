"""
Camera Calibration Toolkit - Core Module
========================================

This module provides core functionality for camera calibration including:
- Single camera intrinsic calibration
- Multiple camera stereo calibration  
- Eye-in-hand calibration for robot-mounted cameras
- Eye-to-hand calibration for external cameras observing robots

The core module is designed to be used independently of the web interface,
allowing easy integration into other projects as a submodule.
"""

from .intrinsic_calibration import IntrinsicCalibrator
from .eye_in_hand_calibration import EyeInHandCalibrator
from .utils import (
    get_objpoints,
    rpy_to_matrix,
    xyz_rpy_to_matrix,
    matrix_to_xyz_rpy,
    inverse_transform_matrix
)

__all__ = [
    'IntrinsicCalibrator',
    'EyeInHandCalibrator',
    'get_objpoints',
    'rpy_to_matrix',
    'xyz_rpy_to_matrix',
    'matrix_to_xyz_rpy',
    'inverse_transform_matrix'
]
