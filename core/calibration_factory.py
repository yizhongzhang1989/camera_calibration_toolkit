"""
Calibration Factory Module
==========================

This module provides a factory pattern for creating different types of calibrators.
It simplifies calibrator instantiation and provides a consistent interface for
creating various calibration types.

The factory supports current and future calibration types:
- intrinsic: Single camera intrinsic parameter calibration
- eye_in_hand: Eye-in-hand (robot-mounted camera) calibration
- stereo: Stereo camera calibration (future)
- hand_eye: Hand-eye calibration (future)
- multi_camera: Multi-camera network calibration (future)
"""

from typing import Dict, Any, Optional
from .intrinsic_calibration import IntrinsicCalibrator
from .eye_in_hand_calibration import EyeInHandCalibrator


class CalibrationFactory:
    """
    Factory class for creating different types of calibrators.
    
    This factory provides a unified interface for creating calibrators
    while hiding the complexity of their initialization parameters.
    """
    
    # Registry of available calibration types
    _calibrator_registry = {
        'intrinsic': IntrinsicCalibrator,
        'eye_in_hand': EyeInHandCalibrator,
        # Future calibration types will be added here:
        # 'stereo': StereoCalibrator,
        # 'hand_eye': HandEyeCalibrator,
        # 'multi_camera': MultiCameraCalibrator,
        # 'lidar_camera': LidarCameraCalibrator,
    }
    
    @classmethod
    def create_calibrator(cls, calibration_type: str, **kwargs):
        """
        Create a calibrator instance of the specified type.
        
        Args:
            calibration_type: Type of calibrator to create
                - 'intrinsic': Single camera intrinsic calibration
                - 'eye_in_hand': Eye-in-hand robot camera calibration
            **kwargs: Arguments passed to the calibrator constructor
            
        Returns:
            BaseCalibrator: Instance of the requested calibrator type
            
        Raises:
            ValueError: If calibration_type is not supported
            
        Examples:
            # Create intrinsic calibrator
            calibrator = CalibrationFactory.create_calibrator(
                'intrinsic',
                image_paths=['img1.jpg', 'img2.jpg'],
                calibration_pattern=pattern
            )
            
            # Create eye-in-hand calibrator
            calibrator = CalibrationFactory.create_calibrator(
                'eye_in_hand',
                image_paths=image_paths,
                robot_poses=poses,
                camera_matrix=K,
                distortion_coefficients=D,
                calibration_pattern=pattern
            )
        """
        if calibration_type not in cls._calibrator_registry:
            available_types = ', '.join(cls._calibrator_registry.keys())
            raise ValueError(
                f"Unknown calibration type: '{calibration_type}'. "
                f"Available types: {available_types}"
            )
        
        calibrator_class = cls._calibrator_registry[calibration_type]
        return calibrator_class(**kwargs)
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available calibration types.
        
        Returns:
            list: List of supported calibration type strings
        """
        return list(cls._calibrator_registry.keys())
    
    @classmethod
    def get_calibrator_info(cls, calibration_type: str) -> Dict[str, Any]:
        """
        Get information about a specific calibrator type.
        
        Args:
            calibration_type: Type of calibrator to get info for
            
        Returns:
            dict: Information about the calibrator type
            
        Raises:
            ValueError: If calibration_type is not supported
        """
        if calibration_type not in cls._calibrator_registry:
            raise ValueError(f"Unknown calibration type: '{calibration_type}'")
        
        calibrator_class = cls._calibrator_registry[calibration_type]
        
        return {
            'type': calibration_type,
            'class_name': calibrator_class.__name__,
            'module': calibrator_class.__module__,
            'docstring': calibrator_class.__doc__ or "No description available",
        }
    
    @classmethod
    def register_calibrator(cls, calibration_type: str, calibrator_class):
        """
        Register a new calibrator type with the factory.
        
        This method allows adding custom calibrator types to the factory.
        
        Args:
            calibration_type: String identifier for the calibrator type
            calibrator_class: Calibrator class that inherits from BaseCalibrator
            
        Example:
            CalibrationFactory.register_calibrator('custom', CustomCalibrator)
        """
        cls._calibrator_registry[calibration_type] = calibrator_class
    
    @classmethod
    def unregister_calibrator(cls, calibration_type: str):
        """
        Remove a calibrator type from the factory registry.
        
        Args:
            calibration_type: String identifier for the calibrator type
            
        Raises:
            ValueError: If calibration_type is not registered
        """
        if calibration_type not in cls._calibrator_registry:
            raise ValueError(f"Calibration type '{calibration_type}' is not registered")
        
        del cls._calibrator_registry[calibration_type]


# Convenience function for direct calibrator creation
def create_calibrator(calibration_type: str, **kwargs):
    """
    Convenience function to create a calibrator instance.
    
    This is a shorthand for CalibrationFactory.create_calibrator().
    
    Args:
        calibration_type: Type of calibrator to create
        **kwargs: Arguments passed to the calibrator constructor
        
    Returns:
        BaseCalibrator: Instance of the requested calibrator type
    """
    return CalibrationFactory.create_calibrator(calibration_type, **kwargs)
