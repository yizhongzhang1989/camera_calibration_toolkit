"""
Eye-in-Hand Calibration Module
==============================

This module handles eye-in-hand calibration for robot-mounted cameras.
It calibrates the transformation between the camera and robot end-effector.
"""

import os
import json
import numpy as np
import cv2
import nlopt
from typing import Tuple, List, Optional, Dict, Any
from glob import glob

from .utils import (
    get_objpoints,
    get_chessboard_corners,
    calculate_single_image_reprojection_error,
    xyz_rpy_to_matrix,
    matrix_to_xyz_rpy,
    inverse_transform_matrix,
    load_images_from_directory
)


class EyeInHandCalibrator:
    """
    Eye-in-hand calibration class for robot-mounted cameras.
    
    This class calibrates the transformation between a camera mounted on a robot
    end-effector and the end-effector coordinate frame.
    """
    
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.cam2end_matrix = None
        self.target2base_matrix = None
        self.calibration_completed = False
        self.optimization_completed = False
        
    def load_camera_intrinsics(self, camera_matrix: np.ndarray, 
                              distortion_coefficients: np.ndarray) -> None:
        """
        Load camera intrinsic parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coefficients: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
    
    def load_camera_intrinsics_from_file(self, intrinsics_directory: str) -> None:
        """
        Load camera intrinsic parameters from files.
        
        Args:
            intrinsics_directory: Directory containing mtx.json and dist.json files
        """
        camera_matrix_path = os.path.join(intrinsics_directory, 'mtx.json')
        distortion_path = os.path.join(intrinsics_directory, 'dist.json')
        
        if not os.path.exists(camera_matrix_path):
            raise FileNotFoundError(f"Camera matrix file not found: {camera_matrix_path}")
        
        if not os.path.exists(distortion_path):
            raise FileNotFoundError(f"Distortion coefficients file not found: {distortion_path}")
        
        # Load camera matrix
        with open(camera_matrix_path, 'r', encoding='utf-8') as f:
            mtx_dict = json.load(f)
            self.camera_matrix = np.array(mtx_dict["camera_matrix"])
        
        # Load distortion coefficients
        with open(distortion_path, 'r', encoding='utf-8') as f:
            dist_dict = json.load(f)
            self.distortion_coefficients = np.array(dist_dict["distortion_coefficients"])
    
    def load_calibration_data(self, data_directory: str, 
                            selected_indices: Optional[List[int]] = None) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
        """
        Load calibration images and corresponding robot poses.
        
        Args:
            data_directory: Directory containing images and pose JSON files
            selected_indices: Optional list of indices to select specific images
            
        Returns:
            Tuple of (image_paths, base2end_matrices, end2base_matrices)
        """
        # Load images
        image_paths = load_images_from_directory(data_directory)
        
        # Load poses
        pose_json_paths = glob(os.path.join(data_directory, "*.json"))
        pose_json_paths = sorted(pose_json_paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0])
                               if os.path.split(x)[-1].split('.')[0].isdigit() else 0)
        
        if selected_indices is not None:
            image_paths = [image_paths[i] for i in selected_indices if i < len(image_paths)]
            pose_json_paths = [pose_json_paths[i] for i in selected_indices if i < len(pose_json_paths)]
        
        if len(image_paths) != len(pose_json_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) does not match number of pose files ({len(pose_json_paths)})")
        
        # Load end-effector poses
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
        
        print(f"Successfully loaded {len(image_paths)} calibration images and poses")
        return image_paths, base2end_matrices, end2base_matrices
    
    def calibrate(self, image_paths: List[str], end2base_matrices: List[np.ndarray], 
                 XX: int, YY: int, L: float, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Perform eye-in-hand calibration.
        
        Args:
            image_paths: List of calibration image paths
            end2base_matrices: List of end-effector to base transformation matrices
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (cam2end_R, cam2end_t, cam2end_4x4, rvecs_target2cam, tvecs_target2cam)
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            raise ValueError("Camera intrinsic parameters have not been loaded")
        
        # Calculate target to camera transformations from images
        rvecs_target2cam, tvecs_target2cam = self._calibrate_extrinsic_parameters(
            image_paths, XX, YY, L, verbose)
        
        # Prepare data for hand-eye calibration
        end2base_Rs = np.array([matrix[:3, :3] for matrix in end2base_matrices])
        end2base_ts = np.array([matrix[:3, 3] for matrix in end2base_matrices])
        
        # Perform eye-in-hand calibration using OpenCV
        cam2end_R, cam2end_t = cv2.calibrateHandEye(
            end2base_Rs, end2base_ts, rvecs_target2cam, tvecs_target2cam, 
            cv2.CALIB_HAND_EYE_HORAUD)
        
        # Create 4x4 transformation matrix
        cam2end_4x4 = np.eye(4)
        cam2end_4x4[:3, :3] = cam2end_R
        cam2end_4x4[:3, 3] = cam2end_t[:, 0]
        
        self.cam2end_matrix = cam2end_4x4
        self.calibration_completed = True
        
        if verbose:
            print("Eye-in-hand calibration completed successfully.")
            print(f"Camera to end-effector transformation matrix:\n{cam2end_4x4}")

        return cam2end_R, cam2end_t, cam2end_4x4, rvecs_target2cam, tvecs_target2cam
    
    def _calibrate_extrinsic_parameters(self, image_paths: List[str], XX: int, YY: int, L: float,
                                       verbose: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculate extrinsic parameters (target to camera transformation) for each image.
        
        Args:
            image_paths: List of calibration image paths
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (rotation_vectors, translation_vectors)
        """
        # Generate 3D object points
        objpoints = get_objpoints(len(image_paths), XX, YY, L)
        
        # Find chessboard corners
        imgpoints = get_chessboard_corners(image_paths, XX, YY)
        
        if len(imgpoints) != len(image_paths):
            print(f"Warning: Only {len(imgpoints)}/{len(image_paths)} images had detectable corners")
        
        # Calculate extrinsic parameters for each valid image
        rvecs = []
        tvecs = []
        
        for i in range(len(imgpoints)):
            ret, rvec, tvec = cv2.solvePnP(objpoints[i], imgpoints[i], 
                                          self.camera_matrix, self.distortion_coefficients)
            if ret:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                print(f"Could not solve PnP for image {i}")
        
        return rvecs, tvecs
    
    def calculate_reprojection_errors(self, image_paths: List[str], 
                                    base2end_matrices: List[np.ndarray],
                                    end2base_matrices: List[np.ndarray],
                                    rvecs_target2cam: List[np.ndarray],
                                    tvecs_target2cam: List[np.ndarray],
                                    XX: int, YY: int, L: float,
                                    vis: bool = False, 
                                    save_dir: Optional[str] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Calculate reprojection errors using eye-in-hand calibration results.
        
        Args:
            image_paths: List of calibration image paths
            base2end_matrices: List of base to end-effector transformation matrices
            end2base_matrices: List of end-effector to base transformation matrices
            rvecs_target2cam: Rotation vectors from target to camera
            tvecs_target2cam: Translation vectors from target to camera
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            vis: Whether to generate visualization images
            save_dir: Directory to save visualization images
            
        Returns:
            Tuple of (reprojection_errors, target2base_matrices)
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        end2cam_4x4 = np.linalg.inv(self.cam2end_matrix)
        errors = []
        target2base_matrices = []
        
        print("Calculating reprojection errors...")
        
        for i in range(len(end2base_matrices)):
            # Calculate target to camera from image detection
            target2cam_4x4 = xyz_rpy_to_matrix(
                tvecs_target2cam[i].flatten().tolist() + rvecs_target2cam[i].flatten().tolist())
            
            # Calculate target to base transformation
            target2base_4x4 = end2base_matrices[i] @ self.cam2end_matrix @ target2cam_4x4
            target2base_matrices.append(target2base_4x4)
            
            # Calculate target to camera using eye-in-hand calibration
            eyeinhand_target2cam_4x4 = end2cam_4x4 @ base2end_matrices[i] @ target2base_4x4
            
            # Calculate reprojection error
            error = calculate_single_image_reprojection_error(
                image_paths[i], 
                eyeinhand_target2cam_4x4[:3, :3], 
                eyeinhand_target2cam_4x4[:3, 3], 
                self.camera_matrix, self.distortion_coefficients, 
                XX, YY, L)
            errors.append(error)
            
            # Generate visualization if requested
            if vis and save_dir:
                self._generate_reprojection_visualization(
                    image_paths[i], eyeinhand_target2cam_4x4, XX, YY, L, save_dir, 
                    suffix="eyeinhand")
        
        return np.array(errors), target2base_matrices
    
    def _generate_reprojection_visualization(self, image_path: str, target2cam_4x4: np.ndarray,
                                           XX: int, YY: int, L: float, save_dir: str,
                                           suffix: str = "") -> None:
        """
        Generate reprojection visualization for a single image.
        
        Args:
            image_path: Path to the image
            target2cam_4x4: Target to camera transformation matrix
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            save_dir: Directory to save the visualization
            suffix: Suffix for the output filename
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Generate 3D object points
        objpoints = get_objpoints(1, XX, YY, L)
        
        # Project 3D points to image
        reprojected_imgpoints, _ = cv2.projectPoints(
            objpoints[0], target2cam_4x4[:3, :3], target2cam_4x4[:3, 3], 
            self.camera_matrix, self.distortion_coefficients)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        img_draw = img.copy()
        
        # Detect actual corners
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, detected_corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            detected_corners = cv2.cornerSubPix(gray, detected_corners, (11, 11), (-1, -1), criteria)
            
            # Draw actual corners in green
            for corner in detected_corners:
                cv2.circle(img_draw, tuple(corner[0].astype(int)), 8, (0, 255, 0), 2)
            
            # Draw reprojected corners in red
            for point in reprojected_imgpoints:
                center = tuple(point[0].astype(int))
                cv2.drawMarker(img_draw, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            
            # Add legend
            cv2.putText(img_draw, "Green: Detected corners", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_draw, "Red: Reprojected corners", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save visualization
        image_name = os.path.basename(image_path).split('.')[0]
        output_name = f'{suffix}_reproject_{image_name}.png' if suffix else f'reproject_{image_name}.png'
        output_path = os.path.join(save_dir, output_name)
        cv2.imwrite(output_path, img_draw)
    
    def optimize_calibration(self, image_paths: List[str], 
                           rvecs_target2cam: List[np.ndarray],
                           tvecs_target2cam: List[np.ndarray],
                           end2base_matrices: List[np.ndarray],
                           base2end_matrices: List[np.ndarray],
                           XX: int, YY: int, L: float,
                           iterations: int = 5,
                           ftol_rel: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize eye-in-hand calibration using nonlinear optimization.
        
        Args:
            image_paths: List of calibration image paths
            rvecs_target2cam: Rotation vectors from target to camera
            tvecs_target2cam: Translation vectors from target to camera
            end2base_matrices: List of end-effector to base transformation matrices
            base2end_matrices: List of base to end-effector transformation matrices
            XX: Number of chessboard corners along x-axis
            YY: Number of chessboard corners along y-axis
            L: Size of chessboard squares in meters
            iterations: Number of optimization iterations
            ftol_rel: Relative tolerance for optimization
            
        Returns:
            Tuple of (optimized_cam2end_matrix, optimized_target2base_matrix)
        """
        if not self.calibration_completed:
            raise ValueError("Initial calibration must be completed before optimization")
        
        # Find the image with minimum reprojection error as starting point
        errors, target2base_matrices = self.calculate_reprojection_errors(
            image_paths, base2end_matrices, end2base_matrices, 
            rvecs_target2cam, tvecs_target2cam, XX, YY, L)
        
        min_error_idx = np.argmin(errors)
        print(f"Using image {min_error_idx} with minimum error {errors[min_error_idx]:.4f} as optimization starting point")
        
        # Initial target2base matrix
        current_target2base_4x4 = target2base_matrices[min_error_idx].copy()
        current_cam2end_4x4 = self.cam2end_matrix.copy()
        
        # Prepare data for optimization
        objpoints = get_objpoints(len(image_paths), XX, YY, L)
        imgpoints = get_chessboard_corners(image_paths, XX, YY)
        
        # Iterative optimization
        for iteration in range(iterations):
            print(f"Optimization iteration {iteration + 1}/{iterations}")
            
            # Optimize target2base
            current_target2base_4x4 = self._optimize_target2base(
                min_error_idx, rvecs_target2cam, tvecs_target2cam, 
                end2base_matrices, current_cam2end_4x4, objpoints, imgpoints, ftol_rel)
            
            # Optimize cam2end
            current_cam2end_4x4 = self._optimize_cam2end(
                min_error_idx, rvecs_target2cam, tvecs_target2cam, 
                end2base_matrices, current_target2base_4x4, objpoints, imgpoints, ftol_rel)
        
        self.cam2end_matrix = current_cam2end_4x4
        self.target2base_matrix = current_target2base_4x4
        self.optimization_completed = True
        
        print("Optimization completed successfully")
        return current_cam2end_4x4, current_target2base_4x4
    
    def _optimize_target2base(self, init_image_idx: int, rvecs_target2cam: List[np.ndarray],
                            tvecs_target2cam: List[np.ndarray], end2base_matrices: List[np.ndarray],
                            cam2end_4x4: np.ndarray, objpoints: List[np.ndarray], 
                            imgpoints: List[np.ndarray], ftol_rel: float) -> np.ndarray:
        """Optimize target2base matrix using NLopt."""
        # Extract initial values from the target2base matrix
        x_target2cam = tvecs_target2cam[init_image_idx][0].item()
        y_target2cam = tvecs_target2cam[init_image_idx][1].item()
        z_target2cam = tvecs_target2cam[init_image_idx][2].item()
        roll_target2cam = rvecs_target2cam[init_image_idx][0].item()
        pitch_target2cam = rvecs_target2cam[init_image_idx][1].item()
        yaw_target2cam = rvecs_target2cam[init_image_idx][2].item()
        
        target2cam_4x4 = xyz_rpy_to_matrix([x_target2cam, y_target2cam, z_target2cam, 
                                          roll_target2cam, pitch_target2cam, yaw_target2cam])
        init_target2base_4x4 = end2base_matrices[init_image_idx] @ cam2end_4x4 @ target2cam_4x4
        
        x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(init_target2base_4x4)
        
        # Setup optimization
        opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
        
        def objective(x_opt, grad):
            x, y, z, roll, pitch, yaw = x_opt
            target2cam_list = self._calculate_target2cam_list_from_target2base(
                x, y, z, roll, pitch, yaw, cam2end_4x4, end2base_matrices)
            
            total_error = 0
            for i, target2cam_4x4 in enumerate(target2cam_list):
                if i < len(imgpoints):
                    proj_points, _ = cv2.projectPoints(
                        objpoints[i], target2cam_4x4[:3, :3], target2cam_4x4[:3, 3],
                        self.camera_matrix, self.distortion_coefficients)
                    error = cv2.norm(imgpoints[i], proj_points, cv2.NORM_L2) / len(proj_points)
                    total_error += error
            
            return total_error / len(target2cam_list)
        
        opt.set_min_objective(objective)
        opt.set_ftol_rel(ftol_rel)
        
        # Optimize
        opt_x = opt.optimize([x, y, z, roll, pitch, yaw])
        
        # Return optimized target2base matrix
        optimized_target2base_4x4 = xyz_rpy_to_matrix(opt_x)
        return optimized_target2base_4x4
    
    def _optimize_cam2end(self, init_image_idx: int, rvecs_target2cam: List[np.ndarray],
                        tvecs_target2cam: List[np.ndarray], end2base_matrices: List[np.ndarray],
                        target2base_4x4: np.ndarray, objpoints: List[np.ndarray], 
                        imgpoints: List[np.ndarray], ftol_rel: float) -> np.ndarray:
        """Optimize cam2end matrix using NLopt."""
        x, y, z, roll, pitch, yaw = matrix_to_xyz_rpy(self.cam2end_matrix)
        
        # Setup optimization
        opt = nlopt.opt(nlopt.LN_NELDERMEAD, 6)
        
        def objective(x_opt, grad):
            x, y, z, roll, pitch, yaw = x_opt
            cam2end_4x4 = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
            target2cam_list = self._calculate_target2cam_list_from_cam2end(
                x, y, z, roll, pitch, yaw, target2base_4x4, end2base_matrices)
            
            total_error = 0
            for i, target2cam_4x4 in enumerate(target2cam_list):
                if i < len(imgpoints):
                    proj_points, _ = cv2.projectPoints(
                        objpoints[i], target2cam_4x4[:3, :3], target2cam_4x4[:3, 3],
                        self.camera_matrix, self.distortion_coefficients)
                    error = cv2.norm(imgpoints[i], proj_points, cv2.NORM_L2) / len(proj_points)
                    total_error += error
            
            return total_error / len(target2cam_list)
        
        opt.set_min_objective(objective)
        opt.set_ftol_rel(ftol_rel)
        
        # Optimize
        opt_x = opt.optimize([x, y, z, roll, pitch, yaw])
        
        # Return optimized cam2end matrix
        optimized_cam2end_4x4 = xyz_rpy_to_matrix(opt_x)
        return optimized_cam2end_4x4
    
    def _calculate_target2cam_list_from_target2base(self, x: float, y: float, z: float,
                                                  roll: float, pitch: float, yaw: float,
                                                  cam2end_4x4: np.ndarray, 
                                                  end2base_matrices: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate target2cam matrices from target2base parameters."""
        target2base_4x4 = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
        end2cam_4x4 = np.linalg.inv(cam2end_4x4)
        
        target2cam_list = []
        for end2base_matrix in end2base_matrices:
            base2end_4x4 = np.linalg.inv(end2base_matrix)
            target2cam_4x4 = end2cam_4x4 @ base2end_4x4 @ target2base_4x4
            target2cam_list.append(target2cam_4x4)
        
        return target2cam_list
    
    def _calculate_target2cam_list_from_cam2end(self, x: float, y: float, z: float,
                                              roll: float, pitch: float, yaw: float,
                                              target2base_4x4: np.ndarray,
                                              end2base_matrices: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate target2cam matrices from cam2end parameters."""
        cam2end_4x4 = xyz_rpy_to_matrix([x, y, z, roll, pitch, yaw])
        end2cam_4x4 = np.linalg.inv(cam2end_4x4)
        
        target2cam_list = []
        for end2base_matrix in end2base_matrices:
            base2end_4x4 = np.linalg.inv(end2base_matrix)
            target2cam_4x4 = end2cam_4x4 @ base2end_4x4 @ target2base_4x4
            target2cam_list.append(target2cam_4x4)
        
        return target2cam_list
    
    def save_results(self, save_directory: str) -> None:
        """
        Save calibration results to JSON file.
        
        Args:
            save_directory: Directory to save the results
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        results = {
            "camera_intrinsics": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist()
            },
            "eye_in_hand_calibration": {
                "cam2end_matrix": self.cam2end_matrix.tolist(),
            }
        }
        
        if self.optimization_completed and self.target2base_matrix is not None:
            results["eye_in_hand_calibration"]["target2base_matrix"] = self.target2base_matrix.tolist()
        
        json_file_path = os.path.join(save_directory, "eye_in_hand_calibration_results.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Eye-in-hand calibration results saved to: {json_file_path}")
    
    def load_results(self, results_file: str) -> None:
        """
        Load calibration results from JSON file.
        
        Args:
            results_file: Path to the results JSON file
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Load camera intrinsics
        if "camera_intrinsics" in results:
            self.camera_matrix = np.array(results["camera_intrinsics"]["camera_matrix"])
            self.distortion_coefficients = np.array(results["camera_intrinsics"]["distortion_coefficients"])
        
        # Load eye-in-hand calibration results
        if "eye_in_hand_calibration" in results:
            self.cam2end_matrix = np.array(results["eye_in_hand_calibration"]["cam2end_matrix"])
            self.calibration_completed = True
            
            if "target2base_matrix" in results["eye_in_hand_calibration"]:
                self.target2base_matrix = np.array(results["eye_in_hand_calibration"]["target2base_matrix"])
                self.optimization_completed = True
        
        print(f"Eye-in-hand calibration results loaded from: {results_file}")
    
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Get the camera to end-effector transformation matrix.
        
        Returns:
            4x4 transformation matrix from camera to end-effector
        """
        if not self.calibration_completed:
            raise ValueError("Calibration has not been completed yet")
        
        return self.cam2end_matrix
