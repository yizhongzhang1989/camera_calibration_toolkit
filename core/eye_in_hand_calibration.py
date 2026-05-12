"""
Eye-in-Hand Calibration Module
==============================

Thin subclass of :class:`HandEyeBaseCalibrator` for the *camera on the robot
end-effector* configuration. All shared math (method sweep, joint
optimization, reprojection error, JSON IO) lives in the base class; this file
only provides the eye-in-hand specific composition kernels, the domain-specific
attribute names and a few thin compatibility wrappers.

Architecture::

    BaseCalibrator (images / patterns)
    └── HandEyeBaseCalibrator (robot poses / shared algorithm / IO)
        └── EyeInHandCalibrator (eye-in-hand specific kernels)
"""

from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from .hand_eye_base_calibration import HandEyeBaseCalibrator


class EyeInHandCalibrator(HandEyeBaseCalibrator):
    """
    Eye-in-hand hand-eye calibration (camera mounted on the end-effector).

    Primary unknown: ``cam2end_matrix`` (camera → end-effector).
    Secondary unknown: ``target2base_matrix`` (target → robot base).

    The transformation chain used for reprojection is
    ``target2cam = inv(cam2end) @ inv(end2base) @ target2base``.
    """

    _primary_name = "cam2end"
    _secondary_name = "target2base"

    # ------------------------------------------------------------------
    # Backwards-compatible attribute aliases
    # ------------------------------------------------------------------

    @property
    def cam2end_matrix(self) -> Optional[np.ndarray]:
        """Camera → end-effector transformation (primary result)."""
        return self._primary_matrix

    @cam2end_matrix.setter
    def cam2end_matrix(self, value: Optional[np.ndarray]) -> None:
        self._primary_matrix = value

    @property
    def target2base_matrix(self) -> Optional[np.ndarray]:
        """Target → robot base transformation (secondary result)."""
        return self._secondary_matrix

    @target2base_matrix.setter
    def target2base_matrix(self, value: Optional[np.ndarray]) -> None:
        self._secondary_matrix = value

    # ------------------------------------------------------------------
    # Math hooks required by HandEyeBaseCalibrator
    # ------------------------------------------------------------------

    def _compose_target2cam(self,
                            primary_matrix: np.ndarray,
                            secondary_matrix: np.ndarray,
                            end2base_matrix: np.ndarray) -> np.ndarray:
        end2cam_matrix = np.linalg.inv(primary_matrix)
        base2end_matrix = np.linalg.inv(end2base_matrix)
        return end2cam_matrix @ base2end_matrix @ secondary_matrix

    def _solve_primary(self,
                       end2base_matrices_valid: List[np.ndarray],
                       target2cam_matrices_valid: List[np.ndarray],
                       method: int) -> np.ndarray:
        end2base_Rs = np.array([m[:3, :3] for m in end2base_matrices_valid])
        end2base_ts = np.array([m[:3, 3] for m in end2base_matrices_valid])

        target2cam_Rs = np.array([m[:3, :3] for m in target2cam_matrices_valid])
        target2cam_ts = np.array([m[:3, 3] for m in target2cam_matrices_valid])

        rvecs_array = np.array([cv2.Rodrigues(R)[0] for R in target2cam_Rs])
        tvecs_array = target2cam_ts.reshape(-1, 3, 1)

        cam2end_R, cam2end_t = cv2.calibrateHandEye(
            end2base_Rs, end2base_ts, rvecs_array, tvecs_array, method
        )

        cam2end_4x4 = np.eye(4)
        cam2end_4x4[:3, :3] = cam2end_R
        cam2end_4x4[:3, 3] = cam2end_t[:, 0]
        return cam2end_4x4

    def _compose_secondary_candidate(self,
                                     primary_matrix: np.ndarray,
                                     end2base_matrix: np.ndarray,
                                     target2cam_matrix: np.ndarray) -> np.ndarray:
        # target2base = end2base @ cam2end @ target2cam
        return end2base_matrix @ primary_matrix @ target2cam_matrix

    def _build_result_dict(self,
                           primary_matrix: np.ndarray,
                           secondary_matrix: np.ndarray,
                           rms_error: float,
                           before_opt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'cam2end_matrix': primary_matrix.copy(),
            'target2base_matrix': secondary_matrix.copy(),
            'rms_error': rms_error,
        }
        if before_opt is not None:
            result['before_opt'] = before_opt
        return result

    # ------------------------------------------------------------------
    # JSON serialization (domain-named keys)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize state, surfacing ``cam2end_matrix`` / ``target2base_matrix``."""
        data = super().to_json()

        if self._primary_matrix is not None:
            data['cam2end_matrix'] = self._primary_matrix.tolist()
        if self._secondary_matrix is not None:
            data['target2base_matrix'] = self._secondary_matrix.tolist()

        data['calibration_type'] = 'eye_in_hand'
        return data

    def from_json(self, data: dict) -> None:
        """Load state from a dict written by :meth:`to_json`."""
        super().from_json(data)

        if 'cam2end_matrix' in data:
            self._primary_matrix = np.array(data['cam2end_matrix'], dtype=np.float32)
        if 'target2base_matrix' in data:
            self._secondary_matrix = np.array(data['target2base_matrix'], dtype=np.float32)

    # ------------------------------------------------------------------
    # Domain-named accessors (back-compat with examples/tests/web app)
    # ------------------------------------------------------------------

    def set_cam2end_matrix(self, matrix: Optional[np.ndarray]) -> None:
        if matrix is None:
            self._primary_matrix = None
            return
        if not isinstance(matrix, np.ndarray):
            raise ValueError("cam2end_matrix must be a numpy array")
        if matrix.shape != (4, 4):
            raise ValueError(f"cam2end_matrix must be 4x4, got shape {matrix.shape}")
        self._primary_matrix = matrix.copy()

    def set_target2base_matrix(self, matrix: Optional[np.ndarray]) -> None:
        if matrix is None:
            self._secondary_matrix = None
            return
        if not isinstance(matrix, np.ndarray):
            raise ValueError("target2base_matrix must be a numpy array")
        if matrix.shape != (4, 4):
            raise ValueError(f"target2base_matrix must be 4x4, got shape {matrix.shape}")
        self._secondary_matrix = matrix.copy()

    def get_cam2end_matrix(self) -> Optional[np.ndarray]:
        return self._primary_matrix

    def get_target2base_matrix(self) -> Optional[np.ndarray]:
        return self._secondary_matrix

    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Return ``cam2end_matrix`` (the primary result for eye-in-hand)."""
        return self._primary_matrix

    # ------------------------------------------------------------------
    # Domain-named validation wrappers
    # ------------------------------------------------------------------

    def validate_eye_in_hand_data(self) -> bool:
        """Domain-named wrapper around :meth:`_validate_handeye_data`."""
        return self._validate_handeye_data()

    def is_eye_in_hand_calibrated(self) -> bool:
        """True iff calibration is complete and ``cam2end_matrix`` is set."""
        return self._is_handeye_calibrated()
