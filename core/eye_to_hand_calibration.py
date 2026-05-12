"""
Eye-to-Hand Calibration Module
==============================

Thin subclass of :class:`HandEyeBaseCalibrator` for the *camera fixed in the
workspace* configuration with the calibration target mounted on the
end-effector. All shared math (method sweep, joint optimization, reprojection
error, JSON IO) lives in the base class; this file only provides the
eye-to-hand specific composition kernels, the domain-specific attribute names
and a few thin compatibility wrappers.

Architecture::

    BaseCalibrator (images / patterns)
    └── HandEyeBaseCalibrator (robot poses / shared algorithm / IO)
        └── EyeToHandCalibrator (eye-to-hand specific kernels)
"""

from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from .hand_eye_base_calibration import HandEyeBaseCalibrator


class EyeToHandCalibrator(HandEyeBaseCalibrator):
    """
    Eye-to-hand hand-eye calibration (camera fixed, target on end-effector).

    Primary unknown: ``base2cam_matrix`` (robot base → camera).
    Secondary unknown: ``target2end_matrix`` (target → end-effector).

    The transformation chain used for reprojection is
    ``target2cam = base2cam @ end2base @ target2end``.
    """

    _primary_name = "base2cam"
    _secondary_name = "target2end"

    # ------------------------------------------------------------------
    # Backwards-compatible attribute aliases
    # ------------------------------------------------------------------

    @property
    def base2cam_matrix(self) -> Optional[np.ndarray]:
        """Robot base → camera transformation (primary result)."""
        return self._primary_matrix

    @base2cam_matrix.setter
    def base2cam_matrix(self, value: Optional[np.ndarray]) -> None:
        self._primary_matrix = value

    @property
    def target2end_matrix(self) -> Optional[np.ndarray]:
        """Target → end-effector transformation (secondary result)."""
        return self._secondary_matrix

    @target2end_matrix.setter
    def target2end_matrix(self, value: Optional[np.ndarray]) -> None:
        self._secondary_matrix = value

    # ------------------------------------------------------------------
    # Math hooks required by HandEyeBaseCalibrator
    # ------------------------------------------------------------------

    def _compose_target2cam(self,
                            primary_matrix: np.ndarray,
                            secondary_matrix: np.ndarray,
                            end2base_matrix: np.ndarray) -> np.ndarray:
        # target2cam = base2cam @ end2base @ target2end
        return primary_matrix @ end2base_matrix @ secondary_matrix

    def _solve_primary(self,
                       end2base_matrices_valid: List[np.ndarray],
                       target2cam_matrices_valid: List[np.ndarray],
                       method: int) -> np.ndarray:
        # Eye-to-hand wants base→end transforms, i.e. the inverses of end→base.
        base2end_Rs = []
        base2end_ts = []
        for m in end2base_matrices_valid:
            base2end = np.linalg.inv(m)
            base2end_Rs.append(base2end[:3, :3])
            base2end_ts.append(base2end[:3, 3])
        base2end_Rs = np.array(base2end_Rs)
        base2end_ts = np.array(base2end_ts)

        target2cam_Rs = np.array([m[:3, :3] for m in target2cam_matrices_valid])
        target2cam_ts = np.array([m[:3, 3] for m in target2cam_matrices_valid])

        rvecs_array = np.array([cv2.Rodrigues(R)[0] for R in target2cam_Rs])
        tvecs_array = target2cam_ts.reshape(-1, 3, 1)

        cam2base_R, cam2base_t = cv2.calibrateHandEye(
            base2end_Rs, base2end_ts, rvecs_array, tvecs_array, method
        )

        cam2base_4x4 = np.eye(4)
        cam2base_4x4[:3, :3] = cam2base_R
        cam2base_4x4[:3, 3] = cam2base_t[:, 0]

        # We want base→cam; invert.
        return np.linalg.inv(cam2base_4x4)

    def _compose_secondary_candidate(self,
                                     primary_matrix: np.ndarray,
                                     end2base_matrix: np.ndarray,
                                     target2cam_matrix: np.ndarray) -> np.ndarray:
        # target2end = inv(end2base) @ inv(base2cam) @ target2cam
        cam2base_matrix = np.linalg.inv(primary_matrix)
        base2end_matrix = np.linalg.inv(end2base_matrix)
        return base2end_matrix @ cam2base_matrix @ target2cam_matrix

    def _build_result_dict(self,
                           primary_matrix: np.ndarray,
                           secondary_matrix: np.ndarray,
                           rms_error: float,
                           before_opt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'base2cam_matrix': primary_matrix.copy(),
            'target2end_matrix': secondary_matrix.copy(),
            'rms_error': rms_error,
        }
        if before_opt is not None:
            result['before_opt'] = before_opt
        return result

    # ------------------------------------------------------------------
    # JSON serialization (domain-named keys)
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize state, surfacing ``base2cam_matrix`` / ``target2end_matrix``."""
        data = super().to_json()

        if self._primary_matrix is not None:
            data['base2cam_matrix'] = self._primary_matrix.tolist()
        if self._secondary_matrix is not None:
            data['target2end_matrix'] = self._secondary_matrix.tolist()

        data['calibration_type'] = 'eye_to_hand'
        return data

    def from_json(self, data: dict) -> None:
        """Load state from a dict written by :meth:`to_json`."""
        super().from_json(data)

        if 'base2cam_matrix' in data:
            self._primary_matrix = np.array(data['base2cam_matrix'], dtype=np.float32)
        if 'target2end_matrix' in data:
            self._secondary_matrix = np.array(data['target2end_matrix'], dtype=np.float32)

    # ------------------------------------------------------------------
    # Domain-named accessors (back-compat with examples/tests/web app)
    # ------------------------------------------------------------------

    def set_base2cam_matrix(self, matrix: Optional[np.ndarray]) -> None:
        if matrix is None:
            self._primary_matrix = None
            return
        if not isinstance(matrix, np.ndarray):
            raise ValueError("base2cam_matrix must be a numpy array")
        if matrix.shape != (4, 4):
            raise ValueError(f"base2cam_matrix must be 4x4, got shape {matrix.shape}")
        self._primary_matrix = matrix.copy()

    def set_target2end_matrix(self, matrix: Optional[np.ndarray]) -> None:
        if matrix is None:
            self._secondary_matrix = None
            return
        if not isinstance(matrix, np.ndarray):
            raise ValueError("target2end_matrix must be a numpy array")
        if matrix.shape != (4, 4):
            raise ValueError(f"target2end_matrix must be 4x4, got shape {matrix.shape}")
        self._secondary_matrix = matrix.copy()

    def get_base2cam_matrix(self) -> Optional[np.ndarray]:
        return self._primary_matrix

    def get_target2end_matrix(self) -> Optional[np.ndarray]:
        return self._secondary_matrix

    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Return ``base2cam_matrix`` (the primary result for eye-to-hand)."""
        return self._primary_matrix

    # ------------------------------------------------------------------
    # Domain-named validation wrappers
    # ------------------------------------------------------------------

    def validate_eye_to_hand_data(self) -> bool:
        """Domain-named wrapper around :meth:`_validate_handeye_data`."""
        return self._validate_handeye_data()

    def is_eye_to_hand_calibrated(self) -> bool:
        """True iff calibration is complete and ``base2cam_matrix`` is set."""
        return self._is_handeye_calibrated()
