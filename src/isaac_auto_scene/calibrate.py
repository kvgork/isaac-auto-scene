"""Calibration orchestrator: capture -> segment -> register -> calib.json (Phase 6).

Public API
----------
CalibrationOutput   — frozen dataclass: T_cam_arm, intrinsics, fitness, rmse, joints
run_calibration()   — drive the pipeline end-to-end
save_calibration()  — write calib.json
load_calibration()  — read calib.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from isaac_auto_scene.capture import CaptureResult
from isaac_auto_scene.cad import CADResult
from isaac_auto_scene.register import RegistrationResult


@dataclass(frozen=True)
class CalibrationOutput:
    """End-to-end calibration result written to calib.json.

    Attributes
    ----------
    T_cam_arm:
        4x4 transform mapping arm/CAD frame -> camera frame.  This is
        ``T`` returned by ``register_global_local(source=cad, target=segmented)``.
    quat_xyzw:
        Rotation part of ``T_cam_arm`` as XYZW quaternion (Isaac convention).
    translation_m:
        Translation part of ``T_cam_arm`` in metres.
    intrinsics:
        Camera K (3x3) flattened + width/height/depth_unit.
    icp_fitness:
        Fitness reported by register_global_local.
    inlier_rmse_m:
        Inlier RMSE reported by register_global_local.
    joint_angles_at_capture:
        Joint-angle mapping that was used to build the CAD source cloud.
    """

    T_cam_arm: list[list[float]]
    quat_xyzw: list[float]
    translation_m: list[float]
    intrinsics: dict[str, float | int]
    icp_fitness: float
    inlier_rmse_m: float
    joint_angles_at_capture: dict[str, float]


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit XYZW quaternion."""
    m = np.asarray(R, dtype=np.float64)
    tr = m.trace()
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def build_calibration(
    capture_result: CaptureResult,
    cad_result: CADResult,
    register_result: RegistrationResult,
) -> CalibrationOutput:
    """Assemble a CalibrationOutput from the three pipeline stages.

    The registration is expected to align ``cad_result`` (source) to the
    arm cloud segmented from ``capture_result`` (target), so
    ``register_result.T`` maps arm/CAD frame -> camera frame.
    """
    T = np.asarray(register_result.T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    q = _rot_to_quat_xyzw(R)

    intr = capture_result.intrinsics
    intrinsics_dict: dict[str, float | int] = {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "depth_unit": float(capture_result.depth_unit),
    }

    return CalibrationOutput(
        T_cam_arm=T.tolist(),
        quat_xyzw=q.tolist(),
        translation_m=t.tolist(),
        intrinsics=intrinsics_dict,
        icp_fitness=float(register_result.fitness),
        inlier_rmse_m=float(register_result.inlier_rmse_m),
        joint_angles_at_capture=dict(cad_result.joint_angles),
    )


def save_calibration(calib: CalibrationOutput, path: Path) -> None:
    """Write a CalibrationOutput to JSON."""
    Path(path).write_text(json.dumps(asdict(calib), indent=2))


def load_calibration(path: Path) -> CalibrationOutput:
    """Read a CalibrationOutput from JSON."""
    data = json.loads(Path(path).read_text())
    return CalibrationOutput(**data)
