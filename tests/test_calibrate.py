"""Tests for isaac_auto_scene.calibrate (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from isaac_auto_scene.calibrate import (
    CalibrationOutput,
    build_calibration,
    load_calibration,
    save_calibration,
    _rot_to_quat_xyzw,
)
from isaac_auto_scene.cad import CADResult
from isaac_auto_scene.capture import CaptureResult
from isaac_auto_scene.register import RegistrationResult
from isaac_auto_scene.utils.intrinsics import CameraIntrinsics


def _fake_capture() -> CaptureResult:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).normal(size=(10, 3)))
    return CaptureResult(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.zeros((4, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(640, 480, 385.0, 385.0, 320.0, 240.0),
        pcd=pcd,
    )


def _fake_cad() -> CADResult:
    import trimesh

    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    return CADResult(
        mesh=mesh,
        points=np.array(mesh.vertices, dtype=np.float64),
        link_transforms={"base": np.eye(4)},
        joint_angles={"j1": 0.5, "j2": -0.3},
    )


def _fake_register(T: np.ndarray, fit: float = 0.9, rmse: float = 0.002) -> RegistrationResult:
    return RegistrationResult(
        T=T, fitness=fit, inlier_rmse_m=rmse, used_fallback=False, n_restarts=1
    )


def test_quat_xyzw_identity() -> None:
    q = _rot_to_quat_xyzw(np.eye(3))
    np.testing.assert_allclose(q, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)


def test_quat_xyzw_unit_norm() -> None:
    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 1.234
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    q = _rot_to_quat_xyzw(R)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-10


def test_build_calibration_roundtrip(tmp_path: Path) -> None:
    cap = _fake_capture()
    cad = _fake_cad()
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.05]
    reg = _fake_register(T)

    calib = build_calibration(cap, cad, reg)
    out = tmp_path / "calib.json"
    save_calibration(calib, out)
    loaded = load_calibration(out)

    assert isinstance(loaded, CalibrationOutput)
    np.testing.assert_allclose(loaded.translation_m, [0.1, -0.2, 0.05])
    assert loaded.icp_fitness == 0.9
    assert loaded.joint_angles_at_capture == {"j1": 0.5, "j2": -0.3}
    assert loaded.intrinsics["width"] == 640
    assert len(loaded.quat_xyzw) == 4
