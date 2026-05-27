"""Tests for isaac_auto_scene.scene_gen (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_auto_scene.calibrate import (
    CalibrationOutput,
    build_calibration,
)
from isaac_auto_scene.cad import CADResult
from isaac_auto_scene.capture import CaptureResult
from isaac_auto_scene.register import RegistrationResult
from isaac_auto_scene.scene_gen import (
    SO101_JOINT_NAMES,
    WARM_UP_FRAMES,
    build_scene_spec,
    resolve_default_so101_usd,
    warm_up_render,
    write_usd_stub,
)
from isaac_auto_scene.utils.intrinsics import CameraIntrinsics
from isaac_auto_scene.utils.transforms import quat_xyzw_to_rotation_matrix


def _stub_calib(tmp_path: Path) -> CalibrationOutput:
    import open3d as o3d
    import trimesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).normal(size=(10, 3)))
    cap = CaptureResult(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.zeros((4, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics(640, 480, 385.0, 385.0, 320.0, 240.0),
        pcd=pcd,
    )
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    cad = CADResult(
        mesh=mesh,
        points=np.array(mesh.vertices, dtype=np.float64),
        link_transforms={"base": np.eye(4)},
        joint_angles={},
    )
    T = np.eye(4)
    T[:3, 3] = [0.05, -0.02, 0.3]
    reg = RegistrationResult(
        T=T, fitness=0.9, inlier_rmse_m=0.002, used_fallback=False, n_restarts=1
    )
    return build_calibration(cap, cad, reg)


def test_build_scene_spec_pinhole(tmp_path: Path) -> None:
    calib = _stub_calib(tmp_path)
    spec = build_scene_spec(calib)
    assert spec.pinhole_cfg["width"] == 640
    assert spec.pinhole_cfg["height"] == 480
    assert spec.pinhole_cfg["focal_length"] > 0
    assert spec.pinhole_cfg["horizontal_aperture"] == pytest.approx(20.955)


def test_write_usd_stub_nonempty(tmp_path: Path) -> None:
    calib = _stub_calib(tmp_path)
    spec = build_scene_spec(calib)
    out = write_usd_stub(spec, tmp_path / "scene.usda")
    assert out.exists()
    contents = out.read_text()
    assert contents.startswith("#usda 1.0")
    assert "def Camera \"D435\"" in contents
    assert "def Xform \"SO101\"" in contents
    # Table is now an Xform with a Cube child (so we can transform it).
    assert "def Xform \"Table\"" in contents
    assert "def Cube \"Geometry\"" in contents
    assert out.stat().st_size > 200


def test_write_usd_stub_with_ros2_flag(tmp_path: Path) -> None:
    calib = _stub_calib(tmp_path)
    spec = build_scene_spec(calib, enable_ros2=True)
    assert spec.enable_ros2 is True


def test_warm_up_render_count() -> None:
    counter = {"n": 0}

    def step() -> None:
        counter["n"] += 1

    warm_up_render(step)
    assert counter["n"] == WARM_UP_FRAMES


def test_warm_up_default_is_30() -> None:
    assert WARM_UP_FRAMES == 30


def test_so101_joint_names_match_urdf() -> None:
    assert SO101_JOINT_NAMES == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )


def test_so101_usd_path_default_none(tmp_path: Path) -> None:
    """SceneSpec.so101_usd_path defaults to None so calibration-only flows skip articulation."""
    calib = _stub_calib(tmp_path)
    spec = build_scene_spec(calib)
    assert spec.so101_usd_path is None


def test_resolve_default_so101_usd_or_none() -> None:
    """resolve_default_so101_usd returns either a valid path or None — never raises."""
    result = resolve_default_so101_usd()
    if result is not None:
        assert Path(result).exists()


def test_scene_spec_table_pose_defaults(tmp_path: Path) -> None:
    """Legacy calib without T_cam_table -> table at world origin (back-compat)."""
    calib = _stub_calib(tmp_path)
    assert calib.T_cam_table is None
    spec = build_scene_spec(calib)
    assert spec.table_position_m == (0.0, 0.0, 0.0)
    assert spec.table_quat_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_scene_spec_table_pose_from_calib(tmp_path: Path) -> None:
    """When calib carries T_cam_table the spec reflects it in the arm-base world frame.

    The stub calib has T_cam_arm = eye(3) | t=[0.05, -0.02, 0.3].
    T_cam_table = Rx(45°) | t=[0.0, 0.2, 0.5].
    T_world_table = inv(T_cam_arm) @ T_cam_table.
      inv(T_cam_arm): R=eye(3), t_inv = -[0.05, -0.02, 0.3] = [-0.05, 0.02, -0.3]
      t_world_table = eye @ [0.0, 0.2, 0.5] + [-0.05, 0.02, -0.3] = [-0.05, 0.22, 0.20]
      R_world_table = eye @ Rx(45°) = Rx(45°)  (same rotation)
    """
    import dataclasses

    calib_no_table = _stub_calib(tmp_path)
    # Build a calib with a non-trivial table pose: rotated 45° about X,
    # translated 0.5m forward, 0.2m down (camera-frame coords).
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.0, 0.2, 0.5]
    calib = dataclasses.replace(calib_no_table, T_cam_table=T.tolist())
    spec = build_scene_spec(calib)

    # Expected table position in arm-base world frame:
    # t_world_table = t_cam_table + t_wc = [0.0, 0.2, 0.5] + [-0.05, 0.02, -0.3]
    np.testing.assert_allclose(
        spec.table_position_m, (-0.05, 0.22, 0.2), atol=1e-9
    )
    # Rotation part is still Rx(45°) (since T_cam_arm has identity rotation).
    q = np.asarray(spec.table_quat_xyzw)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert abs(q[3] - np.cos(np.pi / 8)) < 1e-6  # w of half-angle 45°/2 about X


def test_scene_spec_table_pose_nonidentity_rotation(tmp_path: Path) -> None:
    """Non-identity T_cam_arm rotation catches reversed composition order.

    T_cam_arm: 30° about X, translation [0.1, 0.2, 0.5].
    T_cam_table: 45° about Y, translation [0.0, 0.1, 0.4].

    These rotations do not commute (Rx(30°) @ Ry(45°) != Ry(45°) @ Rx(30°)),
    so if build_scene_spec computes T_cam_table @ inv(T_cam_arm) instead of
    inv(T_cam_arm) @ T_cam_table the rotation and translation will both differ
    from the expected values, causing this test to fail.
    """
    import dataclasses

    from isaac_auto_scene.utils.transforms import (
        quat_xyzw_to_rotation_matrix,
        rotation_matrix_to_quat_xyzw,
    )

    # --- arm pose: 30° about X ---
    theta_arm = np.pi / 6  # 30°
    R_cam_arm = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_arm), -np.sin(theta_arm)],
            [0.0, np.sin(theta_arm), np.cos(theta_arm)],
        ]
    )
    t_cam_arm = np.array([0.1, 0.2, 0.5])

    # --- table pose in camera frame: 45° about Y ---
    theta_tbl = np.pi / 4  # 45°
    R_cam_table = np.array(
        [
            [np.cos(theta_tbl), 0.0, np.sin(theta_tbl)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_tbl), 0.0, np.cos(theta_tbl)],
        ]
    )
    t_cam_table = np.array([0.0, 0.1, 0.4])

    # Build 4x4 matrices.
    T_cam_arm_4x4 = np.eye(4)
    T_cam_arm_4x4[:3, :3] = R_cam_arm
    T_cam_arm_4x4[:3, 3] = t_cam_arm

    T_cam_table_4x4 = np.eye(4)
    T_cam_table_4x4[:3, :3] = R_cam_table
    T_cam_table_4x4[:3, 3] = t_cam_table

    # Expected: T_world_table = inv(T_cam_arm) @ T_cam_table
    T_world_table_expected = np.linalg.inv(T_cam_arm_4x4) @ T_cam_table_4x4
    t_expected = T_world_table_expected[:3, 3]
    R_expected = T_world_table_expected[:3, :3]

    # Wire a CalibrationOutput with these exact arm pose values.
    calib_base = _stub_calib(tmp_path)
    q_arm = rotation_matrix_to_quat_xyzw(R_cam_arm).tolist()
    calib = dataclasses.replace(
        calib_base,
        quat_xyzw=q_arm,
        translation_m=t_cam_arm.tolist(),
        T_cam_table=T_cam_table_4x4.tolist(),
    )

    spec = build_scene_spec(calib)

    # Translation must match the expected world-frame translation.
    np.testing.assert_allclose(
        np.asarray(spec.table_position_m), t_expected, atol=1e-6
    )

    # Rotation matrix recovered from the spec quaternion must match R_expected.
    R_spec = quat_xyzw_to_rotation_matrix(np.asarray(spec.table_quat_xyzw))
    np.testing.assert_allclose(R_spec, R_expected, atol=1e-6)


def test_scene_spec_camera_is_inverse_of_arm_in_camera(tmp_path: Path) -> None:
    """Camera pose in spec must be the exact inverse of T_cam_arm.

    Build a calib with a non-trivial rotation and translation.  Then verify
    that reconstructing T_cam_arm from the spec's camera pose round-trips:

      R = quat_xyzw_to_rotation_matrix(camera_quat_xyzw)  -- this is R_wc
      t = camera_position_m                                -- this is t_wc

    The inverse of T_world_cam gives back T_cam_arm:
      R_cam_arm = R_wc.T
      t_cam_arm = -R_wc.T @ t_wc

    Which must equal calib.quat_xyzw / calib.translation_m.
    Also assert arm is at origin/identity.
    """
    import dataclasses

    # Non-trivial rotation: 30° about Z.
    theta = np.pi / 6
    R_arm = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ]
    )
    t_arm = np.array([0.12, -0.08, 0.45])

    calib_base = _stub_calib(tmp_path)
    from isaac_auto_scene.utils.transforms import rotation_matrix_to_quat_xyzw
    q_arm = rotation_matrix_to_quat_xyzw(R_arm).tolist()

    calib = dataclasses.replace(
        calib_base,
        quat_xyzw=q_arm,
        translation_m=t_arm.tolist(),
    )

    spec = build_scene_spec(calib)

    # Arm must be at the world origin.
    assert spec.arm_position_m == (0.0, 0.0, 0.0)
    assert spec.arm_quat_xyzw == (0.0, 0.0, 0.0, 1.0)

    # Recover T_cam_arm from spec camera pose.
    R_wc = quat_xyzw_to_rotation_matrix(np.asarray(spec.camera_quat_xyzw))
    t_wc = np.asarray(spec.camera_position_m)

    R_recovered = R_wc.T
    t_recovered = -R_wc.T @ t_wc

    # Must round-trip to original T_cam_arm.
    np.testing.assert_allclose(R_recovered, R_arm, atol=1e-9)
    np.testing.assert_allclose(t_recovered, t_arm, atol=1e-9)
