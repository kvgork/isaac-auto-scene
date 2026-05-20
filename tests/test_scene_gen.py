"""Tests for isaac_auto_scene.scene_gen (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_auto_scene.calibrate import (
    CalibrationOutput,
    build_calibration,
    save_calibration,
)
from isaac_auto_scene.cad import CADResult
from isaac_auto_scene.capture import CaptureResult
from isaac_auto_scene.register import RegistrationResult
from isaac_auto_scene.scene_gen import (
    WARM_UP_FRAMES,
    build_scene_spec,
    warm_up_render,
    write_usd_stub,
)
from isaac_auto_scene.utils.intrinsics import CameraIntrinsics


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
    assert "def Cube \"Table\"" in contents
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
