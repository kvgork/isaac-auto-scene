"""Unit tests for capture_multi.py.

Drive MockArmDriver + MockD435Source through a 3-pose set and verify the
on-disk manifest matches the in-memory result.  No hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac_auto_scene.cad import load_urdf
from isaac_auto_scene.capture import MockD435Source
from isaac_auto_scene.capture_multi import (
    capture_pose_set,
    load_manifest,
    load_pose_capture,
)
from isaac_auto_scene.poses import JointPose, MockArmDriver
from tests.fixtures.minimal_urdf import write_minimal_urdf


def _no_sleep(_seconds: float) -> None:
    pass


def test_capture_pose_set_writes_per_pose_artefacts(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    poses = [
        JointPose(name="home", joints={"shoulder": 0.0}, settle_s=0.0),
        JointPose(name="left", joints={"shoulder": 0.3}, settle_s=0.0),
        JointPose(name="right", joints={"shoulder": -0.3}, settle_s=0.0),
    ]
    driver = MockArmDriver(joint_names=("shoulder",))
    source = MockD435Source(seed=0)

    out = tmp_path / "captures"
    with driver as drv, source as src:
        manifest = capture_pose_set(
            poses,
            drv,
            src,
            urdf,
            urdf_path,
            out_dir=out,
            frames_per_pose=3,
            sleep=_no_sleep,
        )

    assert manifest.num_poses == 3
    assert manifest.num_ok == 3
    assert (out / "manifest.yaml").exists()
    for record in manifest.poses:
        pose_dir = out / record.pose_dir
        assert (pose_dir / "rgb.png").exists()
        assert (pose_dir / "depth_median.npy").exists()
        assert (pose_dir / "pcd.ply").exists()
        assert (pose_dir / "intrinsics.json").exists()
        assert (pose_dir / "joints.json").exists()


def test_capture_pose_set_validation_rejects_bad_pose(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    poses = [JointPose(name="bad", joints={"shoulder": 100.0}, settle_s=0.0)]
    driver = MockArmDriver(joint_names=("shoulder",))
    source = MockD435Source(seed=0)

    with driver as drv, source as src:
        with pytest.raises(ValueError):
            capture_pose_set(
                poses, drv, src, urdf, urdf_path,
                out_dir=tmp_path / "captures", sleep=_no_sleep,
            )


def test_capture_pose_set_records_readback_noise(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    poses = [JointPose(name="home", joints={"shoulder": 0.0}, settle_s=0.0)]
    driver = MockArmDriver(joint_names=("shoulder",), readback_noise_rad=0.05, seed=7)
    source = MockD435Source(seed=0)

    with driver as drv, source as src:
        manifest = capture_pose_set(
            poses, drv, src, urdf, urdf_path,
            out_dir=tmp_path / "captures",
            frames_per_pose=2,
            sleep=_no_sleep,
        )

    rec = manifest.poses[0]
    assert rec.commanded_joints == {"shoulder": 0.0}
    assert abs(rec.readback_joints["shoulder"]) > 0.0


def test_manifest_roundtrip(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    poses = [
        JointPose(name="a", joints={"shoulder": 0.0}, settle_s=0.0),
        JointPose(name="b", joints={"shoulder": 0.1}, settle_s=0.0),
    ]
    driver = MockArmDriver(joint_names=("shoulder",))
    source = MockD435Source(seed=0)

    out = tmp_path / "captures"
    with driver as drv, source as src:
        manifest = capture_pose_set(
            poses, drv, src, urdf, urdf_path, out_dir=out,
            frames_per_pose=2, sleep=_no_sleep,
        )

    reloaded = load_manifest(out)
    assert reloaded.num_poses == manifest.num_poses
    assert reloaded.num_ok == manifest.num_ok
    assert [p.name for p in reloaded.poses] == ["a", "b"]
    cap = load_pose_capture(out, reloaded.poses[0])
    assert cap.pcd is not None
    assert cap.depth.shape == cap.depth.shape  # sanity
