"""Unit tests for poses.py — JointPose, ArmDriver mocks, YAML loader,
validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from isaac_auto_scene.cad import load_urdf
from isaac_auto_scene.poses import (
    JointPose,
    MockArmDriver,
    PoseValidationError,
    load_poses,
    validate_pose,
    validate_pose_set,
)
from tests.fixtures.minimal_urdf import write_minimal_urdf


# ---------------------------------------------------------------------------
# JointPose / MockArmDriver
# ---------------------------------------------------------------------------


def test_mock_arm_driver_roundtrip():
    drv = MockArmDriver(joint_names=("a", "b"))
    with drv:
        drv.command_joints({"a": 0.5, "b": -0.2})
        rb = drv.read_joints()
    assert rb == {"a": 0.5, "b": -0.2}


def test_mock_arm_driver_requires_connect():
    drv = MockArmDriver(joint_names=("a",))
    with pytest.raises(RuntimeError):
        drv.command_joints({"a": 0.0})
    with pytest.raises(RuntimeError):
        drv.read_joints()


def test_mock_arm_driver_unknown_joint_raises():
    drv = MockArmDriver(joint_names=("a",))
    with drv:
        with pytest.raises(KeyError):
            drv.command_joints({"unknown": 0.0})


def test_mock_arm_driver_readback_noise_is_deterministic():
    drv = MockArmDriver(joint_names=("a",), readback_noise_rad=0.01, seed=42)
    with drv:
        drv.command_joints({"a": 0.0})
        first = drv.read_joints()["a"]
    drv2 = MockArmDriver(joint_names=("a",), readback_noise_rad=0.01, seed=42)
    with drv2:
        drv2.command_joints({"a": 0.0})
        second = drv2.read_joints()["a"]
    assert first == second
    assert abs(first) > 0.0


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_load_poses_parses_yaml(tmp_path: Path):
    path = tmp_path / "poses.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "poses": [
                    {"name": "home", "joints": {"shoulder": 0.0}, "settle_s": 0.5},
                    {"name": "left", "joints": {"shoulder": 1.0}},
                ]
            }
        )
    )
    poses = load_poses(path)
    assert [p.name for p in poses] == ["home", "left"]
    assert poses[0].settle_s == 0.5
    assert poses[1].settle_s == 1.0  # default
    assert poses[1].joints == {"shoulder": 1.0}


def test_load_poses_rejects_missing_key(tmp_path: Path):
    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump({"poses": [{"name": "x"}]}))  # joints missing
    with pytest.raises(ValueError):
        load_poses(path)


def test_load_poses_rejects_empty(tmp_path: Path):
    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump({"poses": []}))
    with pytest.raises(ValueError):
        load_poses(path)


# ---------------------------------------------------------------------------
# Validation against URDF
# ---------------------------------------------------------------------------


def test_validate_pose_within_limits(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    pose = JointPose(name="home", joints={"shoulder": 0.0})
    assert validate_pose(pose, urdf) == []


def test_validate_pose_below_lower(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    pose = JointPose(name="bad", joints={"shoulder": -10.0})
    errors = validate_pose(pose, urdf)
    assert len(errors) == 1
    assert "lower" in errors[0].reason


def test_validate_pose_above_upper(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    pose = JointPose(name="bad", joints={"shoulder": 10.0})
    errors = validate_pose(pose, urdf)
    assert any("upper" in e.reason for e in errors)


def test_validate_pose_unknown_joint(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    pose = JointPose(name="bad", joints={"nope": 0.0})
    errors = validate_pose(pose, urdf)
    assert any("unknown joint" in e.reason for e in errors)


def test_validate_pose_set_dedups_names(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    p1 = JointPose(name="dup", joints={"shoulder": 0.0})
    p2 = JointPose(name="dup", joints={"shoulder": 0.1})
    report = validate_pose_set([p1, p2], urdf)
    assert not report.ok
    assert any("duplicate" in e.reason for e in report.errors)


def test_validate_pose_set_ok(tmp_path: Path):
    urdf_path = write_minimal_urdf(tmp_path)
    urdf = load_urdf(urdf_path)
    poses = [
        JointPose(name="a", joints={"shoulder": 0.0}),
        JointPose(name="b", joints={"shoulder": 0.5}),
    ]
    report = validate_pose_set(poses, urdf)
    assert report.ok
    assert isinstance(report.errors, tuple)
    assert report.errors == ()
    assert bool(report) is True


def test_pose_validation_error_is_frozen():
    err = PoseValidationError(pose_name="x", reason="r")
    with pytest.raises(Exception):
        err.pose_name = "y"  # type: ignore[misc]
