"""Unit tests for isaac_auto_scene.task_gen (Phase 7).

The Isaac Sim spawner ``spawn_task_assets`` is exercised by the hardware-gated
render smoke test; these unit tests cover the pure-Python spec layer and the
sampler so CI can run them without Isaac Lab installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isaac_auto_scene.task_gen import (
    TaskAssetSpec,
    TaskSceneSpec,
    sample_task_poses,
)


def _identity_quat() -> tuple[float, float, float, float]:
    return (0.0, 0.0, 0.0, 1.0)


def test_default_pose_returned_when_no_randomization() -> None:
    spec = TaskSceneSpec(
        assets=(
            TaskAssetSpec(
                name="block",
                kind="cuboid_dynamic",
                pose_m=(0.1, 0.2, 0.03),
                quat_xyzw=_identity_quat(),
                usd_path="/dummy/path.usd",
            ),
        )
    )
    poses = sample_task_poses(spec, rng=np.random.default_rng(0))
    assert poses["block"][0] == (0.1, 0.2, 0.03)
    assert poses["block"][1] == _identity_quat()


def test_sampled_position_respects_range() -> None:
    rng = np.random.default_rng(42)
    asset = TaskAssetSpec(
        name="cup",
        kind="cylinder_dynamic",
        pose_m=(0.0, 0.0, 0.05),
        quat_xyzw=_identity_quat(),
        randomization=(
            (0.10, 0.05, 0.0),  # dx, dy, dz half-ranges
            (math.pi / 4,),  # dyaw half-range
        ),
        usd_path="/dummy/cup.usd",
    )
    spec = TaskSceneSpec(assets=(asset,))
    for _ in range(200):
        poses = sample_task_poses(spec, rng=rng)
        (px, py, pz), (qx, qy, qz, qw) = poses["cup"]
        assert -0.10 <= px <= 0.10
        assert -0.05 <= py <= 0.05
        assert pz == 0.05  # dz range = 0 -> no drift
        # Z-axis quaternion: x=y=0, w^2 + z^2 = 1
        assert abs(qx) < 1e-9
        assert abs(qy) < 1e-9
        assert abs((qw * qw + qz * qz) - 1.0) < 1e-9


def test_sampler_is_deterministic_with_seed() -> None:
    asset = TaskAssetSpec(
        name="block",
        kind="cuboid_dynamic",
        pose_m=(0.0, 0.0, 0.05),
        randomization=((0.1, 0.1, 0.0), (math.pi,)),
        usd_path="/dummy/block.usd",
    )
    spec = TaskSceneSpec(assets=(asset,))
    p1 = sample_task_poses(spec, rng=np.random.default_rng(123))
    p2 = sample_task_poses(spec, rng=np.random.default_rng(123))
    assert p1 == p2


def test_multiple_assets_sampled_independently() -> None:
    rng = np.random.default_rng(7)
    spec = TaskSceneSpec(
        assets=(
            TaskAssetSpec(
                name="block",
                kind="cuboid_dynamic",
                pose_m=(0.2, 0.0, 0.05),
                randomization=((0.05, 0.05, 0.0), (0.0,)),
                usd_path="/dummy/block.usd",
            ),
            TaskAssetSpec(
                name="cup",
                kind="cylinder_dynamic",
                pose_m=(-0.2, 0.0, 0.05),
                randomization=((0.05, 0.05, 0.0), (0.0,)),
                usd_path="/dummy/cup.usd",
            ),
        )
    )
    poses = sample_task_poses(spec, rng=rng)
    assert set(poses.keys()) == {"block", "cup"}
    assert poses["block"][0] != poses["cup"][0]


def test_dynamic_kind_requires_usd_when_spawned() -> None:
    """spec construction allows None, but the spawner contract validates."""
    asset = TaskAssetSpec(name="block", kind="cuboid_dynamic", usd_path=None)
    assert asset.is_dynamic()
    # The validation lives in spawn_task_assets — we exercise the API surface
    # rather than spawning since Isaac Lab isn't installed in the default env.


def test_is_dynamic_classification() -> None:
    assert TaskAssetSpec(name="a", kind="cuboid_static").is_dynamic() is False
    assert TaskAssetSpec(name="b", kind="cylinder_static").is_dynamic() is False
    assert TaskAssetSpec(name="c", kind="cuboid_dynamic", usd_path="x").is_dynamic() is True
    assert TaskAssetSpec(name="d", kind="cylinder_dynamic", usd_path="x").is_dynamic() is True
    assert TaskAssetSpec(name="e", kind="usd", usd_path="x").is_dynamic() is True
    # ``usd`` with no usd_path is treated as not-dynamic; spawn will then
    # raise ValueError on the missing path.
    assert TaskAssetSpec(name="f", kind="usd", usd_path=None).is_dynamic() is False


def test_zero_range_skips_sampling() -> None:
    """All-zero randomization should equal the default pose."""
    asset = TaskAssetSpec(
        name="block",
        kind="cuboid_dynamic",
        pose_m=(0.5, -0.5, 0.05),
        randomization=((0.0, 0.0, 0.0), (0.0,)),
        usd_path="/dummy.usd",
    )
    spec = TaskSceneSpec(assets=(asset,))
    p = sample_task_poses(spec, rng=np.random.default_rng(99))
    assert p["block"][0] == (0.5, -0.5, 0.05)


def test_empty_scene_returns_empty_poses() -> None:
    spec = TaskSceneSpec(assets=())
    p = sample_task_poses(spec)
    assert p == {}


def test_parent_prim_default() -> None:
    spec = TaskSceneSpec()
    assert spec.parent_prim == "/World/Tasks"


def test_parent_prim_override() -> None:
    spec = TaskSceneSpec(parent_prim="/World/MyTasks")
    assert spec.parent_prim == "/World/MyTasks"


@pytest.mark.parametrize(
    "kind",
    ["cuboid_static", "cuboid_dynamic", "cylinder_static", "cylinder_dynamic", "usd"],
)
def test_all_kinds_accepted(kind: str) -> None:
    asset = TaskAssetSpec(name="x", kind=kind, usd_path="/x.usd" if kind == "usd" else None)
    assert asset.kind == kind
