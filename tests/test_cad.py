"""Tests for isaac_auto_scene.cad (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_auto_scene.cad import assemble_pcd, load_urdf
from tests.fixtures.minimal_urdf import write_minimal_urdf


@pytest.fixture(scope="module")
def urdf(tmp_path_factory: pytest.TempPathFactory):
    path = write_minimal_urdf(tmp_path_factory.mktemp("urdf"))
    return load_urdf(path)


def test_fk_assembly_zero_pose_extent(urdf) -> None:
    """At zero joint angle the assembly extends ~30 cm in +X (arm + base)."""
    result = assemble_pcd(urdf, {"shoulder": 0.0}, target_n_points=4_000)
    pts = result.points

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    extent_x = x_max - x_min
    extent_z = z_max - z_min

    assert 0.18 < extent_x < 0.35, f"X extent {extent_x:.3f} m outside expected range"
    assert 0.10 < extent_z < 0.20, f"Z extent {extent_z:.3f} m outside expected range"


def test_fk_assembly_90deg_rotates_arm(urdf) -> None:
    """At shoulder=+pi/2 the arm should extend into +Y, not +X."""
    res0 = assemble_pcd(urdf, {"shoulder": 0.0}, target_n_points=4_000)
    res90 = assemble_pcd(urdf, {"shoulder": float(np.pi / 2.0)}, target_n_points=4_000)

    y_range_0 = float(res0.points[:, 1].max() - res0.points[:, 1].min())
    y_range_90 = float(res90.points[:, 1].max() - res90.points[:, 1].min())

    assert y_range_90 > y_range_0 + 0.10, (
        f"shoulder=90° did not extend Y range as expected: "
        f"y_range_0={y_range_0:.3f}, y_range_90={y_range_90:.3f}"
    )


def test_point_count_matches_target(urdf) -> None:
    """Poisson-disk sampler returns approximately the requested point count."""
    target = 5_000
    result = assemble_pcd(urdf, target_n_points=target)
    n = len(result.points)
    assert abs(n - target) <= target * 0.1, f"got {n} points, target {target}"


def test_load_urdf_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_urdf(tmp_path / "nope.urdf")


def test_link_transforms_present(urdf) -> None:
    """Every link should have a transform entry."""
    result = assemble_pcd(urdf, target_n_points=2_000)
    assert "base" in result.link_transforms
    assert "arm" in result.link_transforms
    for T in result.link_transforms.values():
        assert T.shape == (4, 4)
