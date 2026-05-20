"""Tests for isaac_auto_scene.segment (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_auto_scene.segment import segment_table_arm
from tests.fixtures.synthetic_pcd import make_table_plus_arm_pcd


def test_plane_normal_within_half_degree() -> None:
    """RANSAC plane must recover the +Z normal within 0.5°."""
    pcd = make_table_plus_arm_pcd(noise_std=0.002, seed=11)
    result = segment_table_arm(pcd)

    a, b, c, _ = result.plane_model
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    if normal[2] < 0:
        normal = -normal
    expected = np.array([0.0, 0.0, 1.0])
    cos_angle = float(np.clip(np.dot(normal, expected), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    assert angle_deg < 0.5, f"plane normal off by {angle_deg:.3f}° (got {normal})"


def test_plane_centroid_within_5mm() -> None:
    """T_world_table origin must be within 5 mm of (0,0,0)."""
    pcd = make_table_plus_arm_pcd(noise_std=0.002, seed=12)
    result = segment_table_arm(pcd)

    origin = result.T_world_table[:3, 3]
    err = float(np.linalg.norm(origin - np.array([0.0, 0.0, 0.0])))
    assert err < 0.005, f"plane origin off by {err*1000:.2f} mm"


def test_inlier_rmse_under_3mm() -> None:
    """Inlier RMSE on synthetic data with 2 mm noise must be < 3 mm."""
    pcd = make_table_plus_arm_pcd(noise_std=0.002, seed=13)
    result = segment_table_arm(pcd)

    assert result.inlier_rmse_m < 0.003, (
        f"inlier RMSE {result.inlier_rmse_m*1000:.2f} mm exceeds 3 mm"
    )


def test_arm_cloud_recovered() -> None:
    """Arm cluster must contain most of the original arm points."""
    pcd = make_table_plus_arm_pcd(n_table=8000, n_arm=2000, noise_std=0.002, seed=14)
    result = segment_table_arm(pcd)

    n_arm = len(result.arm_cloud.points)
    assert n_arm > 1000, f"arm cloud too small: {n_arm} points (expected ~2000)"
    arm_pts = np.asarray(result.arm_cloud.points)
    z_mean = float(arm_pts[:, 2].mean())
    assert z_mean > 0.10, f"arm cluster z-mean {z_mean:.3f} m below expected ~0.15 m"


def test_transform_orthonormal() -> None:
    """T_world_table rotation must be orthonormal with det = +1."""
    pcd = make_table_plus_arm_pcd(seed=15)
    result = segment_table_arm(pcd)

    R = result.T_world_table[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_segment_rejects_tiny_pcd() -> None:
    """ValueError if input PCD has < 100 points."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(50, 3))
    with pytest.raises(ValueError, match="too small"):
        segment_table_arm(pcd)
