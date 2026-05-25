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


def _dense_arm_cluster(rng: np.random.Generator, center: np.ndarray, n: int = 2000) -> np.ndarray:
    """Compact dense cluster so DBSCAN keeps it at default eps."""
    return center + rng.normal(scale=0.008, size=(n, 3))


def test_workspace_z_crop_drops_far_points() -> None:
    """workspace_z_max_m removes background points before plane fit."""
    import open3d as o3d

    rng = np.random.default_rng(0)
    table = rng.uniform(low=[-0.2, -0.2, 0.499], high=[0.2, 0.2, 0.501], size=(1500, 3))
    bg_wall = rng.uniform(low=[-0.5, -0.5, 1.499], high=[0.5, 0.5, 1.501], size=(3000, 3))
    arm = _dense_arm_cluster(rng, np.array([0.0, 0.0, 0.55]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([table, bg_wall, arm]))

    result = segment_table_arm(pcd, workspace_z_max_m=1.0)
    table_z_mean = float(np.asarray(result.table_cloud.points)[:, 2].mean())
    assert 0.495 < table_z_mean < 0.505, f"plane fit picked wrong surface: z_mean={table_z_mean}"


def test_expected_up_rejects_misaligned_plane() -> None:
    """expected_up forces selection of an up-aligned plane over a sloped one."""
    import open3d as o3d

    rng = np.random.default_rng(1)
    wall_y = rng.uniform(-0.3, 0.3, size=(4000, 1))
    wall_z = rng.uniform(0.4, 1.0, size=(4000, 1))
    wall = np.hstack([0.5 * np.ones_like(wall_y) + rng.normal(scale=0.001, size=wall_y.shape), wall_y, wall_z])
    table_x = rng.uniform(-0.2, 0.2, size=(1500, 1))
    table_y = rng.uniform(-0.2, 0.2, size=(1500, 1))
    table = np.hstack([table_x, table_y, 0.45 * np.ones_like(table_x) + rng.normal(scale=0.001, size=table_x.shape)])
    arm = _dense_arm_cluster(rng, np.array([0.0, 0.0, 0.55]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([wall, table, arm]))

    result_default = segment_table_arm(pcd)
    n_default = np.abs(np.asarray(result_default.plane_model[:3]))
    assert n_default.argmax() == 0, "default RANSAC should pick the wall"

    result_up = segment_table_arm(pcd, expected_up=(0.0, 0.0, 1.0))
    n_up = np.asarray(result_up.plane_model[:3])
    n_up /= np.linalg.norm(n_up)
    assert abs(float(n_up[2])) > 0.86, f"plane normal not up-aligned: {n_up}"


def test_arm_merge_radius_unions_nearby_clusters() -> None:
    """arm_merge_radius_m absorbs neighbouring DBSCAN clusters into the arm."""
    import open3d as o3d

    rng = np.random.default_rng(3)
    table = rng.uniform(low=[-0.3, -0.3, 0.499], high=[0.3, 0.3, 0.501], size=(2000, 3))
    # Two dense arm fragments 8 cm apart (DBSCAN at default 15 mm eps will
    # NOT bridge them).
    arm_a = _dense_arm_cluster(rng, np.array([0.0, 0.0, 0.55]), n=1800)
    arm_b = _dense_arm_cluster(rng, np.array([0.08, 0.0, 0.58]), n=900)
    # Distant background cluster — must NOT be merged.
    bg = _dense_arm_cluster(rng, np.array([0.0, 0.0, 1.10]), n=900)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([table, arm_a, arm_b, bg]))

    # Default: arm cloud = only the largest cluster (arm_a).
    seg_default = segment_table_arm(pcd)
    assert 1700 < len(seg_default.arm_cloud.points) < 2000

    # With merge radius 0.15 m: arm_b joins (centroid ~0.085 m away),
    # bg stays out (~0.55 m away).
    seg_merged = segment_table_arm(pcd, arm_merge_radius_m=0.15)
    n = len(seg_merged.arm_cloud.points)
    assert 2600 < n < 3000, f"expected arm_a + arm_b ~2700 points; got {n}"


def test_expected_up_raises_when_no_aligned_plane() -> None:
    """Raises if no plane within tolerance is found within the attempt budget."""
    import open3d as o3d

    # Build a cloud where the only planes are sloped (no horizontal plane).
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.3, 0.3, size=(2000, 3))
    pts[:, 2] = 0.3 * pts[:, 0] + 0.4 * pts[:, 1] + 0.5  # sloped
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    with pytest.raises(ValueError, match="no plane within"):
        segment_table_arm(
            pcd,
            expected_up=(0.0, 0.0, 1.0),
            up_tolerance_deg=10.0,
            max_plane_attempts=2,
        )
