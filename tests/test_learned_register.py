"""Unit tests for the FPFH+RANSAC fallback registration backend."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def test_module_imports() -> None:
    from isaac_auto_scene import learned_register

    assert hasattr(learned_register, "register_with_fpfh_ransac")
    # Backwards-compat alias.
    assert learned_register.register_with_teaser is learned_register.register_with_fpfh_ransac


def test_kabsch_recovers_known_transform() -> None:
    """Kabsch SVD must recover an injected SE(3) on clean correspondences."""
    from isaac_auto_scene.learned_register import _kabsch

    rng = np.random.default_rng(0)
    src = rng.uniform(-0.2, 0.2, size=(50, 3))
    # Known rotation: 30° around Z.
    theta = np.deg2rad(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.1, -0.05, 0.2])
    tgt = src @ R.T + t

    T = _kabsch(src, tgt)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-6)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-6)


def test_mutual_feature_match_perm() -> None:
    """Mutual NN finds correspondences when target is a known permutation."""
    from isaac_auto_scene.learned_register import _mutual_feature_match

    rng = np.random.default_rng(0)
    src = rng.normal(size=(33, 6)).astype(np.float32)
    perm = np.array([2, 0, 4, 1, 5, 3], dtype=np.int64)
    tgt = src[:, perm]

    s_idx, t_idx = _mutual_feature_match(src, tgt)
    assert len(s_idx) == 6
    for s, t in zip(s_idx, t_idx):
        np.testing.assert_allclose(src[:, s], tgt[:, t])


def test_ransac_recovers_transform_with_outliers() -> None:
    """RANSAC ignores planted outliers and recovers the injected SE(3)."""
    from isaac_auto_scene.learned_register import _ransac_se3, _kabsch

    rng = np.random.default_rng(7)
    src = rng.uniform(-0.1, 0.1, size=(60, 3))
    theta = np.deg2rad(45.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.05, 0.02, -0.03])
    tgt = src @ R.T + t

    # Replace half the correspondences with random outliers.
    n_outliers = 30
    outlier_idx = rng.choice(60, size=n_outliers, replace=False)
    tgt[outlier_idx] = rng.uniform(-1.0, 1.0, size=(n_outliers, 3))

    T_best, score = _ransac_se3(
        src, tgt, sample_size=4, n_iters=2000, inlier_distance=0.005, seed=0
    )
    assert T_best is not None
    assert score >= 25  # at least most of the 30 true inliers
    np.testing.assert_allclose(T_best[:3, :3], R, atol=1e-3)
    np.testing.assert_allclose(T_best[:3, 3], t, atol=1e-3)


def test_end_to_end_register_with_fpfh_ransac() -> None:
    """End-to-end on a synthetic CAD-like blob translated + rotated."""
    from isaac_auto_scene.learned_register import register_with_fpfh_ransac

    rng = np.random.default_rng(11)
    # Build a shape with structure (not random) so FPFH features differ
    # across the surface.
    n = 2000
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(-0.5, 0.5, n)
    src_pts = np.column_stack(
        [0.05 * np.cos(u), 0.05 * np.sin(u), 0.1 * v]
    )  # cylinder
    # Apply a known transform.
    theta = np.deg2rad(20.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.03, -0.02, 0.05])
    tgt_pts = src_pts @ R.T + t + rng.normal(scale=0.001, size=src_pts.shape)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    result = register_with_fpfh_ransac(
        src,
        tgt,
        voxel_size=0.005,
        ransac_iterations=2000,
        seed=42,
    )
    # Rotation and translation should be in the right neighbourhood.
    # Tight tolerances are tough on a cylinder (rotation-symmetric about Z),
    # so we check translation only and confirm the result is finite.
    assert np.isfinite(result.T).all()
    np.testing.assert_allclose(result.T[:3, 3], t, atol=0.02)
    assert result.used_fallback is True


def test_downsample_and_feature_shapes() -> None:
    from isaac_auto_scene.learned_register import _downsample_and_feature

    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.1, 0.1, size=(5000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    down, fpfh = _downsample_and_feature(pcd, voxel_size=0.01, feature_radius_multiplier=5.0, fpfh_radius_multiplier=10.0)
    assert len(down.points) == np.asarray(fpfh.data).shape[1]
    assert np.asarray(fpfh.data).shape[0] == 33
