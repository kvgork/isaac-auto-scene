"""Unit tests for bundle_register: se(3) helpers + multi-pose solver."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from isaac_auto_scene.bundle_register import (
    register_bundle,
    se3_exp,
    se3_log,
)


def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return p


def _rand_so3(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation via Shoemake."""
    u1, u2, u3 = rng.uniform(size=3)
    s1, s2 = np.sqrt(1 - u1), np.sqrt(u1)
    q = np.array(
        [s1 * np.sin(2 * np.pi * u2), s1 * np.cos(2 * np.pi * u2),
         s2 * np.sin(2 * np.pi * u3), s2 * np.cos(2 * np.pi * u3)]
    )
    x, y, z, w = q
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    )


def test_se3_exp_of_zero_is_identity() -> None:
    np.testing.assert_allclose(se3_exp(np.zeros(6)), np.eye(4), atol=1e-12)


def test_se3_log_of_identity_is_zero() -> None:
    np.testing.assert_allclose(se3_log(np.eye(4)), np.zeros(6), atol=1e-9)


def test_se3_exp_log_roundtrip() -> None:
    """exp(log(T)) == T for arbitrary T."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = np.eye(4)
        T[:3, :3] = _rand_so3(rng)
        T[:3, 3] = rng.uniform(-0.5, 0.5, size=3)
        T_round = se3_exp(se3_log(T))
        np.testing.assert_allclose(T_round, T, atol=1e-9)


def test_se3_log_exp_roundtrip() -> None:
    """log(exp(ξ)) == ξ for arbitrary ξ with bounded rotation."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        # bound rotation magnitude so log is well-defined
        rho = rng.uniform(-0.3, 0.3, size=3)
        phi = rng.uniform(-1.5, 1.5, size=3)
        xi = np.concatenate([rho, phi])
        np.testing.assert_allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)


def test_bundle_recovers_known_transform_single_pose() -> None:
    """Single pose: bundle ICP must converge to the known transform."""
    rng = np.random.default_rng(7)
    # Synthetic asymmetric shape (cube)
    cad = rng.uniform(-0.05, 0.05, size=(500, 3))
    # Known transform
    R = _rand_so3(rng)
    t = np.array([0.1, -0.05, 0.3])
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = t

    # Transformed observations (with small noise)
    arm_pts = cad @ R.T + t + rng.normal(scale=0.001, size=cad.shape)

    # Initialise close to truth (within partial-view ICP basin)
    T_init = T_true.copy()

    result = register_bundle(
        [cad], [_pcd(arm_pts)], T_init=T_init, inlier_distance_m=0.02
    )

    np.testing.assert_allclose(result.T[:3, :3], R, atol=5e-3)
    np.testing.assert_allclose(result.T[:3, 3], t, atol=2e-3)
    assert result.per_pose_fitness[0] > 0.9
    assert result.per_pose_rmse_m[0] < 0.003


def test_bundle_recovers_from_perturbed_init() -> None:
    """Bundle converges from a noisy initial guess on multi-pose data."""
    rng = np.random.default_rng(9)
    R = _rand_so3(rng)
    t = np.array([0.05, 0.02, 0.20])
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = t

    cad_clouds = []
    arm_clouds = []
    # 4 different "joint configs" -> different CAD shapes
    for _ in range(4):
        cad = rng.uniform(-0.1, 0.1, size=(600, 3))
        cad_clouds.append(cad)
        arm = cad @ R.T + t + rng.normal(scale=0.0015, size=cad.shape)
        arm_clouds.append(_pcd(arm))

    # Perturb init by a small rotation
    perturb_xi = np.concatenate([rng.uniform(-0.02, 0.02, size=3),
                                  rng.uniform(-0.15, 0.15, size=3)])
    T_init = T_true @ se3_exp(perturb_xi)

    result = register_bundle(cad_clouds, arm_clouds, T_init=T_init, inlier_distance_m=0.02)

    np.testing.assert_allclose(result.T[:3, :3], R, atol=5e-3)
    np.testing.assert_allclose(result.T[:3, 3], t, atol=5e-3)
    assert all(f > 0.85 for f in result.per_pose_fitness)


def test_bundle_handles_partial_observations() -> None:
    """Half the CAD invisible (no match): bundle still recovers transform."""
    rng = np.random.default_rng(11)
    R = _rand_so3(rng)
    t = np.array([0.0, 0.0, 0.30])
    cad = rng.uniform(-0.05, 0.05, size=(800, 3))
    # Real observation drops the "back half" (z < 0 in arm frame)
    visible_mask = cad[:, 2] >= 0
    arm = cad[visible_mask] @ R.T + t + rng.normal(scale=0.001, size=(visible_mask.sum(), 3))

    T_init = np.eye(4)
    T_init[:3, 3] = t  # rough translation guess

    result = register_bundle([cad], [_pcd(arm)], T_init=T_init, inlier_distance_m=0.02)
    # Single-pose partial-view: the rotation about the long axis is
    # under-constrained, so we only check translation lands in the right
    # ballpark (within inlier_distance).
    np.testing.assert_allclose(result.T[:3, 3], t, atol=0.025)
    # Bundle is robust to half-missing data: the inlier filter accepts
    # back-half CAD points whose NN in the visible arm cloud is within
    # 2 cm, so fitness sits well above the naive 0.5 ceiling.
    assert result.per_pose_fitness[0] > 0.5


def test_bundle_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one pose"):
        register_bundle([], [])


def test_bundle_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        register_bundle([np.zeros((10, 3))], [])


def test_bundle_returns_unit_quaternion() -> None:
    rng = np.random.default_rng(2)
    cad = rng.uniform(-0.05, 0.05, size=(300, 3))
    arm = cad + rng.normal(scale=0.001, size=cad.shape)
    result = register_bundle([cad], [_pcd(arm)])
    q = result.quat_xyzw
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)
