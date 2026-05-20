"""Tests for isaac_auto_scene.register (Phase 5)."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from isaac_auto_scene.cad import assemble_pcd, load_urdf
from isaac_auto_scene.register import (
    QUALITY_GATE,
    RegistrationResult,
    passes_quality_gate,
    register_global_local,
)
from tests.fixtures.minimal_urdf import write_minimal_urdf


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _make_known_transform(angle_deg: float, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(np.array([0.0, 0.0, 1.0]), float(np.radians(angle_deg)))
    T[:3, 3] = t
    return T


def _apply_T(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """numpy-only transform — avoids Open3D copy-constructor segfault."""
    return np.ascontiguousarray(pts @ T[:3, :3].T + T[:3, 3])


def _pcd_from_np(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    return p


@pytest.fixture(scope="module")
def cad_points(tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    """Source CAD points (numpy) assembled from the minimal URDF."""
    urdf = load_urdf(write_minimal_urdf(tmp_path_factory.mktemp("urdf")))
    res = assemble_pcd(urdf, target_n_points=8_000)
    return np.ascontiguousarray(res.points, dtype=np.float64)


def test_synthetic_ground_truth_recovery(cad_points) -> None:
    """Register a transformed copy back; recovered T must be within 5 mm / 1°."""
    T_gt = _make_known_transform(15.0, np.array([0.05, -0.03, 0.02]))

    rng = np.random.default_rng(0)
    src_pts = cad_points + rng.normal(0.0, 0.0005, cad_points.shape)
    src = _pcd_from_np(src_pts)
    tgt = _pcd_from_np(_apply_T(src_pts, T_gt))

    result = register_global_local(src, tgt, voxel_size=0.005, n_restarts=3)
    T_est = result.T

    T_err = np.linalg.inv(T_gt) @ T_est
    t_err_m = float(np.linalg.norm(T_err[:3, 3]))
    cos_angle = float(np.clip((np.trace(T_err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    angle_err_deg = float(np.degrees(np.arccos(cos_angle)))

    assert t_err_m < 0.005, f"translation error {t_err_m*1000:.2f} mm > 5 mm"
    assert angle_err_deg < 1.0, f"rotation error {angle_err_deg:.2f}° > 1°"


def test_quality_gate_thresholds() -> None:
    """QUALITY_GATE matches research §6 thresholds."""
    f_min, rmse_max = QUALITY_GATE
    assert f_min == pytest.approx(0.65)
    assert rmse_max == pytest.approx(0.005)


def test_quality_gate_pass(cad_points) -> None:
    """A near-identity registration should pass the quality gate."""
    T_gt = _make_known_transform(5.0, np.array([0.01, 0.0, 0.0]))
    src = _pcd_from_np(cad_points)
    tgt = _pcd_from_np(_apply_T(cad_points, T_gt))

    result = register_global_local(src, tgt, voxel_size=0.005, n_restarts=2)
    assert passes_quality_gate(result), (
        f"quality gate failed: fitness={result.fitness}, rmse={result.inlier_rmse_m}"
    )


def test_fallback_fires_below_threshold() -> None:
    """If coarse fitness is below the threshold the fallback callable is used."""
    rng = np.random.default_rng(7)
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(rng.uniform(-1, 1, (400, 3)))
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )
    # Target = totally unrelated point cloud (FGR/ICP cannot align)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(rng.uniform(10, 11, (400, 3)))
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
    )

    fallback_called = {"v": False}

    def _stub_fallback(_src, _tgt) -> RegistrationResult:
        fallback_called["v"] = True
        return RegistrationResult(
            T=np.eye(4),
            fitness=1.0,
            inlier_rmse_m=0.0,
            used_fallback=True,
            n_restarts=0,
        )

    result = register_global_local(
        src, tgt, voxel_size=0.05, n_restarts=1,
        fallback=_stub_fallback,
        fallback_fitness=0.99,
    )
    assert fallback_called["v"], "fallback was not invoked"
    assert result.used_fallback
