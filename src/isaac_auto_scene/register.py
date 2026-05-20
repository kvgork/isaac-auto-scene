"""Point-cloud registration with random-restart ICP (Phase 5).

Public API
----------
RegistrationResult       — frozen dataclass: T (4x4), fitness, rmse, used_fallback
register_global_local()  — N random-rotation restart ICP + point-to-plane refine
QUALITY_GATE             — (fitness_min, rmse_max_m) from research §6

Design note
-----------
Research doc recommends FPFH+FGR for global init, but Open3D 0.18 on this
platform segfaults in ``registration_fgr_*``, ``registration_ransac_*``
and the legacy ``registration_icp``.  The tensor-API ``o3d.t.pipelines
.registration.icp`` is stable, so we use it instead.  Global init is
approximated by ``n_restarts`` random small-rotation seeds around the
centroid offset; this matches the use-case (fixed D435 mount + known
SO-101 pose) where the misalignment is bounded to a few cm / tens of
degrees.  The ``fallback`` hook is the production escape route
(GeoTransformer / TEASER++).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import open3d as o3d


QUALITY_GATE: tuple[float, float] = (0.65, 0.005)  # fitness > 0.65, RMSE < 5 mm


@dataclass(frozen=True)
class RegistrationResult:
    """Output of one registration run."""

    T: np.ndarray
    fitness: float
    inlier_rmse_m: float
    used_fallback: bool
    n_restarts: int


def _to_tpcd(pcd: o3d.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
    """Convert a legacy PointCloud to a tensor PointCloud (float32)."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    t = o3d.core.Tensor(np.ascontiguousarray(pts), o3d.core.float32)
    out = o3d.t.geometry.PointCloud(t)
    if pcd.has_normals():
        ns = np.asarray(pcd.normals, dtype=np.float32)
        out.point.normals = o3d.core.Tensor(np.ascontiguousarray(ns), o3d.core.float32)
    return out


def _voxel_down_with_normals(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: float,
) -> o3d.t.geometry.PointCloud:
    down = pcd.voxel_down_sample(voxel_size)
    if not down.has_normals():
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return _to_tpcd(down)


def _icp_tensor(
    src_t: o3d.t.geometry.PointCloud,
    tgt_t: o3d.t.geometry.PointCloud,
    init: np.ndarray,
    max_correspondence_distance: float,
    use_point_to_plane: bool,
    max_iter: int,
) -> o3d.t.pipelines.registration.RegistrationResult:
    init_t = o3d.core.Tensor(
        np.ascontiguousarray(init, dtype=np.float32), o3d.core.float32
    )
    estimator = (
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        if use_point_to_plane
        else o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    )
    crit = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iter
    )
    return o3d.t.pipelines.registration.icp(
        src_t, tgt_t, max_correspondence_distance, init_t, estimator, crit
    )


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


def register_global_local(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.005,
    n_restarts: int = 5,
    coarse_distance: float = 0.05,
    fine_distance: float = 0.01,
    fallback: Callable[
        [o3d.geometry.PointCloud, o3d.geometry.PointCloud],
        RegistrationResult,
    ] | None = None,
    fallback_fitness: float = 0.40,
) -> RegistrationResult:
    """Random-restart tensor-API ICP global init + point-to-plane refine."""
    radius_normal = voxel_size * 2.0
    src_down_t = _voxel_down_with_normals(source, voxel_size, radius_normal)
    tgt_down_t = _voxel_down_with_normals(target, voxel_size, radius_normal)

    src_pts = src_down_t.point.positions.numpy()
    tgt_pts = tgt_down_t.point.positions.numpy()
    src_centroid = src_pts.mean(axis=0)
    tgt_centroid = tgt_pts.mean(axis=0)
    t_init = (tgt_centroid - src_centroid).astype(np.float64)

    rng = np.random.default_rng(0)
    best_fitness = -1.0
    best_T: np.ndarray = np.eye(4)
    best_T[:3, 3] = t_init

    for i in range(n_restarts):
        T_try = np.eye(4)
        T_try[:3, 3] = t_init
        if i > 0:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(-0.6, 0.6))
            T_try[:3, :3] = _rodrigues(axis, angle)

        # Point-to-point first (more robust to bad normals)
        coarse = _icp_tensor(
            src_down_t, tgt_down_t, T_try, coarse_distance,
            use_point_to_plane=False, max_iter=30,
        )
        fit = float(coarse.fitness)
        if fit > best_fitness:
            best_fitness = fit
            best_T = coarse.transformation.numpy().astype(np.float64)

    if best_fitness < fallback_fitness and fallback is not None:
        return fallback(source, target)

    # Fine refinement with point-to-plane on the downsampled cloud
    fine = _icp_tensor(
        src_down_t, tgt_down_t, best_T, fine_distance,
        use_point_to_plane=True, max_iter=80,
    )

    return RegistrationResult(
        T=fine.transformation.numpy().astype(np.float64),
        fitness=float(fine.fitness),
        inlier_rmse_m=float(fine.inlier_rmse),
        used_fallback=False,
        n_restarts=n_restarts,
    )


def passes_quality_gate(result: RegistrationResult) -> bool:
    f_min, rmse_max = QUALITY_GATE
    return result.fitness >= f_min and result.inlier_rmse_m <= rmse_max
